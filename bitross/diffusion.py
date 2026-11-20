"""Absorbing-state (masked) discrete diffusion: corruption and training loss."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def mask_ratio(u: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "cosine":       # MaskGIT: spends more time at high mask ratios
        r = torch.cos(math.pi / 2 * u)
    elif schedule == "linear":
        r = 1 - u
    else:
        raise ValueError(schedule)
    return r.clamp(1.0 / 256, 1.0)


def _blur(noise: torch.Tensor, grid: int) -> torch.Tensor:
    """Spatially smooth white noise -> blobby fields (random sigma per sample)."""
    B = noise.shape[0]
    x = noise.view(B, 1, grid, grid)
    sig = torch.empty(B, device=noise.device).uniform_(0.8, 3.0)
    r = 4
    k = torch.arange(-r, r + 1, device=noise.device, dtype=torch.float32)
    w = torch.exp(-(k[None] ** 2) / (2 * sig[:, None] ** 2))
    w = w / w.sum(1, keepdim=True)                                  # (B, 2r+1)
    x = F.pad(x, (r, r, r, r), mode="circular")
    x = x.permute(1, 0, 2, 3)                                        # (1, B, H, W) -> grouped conv per sample
    x = F.conv2d(x, w.view(B, 1, 2 * r + 1, 1), groups=B)
    x = F.conv2d(x, w.view(B, 1, 1, 2 * r + 1), groups=B)
    return x.permute(1, 0, 2, 3).reshape(B, -1)


def sample_mask(tokens: torch.Tensor, ratio: torch.Tensor, grid: int, blob_prob: float = 0.25,
                shape_prob: float = 0.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Bool mask (B, N) with exactly round(ratio * n_eligible) masked positions per sample.

    blob masks: spatially coherent holes (teaches inpainting / region edits)
    shape masks: only non-transparent pixels are eligible and *all* of them can be masked; the
                 transparent silhouette stays visible (teaches silhouette-locked recolors)."""
    B, N = tokens.shape
    dev = tokens.device
    score = torch.rand(B, N, device=dev, generator=generator)
    blob = torch.rand(B, device=dev, generator=generator) < blob_prob
    if blob.any():
        score = torch.where(blob[:, None], _blur(score, grid).argsort(1).argsort(1).float() / N + 1e-3 * score, score)
    eligible = torch.ones(B, N, dtype=torch.bool, device=dev)
    shape = torch.rand(B, device=dev, generator=generator) < shape_prob
    if shape.any():
        opaque = tokens != 0
        eligible = torch.where(shape[:, None] & opaque.any(1, keepdim=True), opaque, eligible)
    n_el = eligible.sum(1)
    k = (ratio * n_el.float()).round().long().clamp(min=1)
    k = torch.minimum(k, n_el)
    score = torch.where(eligible, score, torch.full_like(score, 2.0))   # ineligible sort last
    ranks = score.argsort(1).argsort(1)
    return ranks < k[:, None]


def masked_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                smooth: Optional[torch.Tensor] = None, eps: float = 0.0) -> Dict[str, torch.Tensor]:
    """Cross-entropy on masked positions with optional perceptual label smoothing.

    smooth: (K, K) row-stochastic neighbour matrix; target distribution is
            (1 - eps) * onehot(y) + eps * smooth[y]."""
    lg = logits[mask]
    y = target[mask].long()
    logp = F.log_softmax(lg.float(), -1)
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    if smooth is not None and eps > 0:
        soft = -(smooth[y] * logp).sum(-1)
        loss = (1 - eps) * nll + eps * soft
    else:
        loss = nll
    with torch.no_grad():
        acc = (lg.argmax(-1) == y).float().mean()
    return {"loss": loss.mean(), "nll": nll.mean().detach(), "acc": acc}
