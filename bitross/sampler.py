"""Parallel confidence-ordered unmasking with classifier-free guidance and hard constraints.

Each step runs one batched forward pass (conditional + text-dropped when CFG is on), draws a
color for every masked pixel, and commits the most confident fraction so the number of masked
pixels follows a cosine schedule. Extras:

  * annealed Gumbel noise on the commit order (diversity early, precision late)
  * spatial spreading: two uncertain neighbouring pixels are not committed in the same step
  * hard constraints: a global allowed-token set (palette lock) and/or per-pixel allowed sets
    (silhouette lock), plus inpainting via `init` + `keep`
  * soft color steering: an additive per-token logit bias (e.g. "prefer this material's colors")
  * optional polish passes that re-predict the least confident committed pixels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F


@dataclass
class Step:
    index: int
    total: int
    phase: str                 # "decode" | "polish"
    tokens: torch.Tensor       # (B, N) current canvas, masked = mask id
    preview: torch.Tensor      # (B, N) argmax guess everywhere
    confidence: torch.Tensor   # (B, N)
    committed: torch.Tensor    # (B, N) bool, changed this step


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k, -1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -math.inf)
    if top_p < 1.0:
        s, idx = logits.sort(-1, descending=True)
        p = s.softmax(-1)
        drop = (p.cumsum(-1) - p) > top_p
        s = s.masked_fill(drop, -math.inf)
        logits = torch.full_like(logits, -math.inf).scatter(-1, idx, s)
    return logits


def _gumbel(shape, device, gen):
    u = torch.rand(shape, device=device, generator=gen).clamp_(1e-10, 1 - 1e-10)
    return -torch.log(-torch.log(u))


@torch.no_grad()
def sample_iter(model, text: torch.Tensor, pooled: torch.Tensor, asset_type: torch.Tensor,
                tile_mode: torch.Tensor, quality: torch.Tensor, steps: int = 16, cfg_scale: float = 3.0,
                temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, randomness: float = 2.0,
                spread: bool = True, sure: float = 0.9, polish_steps: int = 0, polish_frac: float = 0.06,
                init: Optional[torch.Tensor] = None, keep: Optional[torch.Tensor] = None,
                allowed: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None,
                amp_dtype: Optional[torch.dtype] = None) -> Iterator[Step]:
    cfg = model.cfg
    dev = pooled.device
    B, N, K, M, G = pooled.shape[0], cfg.n_pixels, cfg.vocab_size, cfg.vocab_size, cfg.grid

    if init is not None and keep is not None:
        x = torch.where(keep.to(dev).bool(), init.to(dev).long(), torch.full((B, N), M, device=dev, dtype=torch.long))
    else:
        x = torch.full((B, N), M, device=dev, dtype=torch.long)
    masked = x == M
    n0 = masked.sum(1)
    conf = torch.where(masked, torch.zeros(B, N, device=dev), torch.full((B, N), math.inf, device=dev))

    ban = None
    if allowed is not None:
        allowed = allowed.to(dev).bool()
        ban = ~(allowed if allowed.dim() == 3 else allowed.view(1, 1, K).expand(B, N, K))

    guided = cfg_scale is not None and abs(cfg_scale - 1) > 1e-6
    if guided:
        text2, pooled2 = torch.cat([text, text]), torch.cat([pooled, pooled])
        meta2 = [torch.cat([v, v]) for v in (asset_type, tile_mode, quality)]
        drop2 = torch.cat([torch.zeros(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool, device=dev)])
    sc = torch.zeros(B, N, K, device=dev) if cfg.self_cond else None
    total = steps + polish_steps

    def predict(xin, sc_in):
        ctx = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype is not None else torch.autocast(dev.type, enabled=False)
        with ctx:
            if guided:
                s2 = torch.cat([sc_in, sc_in]) if sc_in is not None else None
                out = model(torch.cat([xin, xin]), text2, pooled2, *meta2, drop_text=drop2, sc_probs=s2).float()
                lc, lu = out.chunk(2)
                lg = lu + cfg_scale * (lc - lu)
            else:
                lg = model(xin, text, pooled, asset_type, tile_mode, quality, sc_probs=sc_in).float()
        if bias is not None:
            lg = lg + bias.to(dev).float()
        if ban is not None:
            lg = lg.masked_fill(ban, -math.inf)
        return lg

    for k in range(steps):
        if not masked.any():
            break
        frac = (k + 1) / steps
        logits = predict(x, sc)
        probs = logits.softmax(-1)
        if temperature <= 0:
            pick = logits.argmax(-1)
            p_pick = probs.gather(-1, pick[..., None]).squeeze(-1)
        else:
            p = _filter(logits / temperature, top_k, top_p).softmax(-1)
            pick = torch.multinomial(p.view(-1, K), 1, generator=generator).view(B, N)
            p_pick = p.gather(-1, pick[..., None]).squeeze(-1)

        left = torch.floor(n0.float() * math.cos(math.pi / 2 * frac)).long() if k < steps - 1 else torch.zeros_like(n0)
        n_masked = masked.sum(1)
        n_commit = torch.minimum((n_masked - left).clamp(min=1), n_masked)

        score = p_pick.clamp(min=1e-12).log()
        if randomness > 0:
            score = score + randomness * (1 - frac) * _gumbel(score.shape, dev, generator)
        prio = score
        if spread:
            cand = masked & (p_pick < sure)
            s = torch.where(cand, score, torch.full_like(score, -math.inf)).view(B, 1, G, G)
            peak = F.max_pool2d(s, 3, 1, 1).view(B, N)
            prio = score + ((masked & (p_pick >= sure)) | (cand & (score >= peak))).float() * 1e4
        prio = torch.where(masked, prio, torch.full_like(prio, -math.inf))
        rank = prio.argsort(1, descending=True).argsort(1)
        commit = (rank < n_commit[:, None]) & masked

        x = torch.where(commit, pick, x)
        conf = torch.where(commit, p_pick, conf)
        masked = masked & ~commit
        if sc is not None:
            sc = probs
        best_p, best = probs.max(-1)
        yield Step(k, total, "decode", x, best, torch.where(masked, best_p, conf), commit)

    n_re = max(1, int(round(polish_frac * N)))
    editable = torch.isfinite(conf)
    for j in range(polish_steps):
        c = torch.where(editable, conf, torch.full_like(conf, math.inf))
        low = (-c).topk(n_re, 1).indices
        remask = torch.zeros(B, N, dtype=torch.bool, device=dev).scatter_(1, low, True) & editable
        xin = torch.where(remask, torch.full_like(x, M), x)
        logits = predict(xin, sc)
        probs = logits.softmax(-1)
        new = logits.argmax(-1)
        x = torch.where(remask, new, x)
        conf = torch.where(remask, probs.gather(-1, new[..., None]).squeeze(-1), conf)
        if sc is not None:
            sc = probs
        yield Step(steps + j, total, "polish", x, probs.argmax(-1), conf, remask)


@torch.no_grad()
def sample(model, *args, **kwargs) -> torch.Tensor:
    x = None
    for st in sample_iter(model, *args, **kwargs):
        x = st.tokens
    if x is None:  # nothing was masked
        init, keep = kwargs.get("init"), kwargs.get("keep")
        x = init.clone() if init is not None else None
    return x
