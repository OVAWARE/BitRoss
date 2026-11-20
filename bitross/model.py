"""BitRoss 2 denoiser: a bidirectional transformer that predicts palette tokens for masked pixels.

Token sequence seen by every block:   [ registers | text | image pixels ]

    16x16 pixels --(depth_in)--> 2x2 merge --(depth_mid on 8x8)--> unmerge + skip --(depth_out)--> logits

Design notes
  * Palette tokens are embedded as a learned vector plus an MLP of their OKLab+alpha color, so
    perceptually close colors start out close, and rare colors borrow strength from common ones.
  * Text (frozen CLIP token states) joins the image tokens in one attention sequence.
  * A global conditioning vector (pooled text, asset type, tile mode, quality bucket, mask ratio)
    drives adaLN-single modulation; every block learns its own offset and starts as identity.
  * 2D rotary position encoding on image tokens; the merged level uses patch-centre coordinates
    so both resolutions share one frame. QK-RMSNorm keeps attention logits well behaved.
  * The FFN gate branch gets a depthwise 3x3 conv. For tileable block textures the conv pads
    toroidally (wrap-around), so seams are seen as neighbours.
  * Self-conditioning: the previous step's predicted distribution enters masked positions as
    the expected color embedding.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


def fourier(x: torch.Tensor, n: int) -> torch.Tensor:
    """(..., C) -> (..., C*2n) sinusoidal features at octave frequencies."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=torch.float32)) * math.pi
    a = x.float().unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def rope_tables(n_prefix: int, side: int, head_dim: int, base: float, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape (n_prefix + side^2, head_dim // 2). Prefix tokens are not rotated."""
    quarter = head_dim // 4
    inv = 1.0 / (base ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    coord = torch.arange(side, dtype=torch.float64) * stride + (stride - 1) / 2
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    ang = torch.cat([yy.reshape(-1, 1) * inv, xx.reshape(-1, 1) * inv], 1)
    ang = torch.cat([torch.zeros(n_prefix, ang.shape[1], dtype=ang.dtype), ang])
    return ang.cos().float(), ang.sin().float()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x.float().chunk(2, -1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q = rotate(self.qn(q), cos, sin)
        k = rotate(self.kn(k), cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(B, N, D))


class LocalFFN(nn.Module):
    """SwiGLU; the gate branch of image tokens also sees a depthwise 3x3 neighbourhood."""

    def __init__(self, dim: int, hidden: int, conv: bool):
        super().__init__()
        self.inp = nn.Linear(dim, 2 * hidden, bias=False)
        self.outp = nn.Linear(hidden, dim, bias=False)
        self.dw = nn.Parameter(torch.zeros(hidden, 1, 3, 3)) if conv else None
        self.dw_b = nn.Parameter(torch.zeros(hidden)) if conv else None

    def forward(self, x, n_prefix: int, side: int, wrap: Optional[torch.Tensor]):
        val, gate = self.inp(x).chunk(2, -1)
        if self.dw is not None:
            B, _, C = gate.shape
            img = gate[:, n_prefix:].transpose(1, 2).reshape(B, C, side, side)
            pz = F.pad(img, (1, 1, 1, 1))
            if wrap is not None:
                pc = F.pad(img, (1, 1, 1, 1), mode="circular")
                pz = torch.where(wrap.view(B, 1, 1, 1), pc, pz)
            conv = F.conv2d(pz, self.dw.to(pz.dtype), self.dw_b.to(pz.dtype), groups=C)
            conv = conv.flatten(2).transpose(1, 2)
            gate = torch.cat([gate[:, :n_prefix], gate[:, n_prefix:] + conv], 1)
        return self.outp(val * F.silu(gate))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, affine=False)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.n2 = RMSNorm(cfg.dim, affine=False)
        self.ffn = LocalFFN(cfg.dim, cfg.hidden, cfg.local_conv)
        self.mod_offset = nn.Parameter(torch.zeros(6, cfg.dim))

    def forward(self, x, mod, cos, sin, n_prefix: int, side: int, wrap):
        sh1, sc1, g1, sh2, sc2, g2 = (mod + self.mod_offset).unsqueeze(2).unbind(1)
        x = x + g1 * self.attn(self.n1(x) * (1 + sc1) + sh1, cos, sin)
        x = x + g2 * self.ffn(self.n2(x) * (1 + sc2) + sh2, n_prefix, side, wrap)
        return x


class BitRossModel(nn.Module):
    def __init__(self, cfg: ModelConfig, palette_feats: Optional[torch.Tensor] = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        K, D, G = cfg.vocab_size, cfg.dim, cfg.grid
        if palette_feats is None:
            palette_feats = torch.zeros(K, 4)
        assert palette_feats.shape == (K, 4), f"palette {tuple(palette_feats.shape)} vs vocab {K}"
        self.register_buffer("palette_feats", palette_feats.float().clone())

        # pixel tokens
        self.tok = nn.Embedding(K + 1, D)
        n_cf = 4 + 4 * 2 * cfg.color_freqs
        self.color_mlp = nn.Sequential(nn.Linear(n_cf, D), nn.SiLU(), nn.Linear(D, D))
        self.row_emb = nn.Parameter(torch.zeros(G, D))
        self.col_emb = nn.Parameter(torch.zeros(G, D))
        self.sc_proj = nn.Linear(D, D, bias=False) if cfg.self_cond else None

        # conditioning
        self.text_in = nn.Sequential(RMSNorm(cfg.text_dim), nn.Linear(cfg.text_dim, D))
        self.null_text = nn.Parameter(torch.zeros(1, cfg.text_len, D))
        self.pool_in = nn.Sequential(RMSNorm(cfg.text_dim), nn.Linear(cfg.text_dim, D), nn.SiLU(), nn.Linear(D, D))
        self.null_pool = nn.Parameter(torch.zeros(1, D))
        self.type_emb = nn.Embedding(cfg.n_asset_types, D)
        self.tile_emb = nn.Embedding(cfg.n_tile_modes, D)
        self.qual_emb = nn.Embedding(cfg.n_quality, D)
        self.ratio_in = nn.Sequential(nn.Linear(16, D), nn.SiLU(), nn.Linear(D, D))
        self.registers = nn.Parameter(torch.zeros(1, cfg.registers, D))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))

        # trunk
        if cfg.hourglass:
            p = cfg.merge
            self.enc = nn.ModuleList(Block(cfg) for _ in range(cfg.depth_in))
            self.down = nn.Sequential(RMSNorm(p * p * D), nn.Linear(p * p * D, D))
            self.mid = nn.ModuleList(Block(cfg) for _ in range(cfg.depth_mid))
            self.up = nn.Linear(D, p * p * D)
            self.dec = nn.ModuleList(Block(cfg) for _ in range(cfg.depth_out))
        else:
            self.enc = nn.ModuleList(Block(cfg) for _ in range(cfg.depth_in + cfg.depth_mid + cfg.depth_out))
            self.mid, self.dec, self.down, self.up = nn.ModuleList(), nn.ModuleList(), None, None

        self.out_norm = RMSNorm(D, affine=False)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        self.head = nn.Linear(D, K)

        self.n_prefix = cfg.registers + cfg.text_len
        c, s = rope_tables(self.n_prefix, G, cfg.head_dim, cfg.rope_base, 1)
        self.register_buffer("cos_full", c, persistent=False)
        self.register_buffer("sin_full", s, persistent=False)
        if cfg.hourglass:
            c, s = rope_tables(self.n_prefix, G // cfg.merge, cfg.head_dim, cfg.rope_base, cfg.merge)
            self.register_buffer("cos_mid", c, persistent=False)
            self.register_buffer("sin_mid", s, persistent=False)
        self.gradient_checkpointing = False
        self._init()

    # ---------------------------------------------------------------------------------------
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for p in (self.row_emb, self.col_emb, self.null_text, self.null_pool, self.registers):
            nn.init.normal_(p, std=0.02)
        zero = [self.mod[-1], self.out_mod[-1], self.head]
        if self.up is not None:
            zero.append(self.up)
        for m in zero:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.sc_proj is not None:
            nn.init.zeros_(self.sc_proj.weight)
        for e in (self.type_emb, self.tile_emb, self.qual_emb):
            nn.init.zeros_(e.weight)

    def color_table(self) -> torch.Tensor:
        """(K+1, D) embedding of every token (palette colors + [MASK])."""
        K = self.cfg.vocab_size
        f = self.palette_feats
        x = torch.cat([f, fourier(f, self.cfg.color_freqs)], -1)
        w = self.tok.weight
        return torch.cat([w[:K] + self.color_mlp(x).to(w.dtype), w[K:]], 0)

    def _merge(self, h):
        B, N, D = h.shape
        G, p = self.cfg.grid, self.cfg.merge
        g = G // p
        return self.down(h.view(B, g, p, g, p, D).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * D))

    def _split(self, h):
        B = h.shape[0]
        G, p, D = self.cfg.grid, self.cfg.merge, self.cfg.dim
        g = G // p
        return self.up(h).view(B, g, g, p, p, D).permute(0, 1, 3, 2, 4, 5).reshape(B, G * G, D)

    def _block(self, blk, *args):
        if self.gradient_checkpointing and self.training:
            return checkpoint(blk, *args, use_reentrant=False)
        return blk(*args)

    # ---------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, text: torch.Tensor, pooled: torch.Tensor,
                asset_type: torch.Tensor, tile_mode: torch.Tensor, quality: torch.Tensor,
                drop_text: Optional[torch.Tensor] = None, sc_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          (B, N) tokens in [0, K], K = [MASK]
        text       (B, L, text_dim) frozen text-encoder states (L == cfg.text_len)
        pooled     (B, text_dim) pooled text embedding
        asset_type / tile_mode / quality   (B,) long condition ids (0 = unspecified)
        drop_text  (B,) bool: replace the text by the learned null condition (CFG)
        sc_probs   (B, N, K) previous prediction for self-conditioning
        returns    (B, N, K) float32 logits
        """
        cfg = self.cfg
        B, N = x.shape
        G, D = cfg.grid, cfg.dim
        masked = x == cfg.vocab_size
        table = self.color_table()
        h = F.embedding(x, table)
        h = h + (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(1, N, D).to(h.dtype)
        if self.sc_proj is not None and sc_probs is not None:
            exp = sc_probs.to(table.dtype) @ table[:cfg.vocab_size]
            h = h + self.sc_proj(exp) * masked.unsqueeze(-1).to(h.dtype)

        t = self.text_in(text)
        c = self.pool_in(pooled)
        if drop_text is not None:
            t = torch.where(drop_text.view(B, 1, 1), self.null_text.to(t.dtype), t)
            c = torch.where(drop_text.view(B, 1), self.null_pool.to(c.dtype), c)
        ratio = masked.float().mean(1, keepdim=True) * 2 - 1
        c = c + self.ratio_in(fourier(ratio, 8)) + self.type_emb(asset_type) + self.tile_emb(tile_mode) + self.qual_emb(quality)
        mod = self.mod(c).view(B, 6, D)
        wrap = tile_mode == 1

        n_pre = self.n_prefix
        xs = torch.cat([self.registers.expand(B, -1, -1).to(h.dtype), t.to(h.dtype), h], 1)
        for blk in self.enc:
            xs = self._block(blk, xs, mod, self.cos_full, self.sin_full, n_pre, G, wrap)
        if cfg.hourglass:
            pre, img = xs[:, :n_pre], xs[:, n_pre:]
            xs = torch.cat([pre, self._merge(img)], 1)
            for blk in self.mid:
                xs = self._block(blk, xs, mod, self.cos_mid, self.sin_mid, n_pre, G // cfg.merge, wrap)
            xs = torch.cat([xs[:, :n_pre], img + self._split(xs[:, n_pre:])], 1)
            for blk in self.dec:
                xs = self._block(blk, xs, mod, self.cos_full, self.sin_full, n_pre, G, wrap)
        img = xs[:, n_pre:]
        shift, scale = self.out_mod(c).unsqueeze(1).chunk(2, -1)
        return self.head(self.out_norm(img) * (1 + scale) + shift).float()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def hidden_matrices(self):
        """2D weights inside the transformer trunk (Muon-eligible)."""
        out = []
        for blocks in (self.enc, self.mid, self.dec):
            for blk in blocks:
                out += [blk.attn.qkv.weight, blk.attn.out.weight, blk.ffn.inp.weight, blk.ffn.outp.weight]
        return out
