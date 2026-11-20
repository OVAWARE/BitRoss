"""Palette tokenizer: RGBA pixels <-> discrete color tokens.

Layout of a palette with K entries (stored as RGBA uint8, shape (K, 4)):
    token 0                       fully transparent
    tokens 1 .. K-1-n_trans       opaque colors (alpha 255)
    last n_trans tokens           translucent colors (glass, ice, water, slime ...)

Colors are clustered with frequency-weighted k-means in OKLab (a perceptually uniform space),
then each centroid is snapped to the closest color that really occurs in the data, and the
most frequent exact colors are pinned so outline blacks / pure whites survive untouched.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import torch

ALPHA_TRANSPARENT = 24      # alpha below this -> token 0
ALPHA_OPAQUE = 224          # alpha at or above this -> opaque
ALPHA_STEP = 32             # translucent alpha is quantised to this step before clustering


# ------------------------------------------------------------------------------------------
# OKLab (Ottosson 2020). numpy or torch, sRGB in [0, 255].
# ------------------------------------------------------------------------------------------

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def srgb_to_oklab(rgb) -> np.ndarray | torch.Tensor:
    if isinstance(rgb, torch.Tensor):
        x = rgb.float() / 255.0
        lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        lms = lin @ torch.tensor(_M1.T, dtype=torch.float32, device=x.device)
        lms = torch.sign(lms) * lms.abs().pow(1 / 3)
        return lms @ torch.tensor(_M2.T, dtype=torch.float32, device=x.device)
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(lin @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lms = (np.asarray(lab, dtype=np.float64) @ _M2_INV.T) ** 3
    lin = np.clip(lms @ _M1_INV.T, 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def color_features(rgba: np.ndarray) -> np.ndarray:
    """(K,4) uint8 -> (K,4) float32 [L, a, b, alpha] used for distances and embeddings."""
    lab = srgb_to_oklab(rgba[:, :3].astype(np.float32))
    alpha = rgba[:, 3:4].astype(np.float32) / 255.0
    return np.concatenate([lab, alpha], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def canonical_rgba(rgba: np.ndarray) -> np.ndarray:
    """Snap alpha into the three regimes the tokenizer understands; zero RGB of transparent px."""
    out = np.array(rgba, dtype=np.uint8, copy=True)
    a = out[..., 3].astype(np.int32)
    transparent = a < ALPHA_TRANSPARENT
    opaque = a >= ALPHA_OPAQUE
    mid = ~transparent & ~opaque
    a_q = np.clip((a // ALPHA_STEP) * ALPHA_STEP + ALPHA_STEP // 2, ALPHA_TRANSPARENT, ALPHA_OPAQUE - 1)
    out[..., 3] = np.where(opaque, 255, np.where(mid, a_q, 0)).astype(np.uint8)
    out[transparent] = 0
    return out


def pack_rgba(rgba: np.ndarray) -> np.ndarray:
    r = rgba.astype(np.uint32)
    return (r[..., 0] << 24) | (r[..., 1] << 16) | (r[..., 2] << 8) | r[..., 3]


def unpack_rgba(keys: np.ndarray) -> np.ndarray:
    k = keys.astype(np.uint32)
    return np.stack([(k >> 24) & 255, (k >> 16) & 255, (k >> 8) & 255, k & 255], -1).astype(np.uint8)


def _weighted_kmeans(x: np.ndarray, w: np.ndarray, k: int, iters: int = 30, seed: int = 0,
                     device: str = "cpu") -> np.ndarray:
    """Weighted k-means with k-means++ seeding (on a frequency-weighted subsample)."""
    if len(x) <= k:
        return x.copy()
    g = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(x).float().to(device)
    W = torch.from_numpy(w).float().to(device)
    # k-means++ seeding with sqrt-dampened weights, so rare-but-distinct hues get a chance
    pw = W.sqrt()
    first = torch.multinomial(pw.cpu(), 1, generator=g).item()
    C = X[first:first + 1]
    d2 = ((X - C) ** 2).sum(1)
    for _ in range(1, k):
        p = (pw * d2).cpu()
        if p.sum() <= 0:
            break
        idx = torch.multinomial(p, 1, generator=g).item()
        C = torch.cat([C, X[idx:idx + 1]])
        d2 = torch.minimum(d2, ((X - X[idx]) ** 2).sum(1))
    for _ in range(iters):
        assign = torch.cat([torch.cdist(X[s:s + 65536], C).argmin(1) for s in range(0, len(X), 65536)])
        sums = torch.zeros_like(C).index_add_(0, assign, X * W[:, None])
        cnt = torch.zeros(len(C), device=device).index_add_(0, assign, W)
        alive = cnt > 0
        C = torch.where(alive[:, None], sums / cnt.clamp(min=1e-9)[:, None], C)
    return C.cpu().numpy()


# ------------------------------------------------------------------------------------------
# the palette
# ------------------------------------------------------------------------------------------

class Palette:
    def __init__(self, rgba: np.ndarray, n_translucent: int = 0):
        rgba = np.asarray(rgba, dtype=np.uint8)
        assert rgba.ndim == 2 and rgba.shape[1] == 4
        assert rgba[0, 3] == 0, "token 0 must be transparent"
        self.rgba = rgba
        self.n_translucent = int(n_translucent)
        self.feats = color_features(rgba)
        self._lut_keys: Optional[np.ndarray] = None
        self._lut_vals: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.rgba)

    # ---------------- construction ----------------
    @classmethod
    def build(cls, color_keys: np.ndarray, counts: np.ndarray, size: int = 1024, n_translucent: int = 64,
              pin_top: int = 64, iters: int = 30, seed: int = 0, device: str = "cpu",
              max_points: int = 400_000) -> "Palette":
        """Build from a color histogram (packed canonical RGBA keys + pixel counts).

        Only the `max_points` most frequent colors are clustered (they cover nearly all pixels);
        every other color still gets its nearest token at encode time."""
        rgba = unpack_rgba(color_keys)
        keep = rgba[:, 3] > 0
        rgba, counts = rgba[keep], counts[keep].astype(np.float64)
        if len(rgba) > max_points:
            top = np.argsort(-counts)[:max_points]
            print(f"  clustering the {max_points:,} most frequent colors "
                  f"({counts[top].sum() / counts.sum():.1%} of pixels)")
            rgba, counts = rgba[top], counts[top]
        opaque = rgba[:, 3] == 255
        n_op = size - 1 - n_translucent

        def cluster(mask, k, alpha_weight):
            cols, w = rgba[mask], counts[mask]
            if len(cols) == 0 or k <= 0:
                return np.zeros((0, 4), np.uint8)
            f = color_features(cols)
            f[:, 3] *= alpha_weight
            # pin the most frequent exact colors
            order = np.argsort(-w)
            pinned = order[:min(pin_top, k // 4, len(cols))]
            rest = k - len(pinned)
            cents = _weighted_kmeans(f, w, rest, iters=iters, seed=seed, device=device) if rest > 0 else f[:0]
            # snap centroids to the nearest real color (with a small frequency bonus)
            Ft = torch.from_numpy(f)
            chosen = set(pinned.tolist())
            out = list(pinned)
            if len(cents):
                Ct = torch.from_numpy(cents).float()
                bonus = 0.002 * torch.from_numpy(np.log1p(w)).float()
                best_v = torch.full((len(Ct), 0), float("inf"))
                best_i = torch.zeros((len(Ct), 0), dtype=torch.long)
                for s in range(0, len(Ft), 100_000):
                    d = torch.cdist(Ct, Ft[s:s + 100_000]) - bonus[None, s:s + 100_000]
                    v, i = d.topk(min(8, d.shape[1]), dim=1, largest=False)
                    best_v = torch.cat([best_v, v], 1)
                    best_i = torch.cat([best_i, i + s], 1)
                    v, j = best_v.topk(min(8, best_v.shape[1]), dim=1, largest=False)
                    best_v, best_i = v, best_i.gather(1, j)
                for i in best_i.tolist():
                    j = next((c for c in i if c not in chosen), i[0])
                    chosen.add(j)
                    out.append(j)
            sel = cols[np.array(out)]
            sel = np.unique(sel, axis=0)
            # top up if snapping collapsed duplicates
            if len(sel) < k:
                extra = [o for o in order if o not in chosen][:k - len(sel)]
                sel = np.concatenate([sel, cols[extra]])
            return sel[:k]

        op = cluster(opaque, n_op, 1.0)
        tr = cluster(~opaque, n_translucent, 0.5)
        # order opaque colors by lightness for readability of dumps
        op = op[np.argsort(color_features(op)[:, 0], kind="stable")]
        tr = tr[np.argsort(color_features(tr)[:, 0], kind="stable")] if len(tr) else tr
        pal = np.concatenate([np.zeros((1, 4), np.uint8), op, tr])
        if len(pal) < size:  # tiny datasets: pad with duplicates of black
            pad = np.tile(np.array([[0, 0, 0, 255]], np.uint8), (size - len(pal), 1))
            pal = np.concatenate([pal[:1 + len(op)], pad, pal[1 + len(op):]])
        return cls(pal, n_translucent=len(tr))

    # ---------------- io ----------------
    def to_json(self) -> str:
        return json.dumps({"rgba": self.rgba.tolist(), "n_translucent": self.n_translucent})

    @classmethod
    def from_json(cls, s: str) -> "Palette":
        d = json.loads(s)
        return cls(np.array(d["rgba"], np.uint8), d.get("n_translucent", 0))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Palette":
        with open(path) as f:
            return cls.from_json(f.read())

    # ---------------- encode / decode ----------------
    def nearest(self, rgba: np.ndarray, chunk: int = 65536, allowed: Optional[np.ndarray] = None) -> np.ndarray:
        """Nearest palette token for each canonical RGBA color (M,4) -> (M,) int64."""
        rgba = np.asarray(rgba, np.uint8).reshape(-1, 4)
        out = np.zeros(len(rgba), np.int64)
        vis = rgba[:, 3] > 0
        if vis.any():
            f = torch.from_numpy(color_features(rgba[vis]))
            P = torch.from_numpy(self.feats)
            cand = np.arange(1, len(self)) if allowed is None else np.asarray([a for a in allowed if a != 0])
            Pc = P[cand]
            res = [torch.cdist(f[s:s + chunk], Pc).argmin(1) for s in range(0, len(f), chunk)]
            out[vis] = cand[torch.cat(res).numpy()]
        return out

    def encode(self, rgba: np.ndarray) -> np.ndarray:
        """(..., H, W, 4) uint8 -> (..., H*W) int16 tokens. Uses a growing exact-color cache."""
        arr = canonical_rgba(rgba)
        lead = arr.shape[:-3]
        keys = pack_rgba(arr).reshape(-1)
        uniq, inv = np.unique(keys, return_inverse=True)
        toks = self._lookup(uniq)
        return toks[inv].reshape(*lead, -1).astype(np.int16)

    def _lookup(self, uniq_keys: np.ndarray) -> np.ndarray:
        if self._lut_keys is None:
            self._lut_keys = np.zeros(0, np.uint32)
            self._lut_vals = np.zeros(0, np.int64)
        pos = np.searchsorted(self._lut_keys, uniq_keys)
        pos_c = np.minimum(pos, max(len(self._lut_keys) - 1, 0))
        hit = (len(self._lut_keys) > 0) & (self._lut_keys[pos_c] == uniq_keys) if len(self._lut_keys) else np.zeros(len(uniq_keys), bool)
        out = np.empty(len(uniq_keys), np.int64)
        if hit.any():
            out[hit] = self._lut_vals[pos_c[hit]]
        miss = ~hit
        if miss.any():
            new = self.nearest(unpack_rgba(uniq_keys[miss]))
            out[miss] = new
            k = np.concatenate([self._lut_keys, uniq_keys[miss]])
            v = np.concatenate([self._lut_vals, new])
            o = np.argsort(k, kind="stable")
            self._lut_keys, self._lut_vals = k[o], v[o]
            if len(self._lut_keys) > 8_000_000:   # keep memory bounded
                self._lut_keys = self._lut_vals = None
        return out

    def decode(self, tokens, grid: int = 16) -> np.ndarray:
        """(..., grid*grid) tokens -> (..., grid, grid, 4) uint8 RGBA."""
        t = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
        t = np.clip(t.astype(np.int64), 0, len(self) - 1)
        return self.rgba[t].reshape(*t.shape[:-1], grid, grid, 4)

    # ---------------- utilities for sampling-time constraints ----------------
    def restrict_to(self, rgba_colors: np.ndarray, max_delta: float = 0.0) -> np.ndarray:
        """Boolean (K,) mask of tokens allowed when the user locks a palette.

        Each requested color allows its nearest token; with max_delta > 0 every token within
        that OKLab distance of a requested color is allowed too. Token 0 is always allowed."""
        cols = canonical_rgba(np.asarray(rgba_colors, np.uint8).reshape(-1, 1, 1, 4)).reshape(-1, 4)
        allowed = np.zeros(len(self), bool)
        allowed[0] = True
        allowed[self.nearest(cols)] = True
        if max_delta > 0:
            d = torch.cdist(torch.from_numpy(color_features(cols)), torch.from_numpy(self.feats))
            allowed |= (d <= max_delta).any(0).numpy()
        return allowed

    def neighbour_matrix(self, tau: float = 0.05, k: int = 16) -> torch.Tensor:
        """Sparse-ish soft targets for perceptual label smoothing: (K, K) row-stochastic matrix
        that spreads mass to the k perceptually closest tokens (excluding self)."""
        F_ = torch.from_numpy(self.feats)
        d = torch.cdist(F_, F_)
        d.fill_diagonal_(float("inf"))
        d[0, :] = float("inf")
        d[:, 0] = float("inf")
        vals, idx = d.topk(k, dim=1, largest=False)
        w = torch.softmax(-vals / tau, dim=1)
        w = torch.where(torch.isfinite(vals), w, torch.zeros_like(w))
        M = torch.zeros(len(self), len(self))
        M.scatter_(1, idx, w)
        M[0] = 0
        M[0, 0] = 1.0  # transparency is never smoothed
        rows = M.sum(1, keepdim=True)
        M = torch.where(rows > 0, M / rows.clamp(min=1e-9), torch.eye(len(self)))
        return M

    def quantization_error(self, rgba: np.ndarray) -> Tuple[float, float]:
        """Mean and p99 OKLab error over visible pixels of (N, H, W, 4)."""
        arr = canonical_rgba(rgba)
        tok = self.encode(arr).reshape(-1)
        src = arr.reshape(-1, 4)
        vis = src[:, 3] > 0
        if not vis.any():
            return 0.0, 0.0
        e = np.linalg.norm(color_features(src[vis]) - self.feats[tok[vis]], axis=1)
        return float(e.mean()), float(np.percentile(e, 99))
