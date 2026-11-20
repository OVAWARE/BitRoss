"""Vectorised per-image measurements used for filtering, quality scoring and conditioning.

Everything works on batches: rgba (N, 16, 16, 4) uint8 -> dict of (N,) arrays.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

from ..palette import canonical_rgba, color_features, srgb_to_oklab


def _lab(rgba: np.ndarray) -> np.ndarray:
    n, h, w, _ = rgba.shape
    return srgb_to_oklab(rgba[..., :3].reshape(-1, 3).astype(np.float32)).reshape(n, h, w, 3)


def _components(mask: np.ndarray) -> np.ndarray:
    """Number of 8-connected components per image (N, H, W) bool."""
    try:
        from scipy import ndimage
        st = np.ones((3, 3), bool)
        return np.array([ndimage.label(m, structure=st)[1] for m in mask], np.int32)
    except ImportError:  # pragma: no cover - slow fallback
        out = []
        for m in mask:
            seen = np.zeros_like(m)
            c = 0
            for y, x in zip(*np.nonzero(m)):
                if seen[y, x]:
                    continue
                c += 1
                stack = [(y, x)]
                seen[y, x] = True
                while stack:
                    a, b = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            u, v = a + dy, b + dx
                            if 0 <= u < m.shape[0] and 0 <= v < m.shape[1] and m[u, v] and not seen[u, v]:
                                seen[u, v] = True
                                stack.append((u, v))
            out.append(c)
        return np.array(out, np.int32)


def measure(rgba: np.ndarray) -> Dict[str, np.ndarray]:
    rgba = canonical_rgba(rgba)
    n, H, W, _ = rgba.shape
    alpha = rgba[..., 3]
    vis = alpha > 0
    opaque = alpha == 255
    n_vis = vis.sum((1, 2))
    n_trans = (vis & ~opaque).sum((1, 2))
    lab = _lab(rgba)
    L = lab[..., 0]

    # distinct colors, dominant-color share
    keys = (rgba[..., 0].astype(np.uint32) << 24 | rgba[..., 1].astype(np.uint32) << 16 |
            rgba[..., 2].astype(np.uint32) << 8 | rgba[..., 3]).reshape(n, -1)
    n_colors = np.zeros(n, np.int32)
    dom_share = np.zeros(n, np.float32)
    for i in range(n):
        k = keys[i][vis[i].reshape(-1)]
        if len(k):
            _, c = np.unique(k, return_counts=True)
            n_colors[i] = len(c)
            dom_share[i] = c.max() / len(k)

    # luminance spread and chroma
    vis_f = vis.astype(np.float32)
    cnt = np.maximum(n_vis, 1)
    L_mean = (L * vis_f).sum((1, 2)) / cnt
    L_std = np.sqrt(np.maximum(((L - L_mean[:, None, None]) ** 2 * vis_f).sum((1, 2)) / cnt, 0))
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    chroma_mean = (chroma * vis_f).sum((1, 2)) / cnt

    # local noise: fraction of visible px whose color differs from all 4 neighbours
    pad = np.pad(keys.reshape(n, H, W), ((0, 0), (1, 1), (1, 1)), mode="edge")
    c = pad[:, 1:-1, 1:-1]
    same = (c == pad[:, :-2, 1:-1]) | (c == pad[:, 2:, 1:-1]) | (c == pad[:, 1:-1, :-2]) | (c == pad[:, 1:-1, 2:])
    noise = ((~same) & vis).sum((1, 2)) / cnt

    # mean perceptual neighbour difference (interior) vs. across the wrap seam -> tileability
    d_h = np.linalg.norm(lab[:, :, 1:] - lab[:, :, :-1], axis=-1).mean((1, 2))
    d_v = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=-1).mean((1, 2))
    seam_h = np.linalg.norm(lab[:, :, 0] - lab[:, :, -1], axis=-1).mean(1)
    seam_v = np.linalg.norm(lab[:, 0] - lab[:, -1], axis=-1).mean(1)
    interior = (d_h + d_v) / 2 + 1e-3
    seam_ratio = np.maximum(seam_h, seam_v) / interior

    # bounding box / coverage / centering
    rows = vis.any(2)
    cols = vis.any(1)
    has = n_vis > 0
    r0 = np.where(has, rows.argmax(1), 0)
    r1 = np.where(has, H - 1 - rows[:, ::-1].argmax(1), 0)
    c0 = np.where(has, cols.argmax(1), 0)
    c1 = np.where(has, W - 1 - cols[:, ::-1].argmax(1), 0)
    bbox_area = np.where(has, (r1 - r0 + 1) * (c1 - c0 + 1), 0)
    bbox_fill = n_vis / np.maximum(bbox_area, 1)

    # outline: share of silhouette-border pixels that are darker than the interior mean
    pv = np.pad(vis, ((0, 0), (1, 1), (1, 1)))
    interior_mask = pv[:, :-2, 1:-1] & pv[:, 2:, 1:-1] & pv[:, 1:-1, :-2] & pv[:, 1:-1, 2:] & vis
    border = vis & ~interior_mask
    nb = np.maximum(border.sum((1, 2)), 1)
    outline_dark = ((L < (L_mean[:, None, None] - 0.08)) & border).sum((1, 2)) / nb

    components = _components(vis)
    symmetry = np.zeros(n, np.float32)
    flipped = vis[:, :, ::-1]
    inter = (vis & flipped).sum((1, 2))
    union = np.maximum((vis | flipped).sum((1, 2)), 1)
    symmetry = inter / union

    return dict(
        n_visible=n_vis.astype(np.int16), n_translucent=n_trans.astype(np.int16),
        n_colors=n_colors.astype(np.int16), dominant_share=dom_share.astype(np.float32),
        lum_mean=L_mean.astype(np.float32), lum_std=L_std.astype(np.float32),
        chroma=chroma_mean.astype(np.float32), noise=noise.astype(np.float32),
        seam_ratio=seam_ratio.astype(np.float32), bbox_area=bbox_area.astype(np.int16),
        bbox_fill=bbox_fill.astype(np.float32), outline_dark=outline_dark.astype(np.float32),
        components=components.astype(np.int16), hsym=symmetry.astype(np.float32),
    )


def pixel_hash(rgba: np.ndarray) -> np.ndarray:
    """Exact-duplicate key (canonical pixels) per image -> (N,) 16-byte hex strings."""
    arr = canonical_rgba(rgba)
    return np.array([hashlib.md5(a.tobytes()).hexdigest() for a in arr])


def structure_hash(rgba: np.ndarray, levels: int = 4) -> np.ndarray:
    """Recolor-invariant key: silhouette + luminance quantised by per-image rank.

    Two textures that are the same drawing in a different material color (very common in
    Tinkers-style mods and 'chipped' variant packs) collide here, which lets us cap them."""
    arr = canonical_rgba(rgba)
    n = len(arr)
    L = _lab(arr)[..., 0]
    vis = arr[..., 3] > 0
    out = []
    for i in range(n):
        v = vis[i]
        q = np.zeros(v.shape, np.uint8)
        if v.any():
            vals = L[i][v]
            edges = np.quantile(vals, np.linspace(0, 1, levels + 1)[1:-1])
            q[v] = 1 + np.searchsorted(edges, vals).astype(np.uint8)
        out.append(hashlib.md5(q.tobytes()).hexdigest())
    return np.array(out)


# ------------------------------------------------------------------------------------------
# color naming (for captions / tags)
# ------------------------------------------------------------------------------------------

_NAMED = {
    "black": (20, 20, 22), "dark gray": (70, 70, 74), "gray": (128, 128, 128), "light gray": (190, 190, 190),
    "white": (245, 245, 245), "red": (200, 30, 35), "dark red": (120, 20, 25), "orange": (235, 125, 30),
    "yellow": (245, 215, 50), "gold": (215, 170, 50), "lime": (130, 215, 50), "green": (50, 150, 50),
    "dark green": (25, 80, 30), "teal": (30, 140, 130), "cyan": (60, 210, 220), "light blue": (120, 175, 240),
    "blue": (45, 75, 200), "navy": (25, 30, 100), "purple": (125, 50, 185), "magenta": (200, 50, 190),
    "pink": (240, 140, 180), "brown": (115, 75, 40), "dark brown": (65, 42, 25), "tan": (200, 165, 115),
    "beige": (225, 210, 170),
}
_NAMED_LIST = list(_NAMED)
_NAMED_FEATS = color_features(np.array([[*v, 255] for v in _NAMED.values()], np.uint8))[:, :3]


def dominant_color_names(rgba: np.ndarray, top: int = 3, min_share: float = 0.12):
    """Per image list of up to `top` color names covering >= min_share of visible pixels."""
    arr = canonical_rgba(rgba)
    lab = _lab(arr)
    vis = arr[..., 3] > 0
    res = []
    for i in range(len(arr)):
        f = lab[i][vis[i]]
        if len(f) == 0:
            res.append([])
            continue
        d = ((f[:, None, :] - _NAMED_FEATS[None]) ** 2).sum(-1)
        idx = d.argmin(1)
        cnt = np.bincount(idx, minlength=len(_NAMED_LIST)) / len(f)
        order = np.argsort(-cnt)
        res.append([_NAMED_LIST[j] for j in order[:top] if cnt[j] >= min_share])
    return res
