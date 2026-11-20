"""High-level inference API.

    from bitross import BitRoss
    br = BitRoss.from_pretrained("runs/bitross-small/bitross.pt")          # or an HF repo id
    imgs = br.generate("ruby sword", n=8, seed=0)
    blocks = br.generate("mossy cobblestone", asset_type="block", tileable=True)
    recolor = br.generate("emerald sword", silhouette=imgs[0])           # same shape, new material
    edit = br.generate("sword with a golden hilt", init=imgs[0], mask=hilt_mask)   # inpaint
    var = br.generate("ruby sword", init=imgs[0], strength=0.4)          # variations
    gems = br.generate("magic staff", palette=["#2b1d0e", "#7a4b23", "#43e0ff"])  # palette lock
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch

from .config import ASSET_TYPES, QUALITY_BUCKETS, TILE_MODES, ModelConfig
from .model import BitRossModel
from .palette import Palette, canonical_rgba, color_features
from .sampler import Step, sample_iter
from .text import TextEncoder

ImageLike = Union["Image.Image", np.ndarray, str]


def _to_rgba(img: ImageLike, grid: int = 16) -> np.ndarray:
    from PIL import Image
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr] * 3 + [np.full_like(arr, 255)], -1)
        if arr.shape[-1] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, arr.dtype)], -1)
        img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    img = img.convert("RGBA")
    if img.size != (grid, grid):
        img = img.resize((grid, grid), Image.NEAREST)
    return np.asarray(img, np.uint8)


def parse_colors(colors) -> np.ndarray:
    """'#ff0000', (255,0,0), [..] or an image (its distinct colors) -> (M, 4) RGBA uint8."""
    from PIL import Image
    if colors is None:
        return np.zeros((0, 4), np.uint8)
    if isinstance(colors, (Image.Image, np.ndarray)) and not (isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] in (3, 4)):
        a = canonical_rgba(_to_rgba(colors) if not isinstance(colors, Image.Image) else np.asarray(colors.convert("RGBA")))
        a = a.reshape(-1, 4)
        return np.unique(a[a[:, 3] > 0], axis=0)
    if isinstance(colors, str):
        colors = [c for c in colors.replace(",", " ").split() if c]
    out = []
    for c in colors:
        if isinstance(c, str):
            h = c.strip().lstrip("#")
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            v = [int(h[i:i + 2], 16) for i in range(0, len(h), 2)]
            out.append(v + [255] if len(v) == 3 else v[:4])
        else:
            v = list(c)
            out.append(v + [255] if len(v) == 3 else v[:4])
    return np.array(out, np.uint8).reshape(-1, 4)


class BitRoss:
    def __init__(self, model: BitRossModel, palette: Palette, text_encoder: TextEncoder, device: torch.device,
                 dtype: Optional[torch.dtype] = None, step: int = 0):
        self.model, self.palette, self.te, self.device = model, palette, text_encoder, device
        self.amp_dtype = dtype
        self.step = step
        self._clip = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None, filename: str = "bitross.pt") -> "BitRoss":
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(path, filename)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(ck["model_config"])
        palette = Palette.from_json(ck["palette"])
        sd = ck.get("weights") or ck.get("ema")
        model = BitRossModel(cfg, torch.from_numpy(palette.feats))
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        model.to(dev).eval().requires_grad_(False)
        amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
        te = TextEncoder(ck.get("text_encoder", "openai/clip-vit-base-patch32"), length=cfg.text_len, device=dev)
        return cls(model, palette, te, dev, amp, ck.get("step", 0))

    # ------------------------------------------------------------------ helpers
    def encode_image(self, img: ImageLike) -> torch.Tensor:
        return torch.from_numpy(self.palette.encode(_to_rgba(img, self.model.cfg.grid)[None])[0].astype(np.int64))

    def decode(self, tokens: torch.Tensor) -> List["Image.Image"]:
        from PIL import Image
        return [Image.fromarray(a, "RGBA") for a in self.palette.decode(tokens, self.model.cfg.grid)]

    def palette_mask(self, colors, tolerance: float = 0.0) -> torch.Tensor:
        return torch.from_numpy(self.palette.restrict_to(parse_colors(colors), tolerance))

    def color_bias(self, colors, strength: float = 2.0, sigma: float = 0.06) -> torch.Tensor:
        """Soft preference for tokens near the given colors (additive logits, (K,))."""
        ref = parse_colors(colors)
        if len(ref) == 0:
            return torch.zeros(len(self.palette))
        d = torch.cdist(torch.from_numpy(self.palette.feats), torch.from_numpy(color_features(ref)))
        b = strength * torch.exp(-(d.min(1).values ** 2) / (2 * sigma ** 2))
        b[0] = 0.0
        return b

    # ------------------------------------------------------------------ generation
    def stream(self, prompt: Union[str, Sequence[str]], n: int = 4, asset_type: str = "item",
               tileable: Optional[bool] = None, quality: str = "high", steps: int = 16, cfg_scale: float = 3.0,
               temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, randomness: float = 2.0,
               polish: int = 0, seed: Optional[int] = None, palette=None, palette_tolerance: float = 0.0,
               palette_mode: str = "soft",
               prefer_colors=None, prefer_strength: float = 2.0, init: Optional[ImageLike] = None,
               strength: float = 0.5, mask: Optional[ImageLike] = None, silhouette: Optional[ImageLike] = None,
               negative_transparent: bool = False) -> Iterator[Step]:
        """Yields a `Step` after every decoding step (live preview). See `generate` for arguments."""
        prompts = [prompt] * n if isinstance(prompt, str) else [p for p in prompt for _ in range(n)]
        B, N, K = len(prompts), self.model.cfg.n_pixels, len(self.palette)
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) if seed is not None else int(torch.randint(0, 2 ** 31, (1,)).item()))

        ts, tp = self.te.encode(prompts)
        typ = torch.full((B,), ASSET_TYPES.index(asset_type if asset_type in ASSET_TYPES else "any"), device=dev)
        if tileable is None:
            tm = "any"
        else:
            tm = "tileable" if tileable else "not_tileable"
        tile = torch.full((B,), TILE_MODES.index(tm), device=dev)
        qual = torch.full((B,), QUALITY_BUCKETS.index(quality if quality in QUALITY_BUCKETS else "any"), device=dev)

        init_t = keep = None
        has_palette = palette is not None and len(parse_colors(palette)) > 0
        allowed = self.palette_mask(palette, palette_tolerance) if has_palette and palette_mode == "hard" else None
        if allowed is not None:
            allowed = allowed.view(1, 1, K).expand(B, N, K).clone()
        if negative_transparent:     # forbid empty pixels entirely (full-tile textures)
            if allowed is None:
                allowed = torch.ones(B, N, K, dtype=torch.bool)
            allowed[..., 0] = False

        if init is not None:
            init_t = self.encode_image(init).to(dev).view(1, N).expand(B, N).clone()
            if mask is not None:
                mk = _to_rgba(mask, self.model.cfg.grid)
                regen = torch.from_numpy((mk[..., :3].mean(-1) > 127) & (mk[..., 3] > 127)).reshape(1, N).to(dev)
                keep = ~regen.expand(B, N)
            else:            # variations: re-draw a random `strength` fraction of the pixels
                r = torch.rand(B, N, device=dev, generator=gen)
                keep = r >= float(np.clip(strength, 0.02, 1.0))
        if silhouette is not None:
            sil = _to_rgba(silhouette, self.model.cfg.grid)[..., 3].reshape(1, N) > 0
            sil = torch.from_numpy(sil).to(dev).expand(B, N)
            empty = torch.zeros(B, N, dtype=torch.long, device=dev)
            if init_t is None:
                init_t, keep = empty.clone(), ~sil
            else:
                init_t = torch.where(sil, init_t, empty)
                keep = keep | ~sil
            if allowed is None:
                allowed = torch.ones(B, N, K, dtype=torch.bool)
            allowed = allowed.to(dev)
            allowed[..., 0] &= ~sil            # inside the silhouette: no holes
        bias = self.color_bias(prefer_colors, prefer_strength) if prefer_colors is not None else None
        if has_palette and palette_mode != "hard":     # steer toward the palette; exact snapping happens after decoding
            pb = self.color_bias(palette, 3.0, sigma=0.08)
            bias = pb if bias is None else bias + pb

        yield from sample_iter(self.model, ts, tp, typ, tile, qual, steps=steps, cfg_scale=cfg_scale,
                               temperature=temperature, top_k=top_k, top_p=top_p, randomness=randomness,
                               polish_steps=polish, init=init_t, keep=keep, allowed=allowed, bias=bias,
                               generator=gen, amp_dtype=self.amp_dtype)

    def generate(self, prompt: Union[str, Sequence[str]], n: int = 4, rerank: bool = False, keep_top: Optional[int] = None,
                 max_colors: Optional[int] = None, **kw) -> List["Image.Image"]:
        """Generate n images per prompt.

        asset_type   "item" | "block" | "any"
        tileable     True -> seamless block texture, False -> standalone, None -> let the model decide
        quality      "high" | "mid" | "low" | "any"   (the quality bucket the model is conditioned on)
        steps, cfg_scale, temperature, top_k, top_p, randomness (commit-order noise), polish (refine passes)
        seed         int for reproducible results
        palette      palette lock: list of hex colors / RGB tuples / a reference image. Default
                     palette_mode="soft" steers sampling toward the palette, then snaps every pixel
                     to its nearest palette color (clean shapes). "hard" masks all other colors at
                     every step (strict, but tiny palettes can break shapes).
        prefer_colors soft color steering (same formats), prefer_strength sets how hard
        init + strength      variations of an existing texture
        init + mask          inpainting: white pixels of `mask` are regenerated
        silhouette   keep this image's transparent/opaque shape and redraw everything inside it
        rerank       score candidates with CLIP and return them best-first (keep_top to truncate)
        max_colors   post-process: reduce each result to at most this many colors
        """
        last = None
        for st in self.stream(prompt, n=n, **kw):
            last = st
        if last is None:
            raise ValueError("nothing to generate (everything was kept)")
        imgs = self.decode(last.tokens)
        pal = kw.get("palette")
        if pal is not None and len(parse_colors(pal)):     # exact requested colors in both modes
            imgs = [snap_to_palette(im, pal) for im in imgs]
        if max_colors:
            imgs = [reduce_colors(im, max_colors) for im in imgs]
        if rerank:
            prompts = [prompt] * len(imgs) if isinstance(prompt, str) else [p for p in prompt for _ in range(n)]
            scores = self.clip_scores(imgs, prompts)
            order = np.argsort(-np.asarray(scores))
            imgs = [imgs[i] for i in order]
            if keep_top:
                imgs = imgs[:keep_top]
        return imgs

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def clip_scores(self, images: List["Image.Image"], prompts: List[str]) -> List[float]:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        if self._clip is None:
            name = self.te.name
            self._clip = (CLIPModel.from_pretrained(name).to(self.device).eval(), CLIPProcessor.from_pretrained(name))
        model, proc = self._clip
        rgb = []
        for im in images:
            bg = Image.new("RGBA", (224, 224), (150, 150, 150, 255))
            bg.alpha_composite(im.resize((224, 224), Image.NEAREST))
            rgb.append(bg.convert("RGB"))
        inp = proc(text=prompts, images=rgb, return_tensors="pt", padding=True).to(self.device)
        out = model(**inp)
        a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (100 * (a * b).sum(-1)).tolist()


# ---------------------------------------------------------------------- image utilities

def reduce_colors(img: "Image.Image", n: int, iters: int = 12) -> "Image.Image":
    """Reduce an RGBA image to at most n opaque colors (OKLab k-means, snapped to existing colors)."""
    from PIL import Image
    a = canonical_rgba(np.asarray(img.convert("RGBA")))
    flat = a.reshape(-1, 4)
    vis = flat[:, 3] > 0
    cols, inv, cnt = np.unique(flat[vis], axis=0, return_inverse=True, return_counts=True)
    if len(cols) <= n:
        return img
    f = color_features(cols)
    rng = np.random.default_rng(0)
    c = f[rng.choice(len(f), n, replace=False, p=cnt / cnt.sum())]
    for _ in range(iters):
        asg = ((f[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(n):
            sel = asg == k
            if sel.any():
                c[k] = (f[sel] * cnt[sel, None]).sum(0) / cnt[sel].sum()
    asg = ((f[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    # snap each cluster to its most frequent member color
    rep = np.zeros_like(cols)
    for k in range(n):
        sel = np.nonzero(asg == k)[0]
        if len(sel):
            rep[sel] = cols[sel[np.argmax(cnt[sel])]]
    out = flat.copy()
    out[vis] = rep[inv.reshape(-1)]
    return Image.fromarray(out.reshape(a.shape), "RGBA")


def snap_to_palette(img: "Image.Image", palette) -> "Image.Image":
    """Map every visible pixel to the perceptually nearest color of `palette` (OKLab)."""
    from PIL import Image
    pal = parse_colors(palette)
    pal = pal[pal[:, 3] > 0]
    a = canonical_rgba(np.asarray(img.convert("RGBA")))
    flat = a.reshape(-1, 4)
    vis = flat[:, 3] > 0
    if not vis.any() or len(pal) == 0:
        return img
    d = ((color_features(flat[vis])[:, None, :3] - color_features(pal)[None, :, :3]) ** 2).sum(-1)
    out = flat.copy()
    out[vis, :3] = pal[d.argmin(1), :3]
    return Image.fromarray(out.reshape(a.shape), "RGBA")


def upscale(img: "Image.Image", factor: int = 16) -> "Image.Image":
    from PIL import Image
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def tile_preview(img: "Image.Image", n: int = 3, factor: int = 8) -> "Image.Image":
    from PIL import Image
    t = Image.new("RGBA", (img.width * n, img.height * n))
    for y in range(n):
        for x in range(n):
            t.paste(img, (x * img.width, y * img.height))
    return upscale(t, factor)


def seam_score(img: "Image.Image") -> float:
    """< ~1.3 means the texture tiles seamlessly (ratio of seam contrast to interior contrast)."""
    from .data.features import measure
    return float(measure(np.asarray(img.convert("RGBA"))[None])["seam_ratio"][0])
