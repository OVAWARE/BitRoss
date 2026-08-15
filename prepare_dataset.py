"""Download and filter OVAWARE/16xModdedMinecraft for BitRoss training.

The Hub dataset is gated parquet (~1.03M rows, 16x16 RGBA). Columns:

    image, file_name, type, project_id, mod_slug, author, license,
    project_type, version_url

There are no text captions. `type` is `item` or `block`. We keep items whose
alpha is mixed (drop ~empty and ~fully-opaque tiles) and build CLIP captions
from the filename + mod slug.

Processed output is a Hugging Face save_to_disk cache (not 500k tiny PNGs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

HF_DATASET = "OVAWARE/16xModdedMinecraft"
ALPHA_LO = 0.01  # drop if >= 99% fully transparent
ALPHA_HI = 0.99  # drop if >= 99% opaque
ALPHA_CUTOFF = 8  # 0-255; treat below this as transparent

JUNK_NAMES = {
    "missing",
    "blank",
    "empty",
    "none",
    "null",
    "unknown",
    "untitled",
    "texture",
    "pack",
    "logo",
    "colormap",
    "color_map",
    "destroy_stage",
}


def opaque_fraction(image: Image.Image, cutoff: int = ALPHA_CUTOFF) -> float:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return float((arr[:, :, 3] > cutoff).mean())


def is_item_type(value) -> bool:
    return str(value or "").strip().lower() in {"item", "items"}


def keep_sprite(image: Image.Image, lo: float = ALPHA_LO, hi: float = ALPHA_HI) -> bool:
    """True if the sprite looks like an item (silhouette with mixed alpha)."""
    frac = opaque_fraction(image)
    return lo < frac < hi


def _pretty_name(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    stem = re.sub(r"[_./\\-]+", " ", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def caption_from_row(file_name: str, mod_slug: str = "", author: str = "") -> str:
    """Filename-derived caption — the Hub dump has no descriptions."""
    parts = ["pixel art minecraft item"]
    name = _pretty_name(file_name)
    if name and name.lower() not in JUNK_NAMES and not name.isdigit():
        parts.append(name)
    slug = _pretty_name(mod_slug)
    if slug and slug.lower() not in name.lower():
        parts.append(f"from {slug}")
    return ", ".join(parts)


def resolve_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        import io

        return Image.open(io.BytesIO(image["bytes"]))
    return Image.open(image)


def filter_batch(batch, lo=ALPHA_LO, hi=ALPHA_HI):
    keep = []
    for image, typ in zip(batch["image"], batch["type"]):
        if not is_item_type(typ):
            keep.append(False)
            continue
        try:
            img = _as_pil(image)
            keep.append(keep_sprite(img, lo=lo, hi=hi))
        except Exception:
            keep.append(False)
    return keep


def add_captions_batch(batch):
    captions = [
        caption_from_row(fn, slug or "", author or "")
        for fn, slug, author in zip(
            batch["file_name"], batch["mod_slug"], batch["author"]
        )
    ]
    return {"caption": captions}


def prepare(
    out_dir: str,
    token: str | None = None,
    max_samples: int | None = None,
    num_proc: int = 2,
    lo: float = ALPHA_LO,
    hi: float = ALPHA_HI,
):
    from datasets import load_dataset

    token = resolve_token(token)
    print(f"Loading {HF_DATASET} (gated — needs HF_TOKEN + accepted terms)...")
    ds = load_dataset(HF_DATASET, split="train", token=token)
    print(f"Raw rows: {len(ds):,}  features={list(ds.features)}")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Capped to {len(ds):,} for this run")

    n_raw = len(ds)
    ds = ds.filter(
        lambda batch: filter_batch(batch, lo=lo, hi=hi),
        batched=True,
        batch_size=512,
        num_proc=num_proc,
        desc="filter items + mixed alpha",
    )
    ds = ds.map(
        add_captions_batch,
        batched=True,
        batch_size=1024,
        desc="filename captions",
    )
    n_keep = len(ds)
    print(f"Kept {n_keep:,} / {n_raw:,} ({n_keep / max(n_raw, 1):.1%})")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed dataset to {out_path} ...")
    ds.save_to_disk(str(out_path))
    stats = {
        "source": HF_DATASET,
        "raw": n_raw,
        "kept": n_keep,
        "alpha_lo": lo,
        "alpha_hi": hi,
        "example_captions": [ds[i]["caption"] for i in range(min(8, n_keep))],
        "example_files": [ds[i]["file_name"] for i in range(min(8, n_keep))],
    }
    (out_path / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return ds


def _synthetic_image(kind: str) -> Image.Image:
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    px = img.load()
    if kind == "empty":
        return img
    if kind == "opaque":
        for y in range(16):
            for x in range(16):
                px[x, y] = (180, 40, 40, 255)
        return img
    # item-like: a small opaque plus in the middle
    for y in range(4, 12):
        for x in range(7, 9):
            px[x, y] = (40, 180, 220, 255)
    for y in range(7, 9):
        for x in range(4, 12):
            px[x, y] = (40, 180, 220, 255)
    return img


def selftest() -> None:
    empty = _synthetic_image("empty")
    opaque = _synthetic_image("opaque")
    item = _synthetic_image("item")
    assert abs(opaque_fraction(empty) - 0.0) < 1e-6, opaque_fraction(empty)
    assert opaque_fraction(opaque) > 0.99, opaque_fraction(opaque)
    frac = opaque_fraction(item)
    assert 0.01 < frac < 0.99, frac
    assert not keep_sprite(empty)
    assert not keep_sprite(opaque)
    assert keep_sprite(item)
    assert is_item_type("item") and is_item_type("Items") and not is_item_type("block")
    cap = caption_from_row("diamond_sword.png", "better-end", "someone")
    assert "diamond sword" in cap and "better end" in cap, cap
    cap2 = caption_from_row("0.png", "coolmod")
    assert "pixel art minecraft item" in cap2 and "coolmod" in cap2
    print("selftest ok", {"item_opaque_frac": frac, "caption": cap})


def parse_args():
    p = argparse.ArgumentParser(description="Prepare the BitRoss training cache from Hugging Face")
    p.add_argument("--out_dir", type=str, default="./processed-items")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_proc", type=int, default=2)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.selftest:
        selftest()
    else:
        prepare(
            args.out_dir,
            token=args.hf_token,
            max_samples=args.max_samples,
            num_proc=args.num_proc,
        )
