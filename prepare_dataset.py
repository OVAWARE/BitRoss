"""Download and filter OVAWARE/16xModdedMinecraft for BitRoss training.

The Hub dataset is gated parquet (~1.03M rows, 16x16 RGBA). Columns:

    image, file_name, type, project_id, mod_slug, author, license,
    project_type, version_url

There are no text captions. `type` is `item` or `block`. We keep items whose
alpha is mixed: drop specks that are ≥95% transparent (fewer than ~13 of 256
pixels opaque) and drop ≥99% opaque tiles. Captions come from the item
filename only (`diamond_sword.png` → `pixel art minecraft item, diamond sword`).
Mod slugs are never in the prompt.

Processed output is a Hugging Face save_to_disk cache (not 500k tiny PNGs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HF_DATASET = "OVAWARE/16xModdedMinecraft"
ALPHA_LO = 0.05  # drop specks: keep only if >5% opaque (~13 of 256 px); ~95% transparent cutoff
ALPHA_HI = 0.99  # drop near-solid tiles (>=99% opaque)
ALPHA_CUTOFF = 8  # 0-255; treat below this as transparent
IMAGE_SIZE = 16
PACK_IMAGES = "images_u8.npy"
PACK_CAPTIONS = "captions.json"

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
    "item",
    "items",
    "icon",
    "gui",
    "overlay",
    "particle",
    "layer",
    "misc",
}

# Weak one-word names: keep a few, but rank them below specific items.
WEAK_NAMES = {
    "block",
    "blocks",
    "ingot",
    "nugget",
    "dust",
    "shard",
    "gem",
    "ore",
}

QUALITY_ALPHA_LO = 0.08  # ~20 of 256 px; tighter than the 5% speck floor
QUALITY_ALPHA_HI = 0.90
MIN_OPAQUE_COLORS = 4
MIN_BBOX_FILL = 0.28
DEFAULT_MAX_IMAGES = 65536


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
    """Filename-derived caption. Item name only — do not condition on the mod."""
    del mod_slug, author
    parts = ["pixel art minecraft item"]
    name = _pretty_name(file_name)
    if name and name.lower() not in JUNK_NAMES and not name.isdigit():
        parts.append(name)
    return ", ".join(parts)


_FROM_PART = re.compile(r"^from\s+\S", re.I)
_FROM_TAIL = re.compile(r",\s*from\s+.+$", re.I)


def strip_mod_clause(caption: str) -> str:
    """Drop leftover 'from <mod>' clauses in already-packed captions.

    The first prepare wrote `pixel art minecraft item, helmet, from overworld
    reforged`. Training must never see that mod clause.
    """
    text = _FROM_TAIL.sub("", str(caption).strip())
    parts = [p.strip() for p in text.split(",") if p.strip()]
    kept = [p for p in parts if not _FROM_PART.match(p)]
    return ", ".join(kept) if kept else "pixel art minecraft item"


def caption_has_mod_clause(caption: str) -> bool:
    return strip_mod_clause(caption) != str(caption).strip()


def caption_item_name(caption: str) -> str:
    text = strip_mod_clause(caption)
    if "," not in text:
        return ""
    return text.split(",", 1)[1].strip().lower()


def _unique_opaque_colors(images: np.ndarray, cutoff: int = ALPHA_CUTOFF) -> np.ndarray:
    """Count distinct RGB colors among opaque pixels. 16x16, one pass per sprite."""
    n = images.shape[0]
    out = np.zeros(n, dtype=np.int16)
    rgb = images[..., :3]
    packed = (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    ).copy()
    packed[images[..., 3] <= cutoff] = 0xFFFFFFFF
    for i in range(n):
        uniq = np.unique(packed[i])
        out[i] = uniq.size - int(uniq[-1] == 0xFFFFFFFF)
        if i and i % 100000 == 0:
            print(f"  unique-colors {i:,}/{n:,}")
    return out


def _bbox_fill(alpha: np.ndarray) -> np.ndarray:
    """Opaque pixels / bounding-box area. Specks and 1px lines score low."""
    row_any = alpha.any(axis=2)
    col_any = alpha.any(axis=1)
    has = row_any.any(axis=1)
    first_y = row_any.argmax(axis=1)
    last_y = 15 - row_any[:, ::-1].argmax(axis=1)
    first_x = col_any.argmax(axis=1)
    last_x = 15 - col_any[:, ::-1].argmax(axis=1)
    area = (last_y - first_y + 1).clip(min=1) * (last_x - first_x + 1).clip(min=1)
    opaque_n = alpha.sum(axis=(1, 2)).astype(np.float32)
    fill = opaque_n / area.astype(np.float32)
    fill[~has] = 0.0
    return fill


def select_quality_sprites(
    images: np.ndarray,
    captions: list[str],
    max_keep: int = DEFAULT_MAX_IMAGES,
    min_colors: int = MIN_OPAQUE_COLORS,
):
    """Keep the strongest item silhouettes, drop exact dups, cap to max_keep.

    Ranking prefers colorful, compact sprites with specific names (diamond sword)
    over generic blobs (block, icon) and copy-pasted duplicates across mods.
    """
    n = len(images)
    if n == 0:
        return images, captions, {"kept": 0}
    print(f"Scoring {n:,} sprites for quality (dedup + silhouette + colors)...")
    alpha = images[..., 3] > ALPHA_CUTOFF
    opaque = alpha.mean(axis=(1, 2))
    colors = _unique_opaque_colors(images)
    fill = _bbox_fill(alpha)
    names = [caption_item_name(c) for c in captions]
    n_words = np.array(
        [len(nm.split()) if nm else 0 for nm in names], dtype=np.int16
    )
    weak = np.array(
        [(nm in WEAK_NAMES) or (nm in JUNK_NAMES) or not nm for nm in names],
        dtype=bool,
    )
    gate = (
        (opaque > QUALITY_ALPHA_LO)
        & (opaque < QUALITY_ALPHA_HI)
        & (colors >= min_colors)
        & (fill >= MIN_BBOX_FILL)
        & (~weak | (n_words >= 2))
    )
    n_gate = int(gate.sum())
    print(
        f"  silhouette gate: {n_gate:,}/{n:,} "
        f"(opaque {QUALITY_ALPHA_LO:.0%}-{QUALITY_ALPHA_HI:.0%}, "
        f">={min_colors} colors, bbox fill>={MIN_BBOX_FILL:.0%}, named)"
    )

    hashes = np.empty(n, dtype=np.uint64)
    for i in range(n):
        hashes[i] = hash(images[i].tobytes()) & 0xFFFFFFFFFFFFFFFF
    _, first = np.unique(hashes, return_index=True)
    uniq = np.zeros(n, dtype=bool)
    uniq[first] = True
    n_dup = int((~uniq).sum())
    print(f"  exact duplicates: {n_dup:,}")

    keep_mask = gate & uniq
    n_pass = int(keep_mask.sum())
    # Rank: more colors, item-like occupancy (~30%), tight bbox, specific name.
    score = (
        colors.astype(np.float32)
        + 8.0 * (1.0 - np.abs(opaque - 0.32))
        + 4.0 * fill
        + 2.0 * np.minimum(n_words, 4).astype(np.float32)
        - 3.0 * weak.astype(np.float32)
    )
    idx = np.flatnonzero(keep_mask)
    idx = idx[np.argsort(-score[idx], kind="stable")]
    cap = n_pass if max_keep is None or max_keep <= 0 else min(n_pass, int(max_keep))
    idx = np.sort(idx[:cap])
    stats = {
        "input": n,
        "duplicates": n_dup,
        "gate_pass": n_pass,
        "kept": int(idx.size),
        "max_keep": max_keep,
        "example_kept": [captions[int(i)] for i in idx[:8]],
    }
    print(
        f"Quality subset: {stats['kept']:,} sprites "
        f"(from {n:,}; dropped {n_dup:,} dups + {n - n_gate:,} weak silhouettes/"
        f"names, then top {cap:,} by score)"
    )
    for c in stats["example_kept"]:
        print(f"  {c}")
    return images[idx], [captions[int(i)] for i in idx], stats


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.copy2(tmp, path)
        tmp.unlink(missing_ok=True)


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
    return {"caption": [caption_from_row(fn) for fn in batch["file_name"]]}


def prepare(
    out_dir: str,
    token: str | None = None,
    max_samples: int | None = None,
    num_proc: int = 2,
    lo: float = ALPHA_LO,
    hi: float = ALPHA_HI,
):
    from datasets import load_dataset
    from huggingface_hub import login

    token = resolve_token(token)
    if token:
        login(token=token, add_to_git_credential=False)
    print(f"Loading {HF_DATASET} (gated — needs HF_TOKEN + accepted terms)...")
    ds = load_dataset(HF_DATASET, split="train", token=token)
    print(f"Raw rows: {len(ds):,}  features={list(ds.features)}")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Capped to {len(ds):,} for this run")

    n_raw = len(ds)
    # Type-only pass first so we never decode ~482k block textures.
    ds = ds.filter(
        lambda types: [is_item_type(t) for t in types],
        batched=True,
        batch_size=4096,
        input_columns=["type"],
        num_proc=num_proc,
        desc="keep type=item",
    )
    n_items = len(ds)
    print(f"Items: {n_items:,} / {n_raw:,}")
    ds = ds.filter(
        lambda batch: filter_batch(batch, lo=lo, hi=hi),
        batched=True,
        batch_size=512,
        num_proc=num_proc,
        desc="drop empty / fully-opaque items",
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
        "items": n_items,
        "kept": n_keep,
        "alpha_lo": lo,
        "alpha_hi": hi,
        "example_captions": [ds[i]["caption"] for i in range(min(8, n_keep))],
        "example_files": [ds[i]["file_name"] for i in range(min(8, n_keep))],
    }
    (out_path / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    write_packed_cache(ds, out_path)
    return ds


def write_packed_cache(ds, out_dir) -> None:
    """Uint8 NHWC tensor + captions so training does not decode PNG every epoch."""
    out_path = Path(out_dir)
    n = len(ds)
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    captions = []
    print(f"Packing {n:,} sprites to {out_path / PACK_IMAGES} ...")
    for i, row in enumerate(ds):
        img = _as_pil(row["image"]).convert("RGBA")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        images[i] = np.asarray(img, dtype=np.uint8)
        captions.append(strip_mod_clause(row.get("caption") or caption_from_row(row.get("file_name", ""))))
        if i and i % 100000 == 0:
            print(f"  packed {i:,}/{n:,}")
    tmp = out_path / (PACK_IMAGES + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, images)
    dest = out_path / PACK_IMAGES
    try:
        os.replace(tmp, dest)
    except OSError:
        shutil.copy2(tmp, dest)
        tmp.unlink(missing_ok=True)
    cap_tmp = out_path / (PACK_CAPTIONS + ".partial")
    cap_tmp.write_text(json.dumps(captions))
    dest_cap = out_path / PACK_CAPTIONS
    try:
        os.replace(cap_tmp, dest_cap)
    except OSError:
        shutil.copy2(cap_tmp, dest_cap)
        cap_tmp.unlink(missing_ok=True)
    print(f"Packed cache: {images.shape} {images.nbytes / 1e6:.0f} MB, {len(captions):,} captions")


def rewrite_packed_captions(out_dir) -> int:
    """Strip leftover mod clauses in captions.json / prepare_stats.json on disk."""
    cap_path = Path(out_dir) / PACK_CAPTIONS
    if not cap_path.exists():
        return 0
    raw = json.loads(cap_path.read_text())
    captions = [strip_mod_clause(c) for c in raw]
    n_mod = sum(a.strip() != b for a, b in zip(raw, captions))
    if not n_mod:
        return 0
    print(f"Rewriting {n_mod:,} packed captions in {cap_path} (dropped 'from <mod>')")
    _atomic_write_text(cap_path, json.dumps(captions))
    stats_path = Path(out_dir) / "prepare_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            stats["example_captions"] = [
                strip_mod_clause(c) for c in stats.get("example_captions") or []
            ]
            stats["mod_clauses_stripped"] = n_mod
            _atomic_write_text(stats_path, json.dumps(stats, indent=2))
        except Exception as e:
            print(f"Could not update prepare_stats.json: {e}")
    return n_mod


def load_packed_cache(out_dir):
    out_path = Path(out_dir)
    img_path = out_path / PACK_IMAGES
    cap_path = out_path / PACK_CAPTIONS
    if not img_path.exists() or not cap_path.exists():
        return None
    rewrite_packed_captions(out_path)
    images = np.load(img_path)
    captions = [strip_mod_clause(c) for c in json.loads(cap_path.read_text())]
    if len(images) != len(captions):
        raise ValueError(f"Packed cache length mismatch: {len(images)} images vs {len(captions)} captions")
    return images, captions


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
    assert 0.05 < frac < 0.99, frac
    assert not keep_sprite(empty)
    assert not keep_sprite(opaque)
    assert keep_sprite(item)
    speck = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    speck.putpixel((8, 8), (255, 0, 0, 255))
    speck.putpixel((9, 8), (255, 0, 0, 255))
    assert opaque_fraction(speck) < 0.05
    assert not keep_sprite(speck)
    assert is_item_type("item") and is_item_type("Items") and not is_item_type("block")
    cap = caption_from_row("diamond_sword.png", "better-end", "someone")
    assert "diamond sword" in cap and "better end" not in cap, cap
    cap2 = caption_from_row("0.png", "coolmod")
    assert "pixel art minecraft item" in cap2 and "coolmod" not in cap2, cap2
    stripped = strip_mod_clause("pixel art minecraft item, helmet, from overworld reforged")
    assert stripped == "pixel art minecraft item, helmet", stripped
    assert "overworld" not in stripped
    assert caption_has_mod_clause("pixel art minecraft item, helmet, from overworld reforged")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dirty = ["pixel art minecraft item, helmet, from overworld reforged"]
        (tmp_path / PACK_CAPTIONS).write_text(json.dumps(dirty))
        (tmp_path / "prepare_stats.json").write_text(
            json.dumps({"example_captions": dirty})
        )
        n = rewrite_packed_captions(tmp_path)
        assert n == 1, n
        cleaned = json.loads((tmp_path / PACK_CAPTIONS).read_text())
        assert cleaned == ["pixel art minecraft item, helmet"], cleaned
        stats = json.loads((tmp_path / "prepare_stats.json").read_text())
        assert "overworld" not in json.dumps(stats)
    # Quality rank: colorful named item beats a 2px speck and a duplicate.
    good = np.zeros((16, 16, 4), dtype=np.uint8)
    for y in range(2, 14):
        for x in range(3, 13):
            good[y, x] = (30 + 8 * x, 40 + 6 * y, 180, 255)
    speck_arr = np.zeros((16, 16, 4), dtype=np.uint8)
    speck_arr[8, 8] = (255, 0, 0, 255)
    clone = good.copy()
    imgs = np.stack([speck_arr, good, clone], axis=0)
    caps = [
        "pixel art minecraft item, speck",
        "pixel art minecraft item, diamond sword",
        "pixel art minecraft item, diamond sword",
    ]
    kept_imgs, kept_caps, q = select_quality_sprites(imgs, caps, max_keep=8)
    assert q["kept"] == 1, q
    assert kept_caps == ["pixel art minecraft item, diamond sword"], kept_caps
    assert q["duplicates"] == 1, q
    print("selftest ok", {"item_opaque_frac": frac, "caption": cap, "quality_kept": q["kept"]})


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
