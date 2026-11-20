"""Dataset preparation: raw parquet shards -> curated, tokenized training cache.

Stages (each is resumable and writes into --work):

    scan     decode every PNG, canonicalise pixels, measure features, parse filenames
    filter   hard quality filters, exact + recolor dedup, per-mod caps, quality score, tile mode
    palette  build the color palette from the kept images (OKLab weighted k-means)
    cache    tokenize kept images, assemble caption variants (optionally merging VLM labels)
    preview  contact sheets of kept / dropped images per reason (for auditing the filters)

    python -m bitross.data.build all --parquet "data/raw/data/*.parquet" --work data/work --out data/cache
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..palette import Palette, canonical_rgba, pack_rgba
from .features import dominant_color_names, measure, pixel_hash, structure_hash
from .names import category_of, parse_name

META_COLS = ["file_name", "type", "mod_slug", "author", "license", "project_id"]
GLASSY = ("glass", "ice", "water", "slime", "honey", "crystal", "gel", "bubble", "window", "pane", "jelly", "ghost")


# ------------------------------------------------------------------------------------------
# scan
# ------------------------------------------------------------------------------------------

def _decode(b: bytes) -> np.ndarray:
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(b))
        im = im.convert("RGBA")
        w, h = im.size
        if (w, h) != (16, 16):
            if w == 16 and h % 16 == 0:          # vertical animation strip -> first frame
                im = im.crop((0, 0, 16, 16))
            elif h == 16 and w % 16 == 0:
                im = im.crop((0, 0, 16, 16))
            elif w == h:
                im = im.resize((16, 16), Image.NEAREST)
            else:
                return np.zeros((16, 16, 4), np.uint8) + np.array([0, 0, 0, 1], np.uint8)  # sentinel
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return np.zeros((16, 16, 4), np.uint8) + np.array([0, 0, 0, 1], np.uint8)


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _scan_chunk(args):
    blobs, meta = args
    names = meta["file_name"].tolist()
    mods = meta["mod_slug"].fillna("").tolist() if "mod_slug" in meta else [""] * len(meta)
    px = np.stack([_decode(b) for b in blobs])
    bad = (px[..., 3] == 1).all((1, 2))
    px = canonical_rgba(px)
    feats = measure(px)
    feats["decode_error"] = bad
    feats["pixel_hash"] = pixel_hash(px)
    feats["struct_hash"] = structure_hash(px)
    parsed = [parse_name(n, m) for n, m in zip(names, mods)]
    feats["name"] = np.array([p[0] for p in parsed], object)
    feats["faces"] = np.array([" ".join(p[1]) for p in parsed], object)
    feats["meaningful"] = np.array([p[2] for p in parsed])
    feats["non_albedo"] = np.array([p[3] for p in parsed])
    feats["category"] = np.array([category_of(p[0]) or "" for p in parsed], object)
    feats["colors"] = np.array([", ".join(c) for c in dominant_color_names(px)], object)
    return px, pd.concat([meta.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def scan(parquet_glob: str, work: str, workers: int = 0, chunk: int = 2048, limit: Optional[int] = None) -> None:
    import pyarrow.parquet as pq
    os.makedirs(work, exist_ok=True)
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(parquet_glob)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    all_px, all_meta = [], []
    t0 = time.time()
    seen = 0

    def jobs():
        nonlocal seen
        for f in files:
            pf = pq.ParquetFile(f)
            cols = ["image"] + [c for c in META_COLS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=chunk, columns=cols):
                if limit is not None and seen >= limit:
                    return
                if limit is not None:
                    batch = batch.slice(0, limit - seen)
                seen += batch.num_rows
                df = batch.drop_columns(["image"]).to_pandas() if hasattr(batch, "drop_columns") else \
                    batch.to_pandas().drop(columns=["image"])
                yield [x["bytes"] for x in batch.column("image").to_pylist()], df

    with Pool(workers, initializer=_worker_init) as pool:
        for i, (px, meta) in enumerate(pool.imap(_scan_chunk, jobs())):
            all_px.append(px)
            all_meta.append(meta)
            if i % 20 == 0:
                n = sum(len(p) for p in all_px)
                print(f"  scanned {n:,} images ({n / (time.time() - t0):.0f}/s)", flush=True)
    px = np.concatenate(all_px)
    meta = pd.concat(all_meta, ignore_index=True)
    meta["source_index"] = np.arange(len(meta))
    np.save(os.path.join(work, "pixels.npy"), px)
    meta.to_parquet(os.path.join(work, "scan.parquet"))
    print(f"scan done: {len(meta):,} images in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------
# filter
# ------------------------------------------------------------------------------------------

def quality_score(m: pd.DataFrame) -> np.ndarray:
    """Heuristic 0..1 'is this good pixel art' score (used for conditioning + tie-breaking)."""
    is_item = (m["tile_class"] != "full").to_numpy()
    nc = m["n_colors"].to_numpy().astype(np.float32)
    s_col = np.clip((nc - 2) / 6, 0, 1) * np.where(nc > 64, np.clip(1 - (nc - 64) / 160, 0.3, 1), 1)
    s_shade = np.clip(m["lum_std"].to_numpy() / 0.07, 0, 1)
    s_noise = 1 - np.clip((m["noise"].to_numpy() - 0.35) / 0.45, 0, 1)
    s_dom = 1 - np.clip((m["dominant_share"].to_numpy() - 0.55) / 0.4, 0, 1)
    nv = m["n_visible"].to_numpy().astype(np.float32)
    s_cov = np.where(is_item, np.clip((nv - 12) / 40, 0, 1) * np.clip((250 - nv) / 40, 0.4, 1), 1.0)
    s_outline = np.where(is_item, 0.6 + 0.4 * np.clip(m["outline_dark"].to_numpy() / 0.5, 0, 1), 1.0)
    comp = m["components"].to_numpy().astype(np.float32)
    s_comp = np.where(is_item, np.clip(1 - (comp - 2) / 8, 0.2, 1), 1.0)
    s_name = np.where(m["meaningful"].to_numpy(), 1.0, 0.75)
    parts = np.stack([s_col, s_shade, s_noise, s_dom, s_cov, s_outline, s_comp, s_name])
    return np.exp(np.log(np.clip(parts, 0.05, 1)).mean(0)).astype(np.float32)


def filter_stage(work: str, max_per_struct: int = 6, max_per_mod: int = 4000, licenses: Optional[List[str]] = None,
                 seed: int = 0) -> pd.DataFrame:
    m = pd.read_parquet(os.path.join(work, "scan.parquet"))
    rng = np.random.default_rng(seed)
    reason = np.full(len(m), "", object)

    def drop(mask, why):
        mask = np.asarray(mask) & (reason == "")
        reason[mask] = why

    nv = m["n_visible"].to_numpy()
    full = nv >= 256
    m["tile_class"] = np.where(full, "full", "cutout")
    trans_ratio = m["n_translucent"].to_numpy() / np.maximum(nv, 1)
    glassy = m["file_name"].str.lower().str.contains("|".join(GLASSY)).to_numpy()

    drop(m["decode_error"].to_numpy(), "decode_error")
    if licenses:
        drop(~m["license"].isin(licenses).to_numpy(), "license")
    drop(m["non_albedo"].to_numpy(), "non_albedo_map")
    drop(nv < 8, "empty")
    drop((nv < 20) & (m["bbox_area"].to_numpy() < 36), "tiny_fragment")
    drop(m["n_colors"].to_numpy() <= 2, "too_few_colors")
    drop(m["dominant_share"].to_numpy() > 0.9, "flat")
    drop((trans_ratio > 0.6) & ~glassy & ~full, "translucent_overlay")
    drop((m["components"].to_numpy() > 8) & (nv < 80), "scattered")
    drop((m["noise"].to_numpy() > 0.95) & (m["n_colors"].to_numpy() > 180) & full, "noise")
    drop(full & (m["lum_std"].to_numpy() < 0.012) & (m["chroma"].to_numpy() < 0.02), "featureless")

    m["quality"] = quality_score(m)
    # prefer meaningful names, then quality, when choosing which duplicate to keep
    pref = m["meaningful"].to_numpy().astype(np.float32) * 2 + m["quality"].to_numpy() + rng.random(len(m)) * 1e-3
    order = np.argsort(-pref, kind="stable")

    alive = reason == ""
    seen_px, struct_count, mod_count = set(), Counter(), Counter()
    ph, sh = m["pixel_hash"].to_numpy(), m["struct_hash"].to_numpy()
    mods, types = m["mod_slug"].fillna("").to_numpy(), m["type"].fillna("").to_numpy()
    for i in order:
        if not alive[i]:
            continue
        if ph[i] in seen_px:
            reason[i] = "duplicate"
            continue
        seen_px.add(ph[i])
        if struct_count[sh[i]] >= max_per_struct:
            reason[i] = "recolor_cap"
            continue
        key = (mods[i], types[i])
        if mod_count[key] >= max_per_mod:
            reason[i] = "mod_cap"
            continue
        struct_count[sh[i]] += 1
        mod_count[key] += 1

    m["drop_reason"] = reason
    m["keep"] = reason == ""

    # tile mode for kept images
    tile = np.where(full & (m["seam_ratio"].to_numpy() < 1.35), "tileable", "not_tileable")
    m["tile_mode"] = tile
    # quality buckets by quantile among kept
    q = m.loc[m["keep"], "quality"].to_numpy()
    lo, hi = np.quantile(q, [0.3, 0.65]) if len(q) else (0, 1)
    m["quality_bucket"] = np.where(m["quality"] < lo, "low", np.where(m["quality"] < hi, "mid", "high"))

    # sampling weights: soften big mods, balance item/block share
    k = m["keep"].to_numpy()
    mod_sizes = m[k].groupby(["mod_slug", "type"]).size()
    sz = np.array([mod_sizes.get((a, b), 1) for a, b in zip(mods, types)], np.float32)
    w = 1.0 / np.sqrt(sz / 50.0).clip(min=1.0)
    w *= np.where(m["meaningful"].to_numpy(), 1.0, 0.5)
    w *= 0.5 + m["quality"].to_numpy()
    for t, share in (("item", 0.6), ("block", 0.4)):
        sel = k & (types == t)
        if sel.any():
            w[sel] *= share / w[sel].sum()
    m["weight"] = np.where(k, w / max(w[k].mean(), 1e-12), 0).astype(np.float32)

    m.to_parquet(os.path.join(work, "filtered.parquet"))
    counts = Counter(reason)
    summary = {
        "total": int(len(m)), "kept": int(m["keep"].sum()),
        "kept_by_type": {t: int(((types == t) & k).sum()) for t in ("item", "block")},
        "dropped": {r: int(c) for r, c in counts.most_common() if r},
        "tileable_kept": int((k & (tile == "tileable")).sum()),
        "quality_cutoffs": [float(lo), float(hi)],
    }
    with open(os.path.join(work, "filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return m


# ------------------------------------------------------------------------------------------
# palette
# ------------------------------------------------------------------------------------------

def palette_stage(work: str, size: int = 1024, n_translucent: int = 64, device: str = "cpu") -> Palette:
    m = pd.read_parquet(os.path.join(work, "filtered.parquet"), columns=["keep"])
    px = np.load(os.path.join(work, "pixels.npy"), mmap_mode="r")
    idx = np.nonzero(m["keep"].to_numpy())[0]
    keys = []
    for s in range(0, len(idx), 100_000):
        keys.append(pack_rgba(np.asarray(px[idx[s:s + 100_000]])).reshape(-1))
    keys = np.concatenate(keys)
    uniq, counts = np.unique(keys, return_counts=True)
    print(f"palette: {len(uniq):,} distinct canonical colors over {len(keys):,} pixels")
    t0 = time.time()
    pal = Palette.build(uniq, counts, size=size, n_translucent=n_translucent, device=device)
    pal.save(os.path.join(work, "palette.json"))
    sample = np.asarray(px[np.random.default_rng(0).choice(idx, min(20000, len(idx)), replace=False)])
    err, p99 = pal.quantization_error(sample)
    print(f"palette built in {time.time() - t0:.0f}s: {len(pal)} tokens, mean dE {err:.4f}, p99 dE {p99:.4f}")
    return pal


# ------------------------------------------------------------------------------------------
# cache
# ------------------------------------------------------------------------------------------

def _article_free(s: str) -> str:
    s = s.strip().rstrip(".")
    for a in ("a ", "an ", "the "):
        if s.lower().startswith(a):
            return s[len(a):]
    return s


def caption_variants(row: Dict, vlm: Optional[Dict]) -> List[str]:
    """All the ways an image can be described. The trainer samples one per step."""
    name = (vlm or {}).get("name") or (row["name"] if row["meaningful"] else "")
    name = name.strip().lower()
    colors = [c for c in (row.get("colors") or "").split(", ") if c]
    kind = "block" if row["type"] == "block" else "item"
    faces = row.get("faces") or ""
    out = []
    if vlm and vlm.get("caption"):
        out.append(_article_free(vlm["caption"]))
    if name:
        n = f"{name} ({faces})" if faces and kind == "block" else name
        out.append(n)
        if colors:
            out.append(f"{n}, {' and '.join(colors[:2])}")
            if colors[0].split()[-1] not in name:
                out.append(f"{colors[0]} {name}")
        if row.get("category"):
            out.append(f"{name}, {row['category']}")
    if not out:
        col = " and ".join(colors[:2]) or "colorful"
        cat = row.get("category") or kind
        out.append(f"{col} {cat} texture")
    # dedupe preserving order
    seen, res = set(), []
    for o in out:
        if o and o not in seen:
            seen.add(o)
            res.append(o)
    return res


def load_vlm_labels(path: Optional[str]) -> Dict[int, Dict]:
    labels: Dict[int, Dict] = {}
    if not path:
        return labels
    for p in sorted(glob.glob(path, recursive=True)):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    labels[int(d["source_index"])] = d
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return labels


def cache_stage(work: str, out: str, vlm_labels: Optional[str] = None, val_frac: float = 0.01,
                min_vlm_quality: int = 0, seed: int = 0) -> None:
    os.makedirs(out, exist_ok=True)
    m = pd.read_parquet(os.path.join(work, "filtered.parquet"))
    m = m[m["keep"]].reset_index(drop=True)
    pal = Palette.load(os.path.join(work, "palette.json"))
    labels = load_vlm_labels(vlm_labels)
    if labels:
        print(f"merging {len(labels):,} VLM labels")
        vq = m["source_index"].map(lambda i: labels.get(int(i), {}).get("quality", 5)).astype(float)
        vlm_bad = m["source_index"].map(lambda i: labels.get(int(i), {}).get("valid", True) is False).to_numpy()
        before = len(m)
        m = m[(vq >= min_vlm_quality).to_numpy() & ~vlm_bad].reset_index(drop=True)
        print(f"  VLM quality / validity filter removed {before - len(m):,}")
    px = np.load(os.path.join(work, "pixels.npy"), mmap_mode="r")
    idx = m["source_index"].to_numpy()
    tokens = np.zeros((len(m), 256), np.int16)
    for s in range(0, len(idx), 50_000):
        tokens[s:s + 50_000] = pal.encode(np.asarray(px[idx[s:s + 50_000]]))
        print(f"  tokenized {min(s + 50_000, len(idx)):,}/{len(idx):,}", flush=True)
    rows = m.to_dict("records")
    caps = [caption_variants(r, labels.get(int(r["source_index"]))) for r in rows]
    rng = np.random.default_rng(seed)
    split = np.where(rng.random(len(m)) < val_frac, "val", "train")
    meta = pd.DataFrame({
        "source_index": m["source_index"], "file_name": m["file_name"], "mod_slug": m["mod_slug"],
        "license": m.get("license"), "type": m["type"], "tile_mode": m["tile_mode"],
        "quality": m["quality"], "quality_bucket": m["quality_bucket"], "weight": m["weight"],
        "category": m["category"], "colors": m["colors"], "name": [
            (labels.get(int(r["source_index"])) or {}).get("name") or r["name"] for r in rows],
        "captions": [json.dumps(c) for c in caps], "split": split,
        "has_vlm": [bool((labels.get(int(r["source_index"])) or {}).get("caption")) for r in rows],
    })
    np.save(os.path.join(out, "tokens.npy"), tokens)
    meta.to_parquet(os.path.join(out, "meta.parquet"))
    pal.save(os.path.join(out, "palette.json"))
    info = {"n": int(len(meta)), "n_train": int((split == "train").sum()), "n_val": int((split == "val").sum()),
            "palette_size": len(pal), "grid": 16, "vlm_labels": len(labels), "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


# ------------------------------------------------------------------------------------------
# preview
# ------------------------------------------------------------------------------------------

def contact_sheet(px: np.ndarray, cols: int = 24, scale: int = 3, bg=(54, 57, 66, 255)):
    from PIL import Image
    n = len(px)
    rows = max(1, (n + cols - 1) // cols)
    cell = 16 * scale + 4
    sheet = Image.new("RGBA", (cols * cell, rows * cell), bg)
    for i, a in enumerate(px):
        im = Image.fromarray(np.asarray(a, np.uint8), "RGBA").resize((16 * scale, 16 * scale), Image.NEAREST)
        sheet.alpha_composite(im, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    return sheet


def preview_stage(work: str, per_reason: int = 96, seed: int = 0) -> None:
    m = pd.read_parquet(os.path.join(work, "filtered.parquet"), columns=["drop_reason", "keep", "quality_bucket"])
    px = np.load(os.path.join(work, "pixels.npy"), mmap_mode="r")
    rng = np.random.default_rng(seed)
    d = os.path.join(work, "preview")
    os.makedirs(d, exist_ok=True)
    groups = {f"drop_{r}": m.index[m["drop_reason"] == r] for r in m["drop_reason"].unique() if r}
    for b in ("low", "mid", "high"):
        groups[f"keep_{b}"] = m.index[m["keep"] & (m["quality_bucket"] == b)]
    for name, ix in groups.items():
        if len(ix) == 0:
            continue
        pick = np.sort(rng.choice(ix.to_numpy(), min(per_reason, len(ix)), replace=False))
        contact_sheet(np.asarray(px[pick])).save(os.path.join(d, f"{name}.png"))
    print(f"previews written to {d}")


# ------------------------------------------------------------------------------------------
# cli
# ------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["scan", "filter", "palette", "cache", "preview", "all"])
    ap.add_argument("--parquet", default="data/raw/data/*.parquet")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only scan the first N images (debugging)")
    ap.add_argument("--max-per-struct", type=int, default=6)
    ap.add_argument("--max-per-mod", type=int, default=4000)
    ap.add_argument("--licenses", nargs="*", default=None, help="keep only these license ids")
    ap.add_argument("--palette-size", type=int, default=1024)
    ap.add_argument("--translucent", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vlm-labels", default=None, help="glob of VLM label jsonl files")
    ap.add_argument("--val-frac", type=float, default=0.01)
    a = ap.parse_args(argv)
    st = a.stage
    if st in ("scan", "all"):
        scan(a.parquet, a.work, a.workers, limit=a.limit)
    if st in ("filter", "all"):
        filter_stage(a.work, a.max_per_struct, a.max_per_mod, a.licenses)
    if st in ("palette", "all"):
        palette_stage(a.work, a.palette_size, a.translucent, a.device)
    if st in ("preview", "all"):
        preview_stage(a.work)
    if st in ("cache", "all"):
        cache_stage(a.work, a.out, a.vlm_labels, a.val_frac)


if __name__ == "__main__":
    main()
