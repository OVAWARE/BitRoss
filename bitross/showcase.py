"""Render a showcase of a trained checkpoint: text-to-texture grids, seamless blocks, silhouette
recolors, palette lock, variations and a full material set with mod-asset export.

    python -m bitross.showcase --model runs/bitross-small/bitross.pt --out runs/bitross-small/showcase
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ITEMS = [
    "ruby sword", "iron sword", "golden apple", "emerald pickaxe", "diamond helmet", "copper ingot",
    "health potion in a glass bottle", "raw steak", "enchanted spell book", "bow made of bone",
    "lava bucket", "blue mushroom", "magic wand with a purple crystal", "cooked salmon",
    "amethyst shard", "leather boots", "wooden shield with iron rim", "cheeseburger",
    "sword with blue blade and golden hilt", "glowing green slime ball",
]
BLOCKS = [
    "mossy cobblestone", "oak planks", "ruby ore in stone", "cherry wood log top", "sandstone bricks",
    "obsidian", "blue glazed terracotta", "snowy grass block side",
]


def grid(images, cols, scale=4, pad=4, bg=(54, 57, 66, 255)):
    from PIL import Image
    rows = (len(images) + cols - 1) // cols
    w = images[0].width * scale
    sheet = Image.new("RGBA", (cols * (w + pad) + pad, rows * (w + pad) + pad), bg)
    for i, im in enumerate(images):
        sheet.alpha_composite(im.resize((w, w), Image.NEAREST), (pad + (i % cols) * (w + pad), pad + (i // cols) * (w + pad)))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="fewer prompts (CPU smoke test)")
    a = ap.parse_args(argv)
    from .minecraft import material_set, mod_assets
    from .pipeline import BitRoss, seam_score, tile_preview

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    br = BitRoss.from_pretrained(a.model)
    items = ITEMS[:3] if a.quick else ITEMS
    blocks = BLOCKS[:2] if a.quick else BLOCKS
    report = {"model": a.model, "step": br.step}

    # 1) text -> item textures (one row per prompt), CLIP-scored
    imgs, prompts = [], []
    for i, p in enumerate(items):
        out = br.generate(p, n=a.n, asset_type="item", tileable=False, seed=i)
        imgs += out
        prompts += [p] * len(out)
    grid(imgs, a.n).save(os.path.join(a.out, "items.png"))
    try:
        sc = br.clip_scores(imgs, prompts)
        report["clip_items"] = float(np.mean(sc))
    except Exception as e:  # CLIP is optional
        report["clip_items_error"] = str(e)

    # 2) seamless blocks + 3x3 tiling previews
    bimgs, seams, tiles = [], [], []
    for i, p in enumerate(blocks):
        out = br.generate(p, n=a.n, asset_type="block", tileable=True, seed=100 + i)
        bimgs += out
        seams += [seam_score(im) for im in out]
        tiles.append(tile_preview(out[0], 3, 4))
    grid(bimgs, a.n).save(os.path.join(a.out, "blocks.png"))
    grid(tiles, min(4, len(tiles)), scale=1, pad=8).save(os.path.join(a.out, "blocks_tiled_3x3.png"))
    report["seam_score_median"] = float(np.median(seams))
    report["seam_tileable_frac"] = float(np.mean(np.array(seams) < 1.3))

    # 3) silhouette-locked recolors of one sword
    base = br.generate("iron sword", n=1, asset_type="item", tileable=False, seed=7)[0]
    rec = [base]
    for p in ["golden sword", "diamond sword", "emerald sword", "netherite sword", "ruby sword"]:
        rec += br.generate(p, n=1, asset_type="item", silhouette=base, seed=8)
    grid(rec, len(rec), scale=6).save(os.path.join(a.out, "recolor_silhouette.png"))
    a0 = np.asarray(base)[..., 3] > 0
    report["silhouette_preserved"] = bool(all((np.asarray(r)[..., 3] > 0).tolist() == a0.tolist() for r in rec[1:]))

    # 4) palette lock + variations
    pal = "#1e1b2e #3b2f5c #6a4fb3 #b48ef0 #f2e9ff #4a3420 #8a6236"
    locked = br.generate("magic staff", n=a.n, asset_type="item", palette=pal, seed=9)
    var = br.generate("iron sword", n=a.n, asset_type="item", init=base, strength=0.35, seed=10)
    grid(locked + var, a.n).save(os.path.join(a.out, "palette_lock_and_variations.png"))

    # 5) material set + mod asset export
    kinds = ["ingot", "sword", "pickaxe", "helmet", "ore", "block"] if a.quick else None
    ms = material_set(br, "ruby", kinds=kinds, seed=3, candidates=1 if a.quick else 3)
    grid(list(ms.values()), len(ms), scale=6).save(os.path.join(a.out, "material_set_ruby.png"))
    names = {f"ruby_{k}": im for k, im in ms.items()}
    knd = {n: ("block" if n.endswith(("_ore", "_block")) else "item") for n in names}
    with open(os.path.join(a.out, "ruby_mod_assets.zip"), "wb") as f:
        f.write(mod_assets(names, knd, modid="bitrossdemo"))

    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
