"""Command-line generation.

    bitross-generate "ruby sword" -n 8 --model runs/bitross-small/bitross.pt
    bitross-generate "mossy cobblestone" --block --tileable --out blocks/
    bitross-generate --prompt-file prompts.txt --rerank --keep 2
    bitross-generate "emerald sword" --silhouette my_sword.png
    bitross-generate "ruby sword" --init sword.png --strength 0.4
    bitross-generate "sword with golden hilt" --init sword.png --mask hilt_mask.png
    bitross-generate --material-set ruby --modid gemmod --out ruby_set/
"""

from __future__ import annotations

import argparse
import os

from .minecraft import material_set, mod_assets, resource_pack, slug, sprite_sheet
from .pipeline import BitRoss, upscale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--prompt-file")
    ap.add_argument("--model", default=os.environ.get("BITROSS_MODEL", "runs/bitross-small/bitross.pt"))
    ap.add_argument("--device")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--block", action="store_true")
    ap.add_argument("--tileable", action="store_true")
    ap.add_argument("--quality", default="high")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--polish", type=int, default=0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--palette", help="hex colors to lock to, or an image path")
    ap.add_argument("--prefer", help="hex colors to softly prefer")
    ap.add_argument("--max-colors", type=int)
    ap.add_argument("--init")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--mask")
    ap.add_argument("--silhouette")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--keep", type=int)
    ap.add_argument("--scale", type=int, default=1, help="also save an upscaled copy (nearest neighbour)")
    ap.add_argument("--sheet", action="store_true", help="also save a sprite sheet")
    ap.add_argument("--material-set")
    ap.add_argument("--modid", default="mymod")
    ap.add_argument("--pack-format", type=int, default=46)
    a = ap.parse_args(argv)

    br = BitRoss.from_pretrained(a.model, device=a.device)
    os.makedirs(a.out, exist_ok=True)

    if a.material_set:
        res = material_set(br, a.material_set, seed=a.seed or 0)
        names = {f"{slug(a.material_set)}_{k}": im for k, im in res.items()}
        kinds = {n: ("block" if n.endswith(("_ore", "_block")) else "item") for n in names}
        for n, im in names.items():
            im.save(os.path.join(a.out, f"{n}.png"))
        with open(os.path.join(a.out, f"{slug(a.modid)}_assets.zip"), "wb") as f:
            f.write(mod_assets(names, kinds, modid=a.modid))
        with open(os.path.join(a.out, "resource_pack.zip"), "wb") as f:
            f.write(resource_pack(names, kinds, pack_format=a.pack_format))
        print(f"wrote {len(names)} textures + mod assets + resource pack to {a.out}")
        return

    prompts = []
    if a.prompt:
        prompts.append(a.prompt)
    if a.prompt_file:
        with open(a.prompt_file) as f:
            prompts += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not prompts:
        ap.error("give a prompt, --prompt-file or --material-set")

    palette = a.palette
    if palette and os.path.exists(palette):
        from PIL import Image
        palette = Image.open(palette)
    for p in prompts:
        imgs = br.generate(p, n=a.n, asset_type="block" if a.block else "item", tileable=True if a.tileable else (False if not a.block else None),
                           quality=a.quality, steps=a.steps, cfg_scale=a.cfg, temperature=a.temperature, polish=a.polish,
                           seed=a.seed, palette=palette, prefer_colors=a.prefer, max_colors=a.max_colors,
                           init=a.init, strength=a.strength, mask=a.mask, silhouette=a.silhouette,
                           rerank=a.rerank, keep_top=a.keep)
        base = slug(p)
        for i, im in enumerate(imgs):
            im.save(os.path.join(a.out, f"{base}_{i:02d}.png"))
            if a.scale > 1:
                upscale(im, a.scale).save(os.path.join(a.out, f"{base}_{i:02d}_x{a.scale}.png"))
        if a.sheet:
            sprite_sheet(imgs, cols=min(8, len(imgs)), scale=max(1, a.scale)).save(os.path.join(a.out, f"{base}_sheet.png"))
        print(f"{p!r}: {len(imgs)} images -> {a.out}")


if __name__ == "__main__":
    main()
