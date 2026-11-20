"""Minecraft-specific tooling: material sets, resource packs and mod asset export."""

from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# (kind, prompt template, asset type, tileable)
MATERIAL_SET: List[Tuple[str, str, str, Optional[bool]]] = [
    ("ingot", "{m} ingot", "item", False),
    ("nugget", "{m} nugget", "item", False),
    ("raw", "raw {m} chunk", "item", False),
    ("sword", "{m} sword", "item", False),
    ("pickaxe", "{m} pickaxe", "item", False),
    ("axe", "{m} axe", "item", False),
    ("shovel", "{m} shovel", "item", False),
    ("hoe", "{m} hoe", "item", False),
    ("helmet", "{m} helmet", "item", False),
    ("chestplate", "{m} chestplate", "item", False),
    ("leggings", "{m} leggings", "item", False),
    ("boots", "{m} boots", "item", False),
    ("ore", "{m} ore in stone", "block", True),
    ("block", "block of {m}", "block", True),
]
GEM_WORDS = ("ruby", "sapphire", "emerald", "diamond", "amethyst", "topaz", "jade", "opal", "crystal", "gem", "quartz")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_") or "texture"


def material_set(br, material: str, kinds: Optional[Sequence[str]] = None, seed: int = 0, candidates: int = 4,
                 steer: float = 2.5, rerank: bool = False, **gen_kw) -> Dict[str, "Image.Image"]:
    """Generate a coherent family of textures for one material.

    The base material item (ingot, or gem for gem-like materials) is generated first; its colors
    then softly steer every other piece, so the sword, armor and ore all share one palette."""
    from .pipeline import parse_colors
    rows = [r for r in MATERIAL_SET if kinds is None or r[0] in kinds]
    gem = any(w in material.lower() for w in GEM_WORDS)
    base_prompt = f"{material} gem" if gem else f"{material} ingot"
    base = br.generate(base_prompt, n=candidates, asset_type="item", tileable=False, seed=seed,
                       rerank=rerank or candidates > 1, keep_top=1, **gen_kw)[0]
    colors = parse_colors(base)
    out: Dict[str, "Image.Image"] = {}
    for i, (kind, tpl, atype, tile) in enumerate(rows):
        if kind == "ingot" and not gem:
            out[kind] = base
            continue
        prompt = tpl.format(m=material)
        if kind == "ingot" and gem:
            prompt = base_prompt
            out["gem"] = base
            continue
        out[kind] = br.generate(prompt, n=candidates, asset_type=atype, tileable=tile, seed=seed + 1 + i,
                                prefer_colors=colors, prefer_strength=steer, rerank=rerank or candidates > 1,
                                keep_top=1, **gen_kw)[0]
    return out


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def resource_pack(textures: Dict[str, "Image.Image"], kinds: Dict[str, str], description: str = "Made with BitRoss",
                  pack_format: int = 46, icon=None) -> bytes:
    """Zip that *replaces* vanilla textures. textures: {'diamond_sword': img}; kinds: {'diamond_sword': 'item'}.

    pack_format depends on the game version (e.g. 34 = 1.21-1.21.1, 46 = 1.21.4); set it for yours."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pack.mcmeta", json.dumps({"pack": {"pack_format": pack_format, "description": description}}, indent=2))
        if icon is not None:
            z.writestr("pack.png", _png(icon.resize((128, 128), 0)))
        for name, img in textures.items():
            folder = "block" if kinds.get(name) == "block" else "item"
            z.writestr(f"assets/minecraft/textures/{folder}/{slug(name)}.png", _png(img))
    return buf.getvalue()


def mod_assets(textures: Dict[str, "Image.Image"], kinds: Dict[str, str], modid: str = "mymod",
               item_definitions: bool = True, lang: bool = True) -> bytes:
    """Zip of an `assets/<modid>/` tree ready to drop into a Fabric / NeoForge / Forge mod:
    textures, item + block models, blockstates, 1.21.4+ item definitions and an en_us lang file."""
    modid = slug(modid)
    buf = io.BytesIO()
    names = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        root = f"assets/{modid}"
        for name, img in textures.items():
            s = slug(name)
            names[s] = name
            if kinds.get(name) == "block":
                z.writestr(f"{root}/textures/block/{s}.png", _png(img))
                z.writestr(f"{root}/models/block/{s}.json", json.dumps(
                    {"parent": "minecraft:block/cube_all", "textures": {"all": f"{modid}:block/{s}"}}, indent=2))
                z.writestr(f"{root}/blockstates/{s}.json", json.dumps(
                    {"variants": {"": {"model": f"{modid}:block/{s}"}}}, indent=2))
                z.writestr(f"{root}/models/item/{s}.json", json.dumps({"parent": f"{modid}:block/{s}"}, indent=2))
                model_ref = f"{modid}:block/{s}"
            else:
                z.writestr(f"{root}/textures/item/{s}.png", _png(img))
                handheld = any(t in s for t in ("sword", "pickaxe", "axe", "shovel", "hoe", "staff", "wand", "hammer"))
                z.writestr(f"{root}/models/item/{s}.json", json.dumps(
                    {"parent": "minecraft:item/handheld" if handheld else "minecraft:item/generated",
                     "textures": {"layer0": f"{modid}:item/{s}"}}, indent=2))
                model_ref = f"{modid}:item/{s}"
            if item_definitions:
                z.writestr(f"{root}/items/{s}.json", json.dumps(
                    {"model": {"type": "minecraft:model", "model": model_ref}}, indent=2))
        if lang:
            entries = {}
            for s, name in names.items():
                kind = "block" if kinds.get(name) == "block" else "item"
                entries[f"{kind}.{modid}.{s}"] = " ".join(w.capitalize() for w in s.split("_"))
            z.writestr(f"{root}/lang/en_us.json", json.dumps(entries, indent=2))
    return buf.getvalue()


def sprite_sheet(images: Sequence["Image.Image"], cols: int = 8, scale: int = 1, pad: int = 0):
    from PIL import Image
    n = len(images)
    rows = max(1, (n + cols - 1) // cols)
    w = images[0].width * scale
    sheet = Image.new("RGBA", (cols * (w + pad), rows * (w + pad)), (0, 0, 0, 0))
    for i, im in enumerate(images):
        sheet.alpha_composite(im.resize((w, w), Image.NEAREST), ((i % cols) * (w + pad), (i // cols) * (w + pad)))
    return sheet
