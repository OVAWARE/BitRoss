"""Filename -> human readable name, category, and junk / non-albedo detection."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Texture maps that are not plain color art (PBR, overlays, masks, emissive layers, UI bits).
_NON_ALBEDO_SUFFIX = re.compile(
    r"_(n|s|e|mer|normal|normals|specular|spec|height|heightmap|ao|roughness|metallic|metalness|"
    r"emissive|emission|glow|glowmask|lightmap|overlay|overlays|tint|layer\d*|particle|particles|uv)$"
)
_NON_ALBEDO_ANY = re.compile(
    r"(overlay|emissive|glowmask|specular|heightmap|normalmap|_normal_|missing|debug|placeholder|template)"
)


class _NonAlbedo:
    @staticmethod
    def search(s: str) -> bool:
        return bool(_NON_ALBEDO_SUFFIX.search(s) or _NON_ALBEDO_ANY.search(s))


NON_ALBEDO = _NonAlbedo()
UI_WORDS = {"gui", "button", "slot", "scrollbar", "tab", "widget", "hud", "cursor", "arrow_left", "arrow_right"}

JUNK_TOKENS = {
    "png", "tex", "texture", "textures", "item", "items", "block", "blocks", "icon", "icons", "img", "image",
    "sprite", "new", "old", "final", "copy", "default", "base", "inv", "inventory", "model", "the",
}
FACE_TOKENS = {"top", "bottom", "side", "sides", "front", "back", "left", "right", "end", "north", "south",
               "east", "west", "up", "down", "inner", "outer", "edge", "corner", "middle", "center"}
STATE_TOKENS = re.compile(r"^(stage|age|state|frame|anim|variant|var|level|lvl|phase|tier|mode)\d*$")

# keyword -> category. The first matching keyword (longest first) wins.
CATEGORIES = {
    "sword": "weapon", "katana": "weapon", "dagger": "weapon", "rapier": "weapon", "saber": "weapon",
    "scythe": "weapon", "spear": "weapon", "halberd": "weapon", "mace": "weapon", "hammer": "tool",
    "warhammer": "weapon", "battleaxe": "weapon", "glaive": "weapon", "trident": "weapon", "staff": "magic",
    "wand": "magic", "bow": "ranged weapon", "crossbow": "ranged weapon", "arrow": "ammo", "bullet": "ammo",
    "gun": "ranged weapon", "pistol": "ranged weapon", "rifle": "ranged weapon", "shield": "armor",
    "pickaxe": "tool", "axe": "tool", "shovel": "tool", "spade": "tool", "hoe": "tool", "paxel": "tool",
    "sickle": "tool", "shears": "tool", "wrench": "tool", "drill": "tool", "chisel": "tool", "saw": "tool",
    "helmet": "armor", "chestplate": "armor", "leggings": "armor", "boots": "armor", "cap": "armor",
    "tunic": "armor", "pants": "armor", "hood": "armor", "crown": "armor", "ring": "accessory",
    "amulet": "accessory", "necklace": "accessory", "pendant": "accessory", "charm": "accessory",
    "belt": "accessory", "ingot": "material", "nugget": "material", "gem": "material", "dust": "material",
    "plate": "material", "gear": "material", "rod": "material", "wire": "material", "shard": "material",
    "crystal": "material", "chunk": "material", "raw": "material", "scrap": "material", "coin": "material",
    "ore": "ore", "log": "wood", "wood": "wood", "planks": "wood", "plank": "wood", "bark": "wood",
    "leaves": "plant", "leaf": "plant", "sapling": "plant", "flower": "plant", "mushroom": "plant",
    "vine": "plant", "grass": "plant", "seeds": "plant", "seed": "plant", "crop": "plant", "bush": "plant",
    "potion": "potion", "flask": "potion", "vial": "potion", "elixir": "potion", "bottle": "container",
    "bucket": "container", "jar": "container", "bag": "container", "pouch": "container", "chest": "container",
    "apple": "food", "bread": "food", "cake": "food", "pie": "food", "soup": "food", "stew": "food",
    "steak": "food", "meat": "food", "fish": "food", "cookie": "food", "berry": "food", "berries": "food",
    "fruit": "food", "cheese": "food", "carrot": "food", "potato": "food", "sandwich": "food", "candy": "food",
    "egg": "food", "bricks": "building", "brick": "building", "tiles": "building", "tile": "building",
    "stone": "stone", "cobblestone": "stone", "slab": "building", "stairs": "building", "pillar": "building",
    "glass": "glass", "pane": "glass", "door": "building", "trapdoor": "building", "lamp": "light",
    "lantern": "light", "torch": "light", "book": "magic", "scroll": "magic", "rune": "magic", "orb": "magic",
    "spawn_egg": "spawn egg", "disc": "music disc", "record": "music disc", "key": "misc", "card": "misc",
    "machine": "machine", "furnace": "machine", "generator": "machine", "pipe": "machine", "cable": "machine",
    "circuit": "machine", "battery": "machine", "wool": "building", "concrete": "building", "terracotta": "building",
    "sand": "terrain", "dirt": "terrain", "gravel": "terrain", "clay": "terrain", "ice": "terrain", "snow": "terrain",
}
_CAT_KEYS = sorted(CATEGORIES, key=len, reverse=True)

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def split_words(stem: str) -> List[str]:
    s = _CAMEL.sub(" ", stem)
    s = re.sub(r"[_\-\.\s]+", " ", s).strip().lower()
    return [w for w in s.split(" ") if w]


def parse_name(file_name: str, mod_slug: Optional[str] = None) -> Tuple[str, List[str], bool, bool]:
    """Returns (clean_name, face_words, meaningful, non_albedo).

    clean_name   'ruby pickaxe'  (junk/state/number tokens removed, mod-name tokens removed)
    face_words   ['top'] for e.g. 'oak_log_top' (blocks only care about this)
    meaningful   False when nothing descriptive survives ('1.png', 'top.png', 'i.png')
    non_albedo   True for normal/specular/overlay/emissive/mask maps that are not color art
    """
    stem = file_name.rsplit("/", 1)[-1]
    stem = stem[:-4] if stem.lower().endswith(".png") else stem
    low = stem.lower()
    non_albedo = bool(NON_ALBEDO.search(low)) or any(u in low.split("_") for u in UI_WORDS)
    words = split_words(stem)
    mod_words = set(split_words(mod_slug or "")) - {"the", "of", "and"}
    faces, keep = [], []
    for w in words:
        if w in FACE_TOKENS:
            faces.append(w)
        elif w in JUNK_TOKENS or STATE_TOKENS.match(w) or w.isdigit():
            continue
        elif w in mod_words and len(words) > 2:
            continue
        elif len(w) == 1 and not w.isalpha():
            continue
        else:
            keep.append(w)
    # drop leading single letters (often mod prefixes like 'b_' / 'i_')
    while keep and len(keep[0]) == 1:
        keep.pop(0)
    name = " ".join(keep)
    meaningful = any(len(w) >= 3 and w.isalpha() for w in keep)
    return name, faces, meaningful, non_albedo


def category_of(name: str) -> Optional[str]:
    padded = f" {name} "
    if "spawn egg" in name:
        return "spawn egg"
    for k in _CAT_KEYS:
        if f" {k.replace('_', ' ')} " in padded or padded.rstrip().endswith(" " + k):
            return CATEGORIES[k]
    for k in _CAT_KEYS:   # substring fallback for glued words like 'rubysword'
        if len(k) >= 4 and k in name.replace(" ", ""):
            return CATEGORIES[k]
    return None
