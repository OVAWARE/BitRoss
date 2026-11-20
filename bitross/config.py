"""Configuration dataclasses and model presets for BitRoss 2."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Conditioning vocabularies. Index 0 of every field means "unspecified" so the model can be
# prompted without it, and so conditioning dropout has a natural null value.
ASSET_TYPES = ["any", "item", "block"]
TILE_MODES = ["any", "tileable", "not_tileable"]
QUALITY_BUCKETS = ["any", "low", "mid", "high"]


@dataclass
class ModelConfig:
    """Architecture of the masked-diffusion pixel transformer."""

    vocab_size: int = 1024        # palette entries; token 0 = fully transparent; MASK id == vocab_size
    grid: int = 16                # canvas is grid x grid tokens
    dim: int = 512
    heads: int = 8
    depth_in: int = 2             # full-resolution blocks before the 2x2 merge
    depth_mid: int = 8            # blocks on the merged (grid/2)^2 tokens
    depth_out: int = 2            # full-resolution blocks after the unmerge
    merge: int = 2                # 1 disables the hourglass
    mlp_ratio: float = 3.0
    registers: int = 4
    text_dim: int = 512
    text_len: int = 32
    rope_base: float = 100.0
    local_conv: bool = True       # depthwise 3x3 conv inside the FFN (tile-aware padding)
    color_freqs: int = 6
    self_cond: bool = True
    n_asset_types: int = len(ASSET_TYPES)
    n_tile_modes: int = len(TILE_MODES)
    n_quality: int = len(QUALITY_BUCKETS)

    @property
    def mask_id(self) -> int:
        return self.vocab_size

    @property
    def n_pixels(self) -> int:
        return self.grid * self.grid

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden(self) -> int:
        return max(64, int(self.dim * self.mlp_ratio + 63) // 64 * 64)

    @property
    def hourglass(self) -> bool:
        return self.merge > 1 and self.depth_mid > 0

    def validate(self) -> None:
        assert self.dim % self.heads == 0
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        if self.hourglass:
            assert self.grid % self.merge == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


PRESETS: Dict[str, Dict[str, Any]] = {
    # name: overrides                                     approx params (excl. text encoder)
    "nano": dict(dim=256, heads=4, depth_in=1, depth_mid=4, depth_out=1),        # ~8M
    "small": dict(dim=512, heads=8, depth_in=2, depth_mid=8, depth_out=2),       # ~50M
    "base": dict(dim=768, heads=12, depth_in=3, depth_mid=12, depth_out=3),      # ~155M
}


def preset_config(name: str, **overrides: Any) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    kw = dict(PRESETS[name])
    kw.update(overrides)
    cfg = ModelConfig(**kw)
    cfg.validate()
    return cfg


SAMPLE_PROMPTS: List[str] = [
    "ruby sword",
    "golden apple",
    "emerald pickaxe",
    "health potion in a glass bottle",
    "copper ingot",
    "diamond helmet",
    "mossy cobblestone",
    "oak planks",
    "glowing purple crystal shard",
    "raw steak",
    "obsidian ore block",
    "enchanted spell book",
    "cherry wood log top",
    "bow made of bone",
    "blue mushroom",
    "lava bucket",
]


@dataclass
class TrainConfig:
    data_dir: str = "data/cache"
    out_dir: str = "runs/bitross-small"
    preset: str = "small"
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    text_encoder: str = "openai/clip-vit-base-patch32"
    init_from: Optional[str] = None          # fine-tune from an existing checkpoint

    # optimisation
    batch_size: int = 384
    grad_accum: int = 1
    total_steps: int = 60_000
    lr: float = 6e-4
    warmup_steps: int = 1_000
    decay_frac: float = 0.2                  # warmup-stable-decay
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    optimizer: str = "muon"                  # "muon" or "adamw"
    ema_decay: float = 0.9995

    # masked diffusion
    mask_schedule: str = "cosine"
    blob_mask_prob: float = 0.25             # spatially-correlated masks (teach inpainting)
    shape_mask_prob: float = 0.10            # mask only opaque pixels (teach silhouette-lock)
    label_smoothing: float = 0.1             # perceptual (OKLab-neighbour) label smoothing
    smooth_tau: float = 0.05
    self_cond_prob: float = 0.5
    cfg_drop_text: float = 0.1
    cfg_drop_meta: float = 0.3               # drop each metadata field independently
    hflip_prob: float = 0.5
    rot_prob_blocks: float = 0.25            # random 90-degree rotations for tileable blocks
    roll_prob_tileable: float = 0.5          # random toroidal shifts for tileable blocks
    sample_weight_power: float = 1.0         # use manifest weights (rebalancing) ^ power

    # systems
    amp_dtype: str = "bf16"
    compile: bool = True
    seed: int = 1234
    num_workers: int = 0

    # logging
    log_every: int = 50
    eval_every: int = 2_000
    eval_batches: int = 4
    eval_batch_size: int = 256
    sample_every: int = 2_000
    ckpt_every_steps: int = 2_000
    ckpt_every_minutes: float = 15.0
    keep_every: int = 20_000
    sample_steps: int = 16
    sample_cfg: float = 3.0
    sample_prompts: List[str] = field(default_factory=lambda: list(SAMPLE_PROMPTS))
    wandb_project: Optional[str] = None
    resume: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        bad = set(d) - names
        if bad:
            raise ValueError(f"unknown TrainConfig keys: {sorted(bad)}")
        return cls(**d)
