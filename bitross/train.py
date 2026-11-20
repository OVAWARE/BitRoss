"""Training loop for BitRoss 2.

    python -m bitross.train --data data/cache --out runs/bitross-small --preset small --steps 60000
    python -m bitross.train --config configs/small_l4.json

Everything (tokens, metadata, tokenized captions) lives on the GPU, so the loop never waits for a
data loader. Checkpoints are written atomically and training resumes automatically from
`<out>/last.pt`, which makes it safe on preemptible Colab sessions (put --out on Google Drive).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .config import ASSET_TYPES, QUALITY_BUCKETS, TILE_MODES, TrainConfig, preset_config, ModelConfig
from .diffusion import mask_ratio, masked_loss, sample_mask
from .model import BitRossModel
from .palette import Palette
from .sampler import sample
from .text import TextEncoder, tokenize


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------

class GPUData:
    def __init__(self, data_dir: str, text_encoder: TextEncoder, device: torch.device, split: str):
        meta = pd.read_parquet(os.path.join(data_dir, "meta.parquet"))
        tokens = np.load(os.path.join(data_dir, "tokens.npy"), mmap_mode="r")
        sel = np.nonzero((meta["split"] == split).to_numpy())[0]
        meta = meta.iloc[sel].reset_index(drop=True)
        self.n = len(meta)
        self.tokens = torch.from_numpy(np.asarray(tokens[sel])).to(device)
        self.type = torch.tensor([ASSET_TYPES.index(t) if t in ASSET_TYPES else 0 for t in meta["type"]], device=device)
        self.tile = torch.tensor([TILE_MODES.index(t) if t in TILE_MODES else 0 for t in meta["tile_mode"]], device=device)
        self.qual = torch.tensor([QUALITY_BUCKETS.index(q) for q in meta["quality_bucket"]], device=device)
        w = meta["weight"].to_numpy().astype(np.float64)
        self.weight = torch.from_numpy(np.maximum(w, 1e-6)).float().to(device)
        self.has_vlm = torch.from_numpy(meta.get("has_vlm", pd.Series(False, index=meta.index)).to_numpy().astype(bool)).to(device)
        # captions: flatten every variant and tokenize once (cached on disk)
        caps = [json.loads(c) for c in meta["captions"]]
        counts = np.array([len(c) for c in caps], np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        flat = [c for cs in caps for c in cs]
        slug = text_encoder.name.replace("/", "__")
        cache = os.path.join(data_dir, f"text_ids_{split}_{slug}_{text_encoder.length}.npy")
        if os.path.exists(cache) and len(np.load(cache, mmap_mode="r")) == len(flat):
            ids = np.load(cache)
        else:
            print(f"tokenizing {len(flat):,} captions for split={split} ...", flush=True)
            ids = np.concatenate([tokenize(text_encoder.tokenizer, flat[s:s + 50000], text_encoder.length)
                                  for s in range(0, len(flat), 50000)])
            np.save(cache, ids)
        self.cap_ids = torch.from_numpy(ids).to(device)
        self.cap_start = torch.from_numpy(starts).to(device)
        self.cap_count = torch.from_numpy(counts).to(device)
        self.captions = caps
        self.device = device

    def batch(self, B: int, gen: torch.Generator, idx: Optional[torch.Tensor] = None, vlm_prob: float = 0.4) -> Dict[str, torch.Tensor]:
        if idx is None:
            idx = torch.multinomial(self.weight, B, replacement=True, generator=gen)
        cnt = self.cap_count[idx]
        j = (torch.rand(len(idx), device=self.device, generator=gen) * cnt).long().clamp(max=cnt - 1)
        use_vlm = self.has_vlm[idx] & (torch.rand(len(idx), device=self.device, generator=gen) < vlm_prob)
        j = torch.where(use_vlm, torch.zeros_like(j), j)
        return {"tokens": self.tokens[idx].long(), "type": self.type[idx], "tile": self.tile[idx],
                "qual": self.qual[idx], "text_ids": self.cap_ids[self.cap_start[idx] + j].long(), "idx": idx}


def augment(b: Dict[str, torch.Tensor], grid: int, cfg: TrainConfig, gen: torch.Generator) -> None:
    """In-place GPU augmentation. Items are never mirrored: Minecraft item art has a fixed
    handedness (tools and swords point to the top-right)."""
    x = b["tokens"].view(-1, grid, grid)
    B = x.shape[0]
    dev = x.device
    r = torch.rand(B, 3, device=dev, generator=gen)
    is_block = b["type"] == ASSET_TYPES.index("block")
    tileable = b["tile"] == TILE_MODES.index("tileable")
    flip = is_block & (r[:, 0] < cfg.hflip_prob)
    x = torch.where(flip[:, None, None], x.flip(2), x)
    roll = tileable & (r[:, 1] < cfg.roll_prob_tileable)
    if roll.any():
        dy = torch.randint(0, grid, (B,), device=dev, generator=gen) * roll
        dx = torch.randint(0, grid, (B,), device=dev, generator=gen) * roll
        ar = torch.arange(grid, device=dev)
        rows = (ar[None, :] + dy[:, None]) % grid
        cols = (ar[None, :] + dx[:, None]) % grid
        x = x.gather(1, rows[:, :, None].expand(B, grid, grid)).gather(2, cols[:, None, :].expand(B, grid, grid))
    rot = tileable & (r[:, 2] < cfg.rot_prob_blocks)
    if rot.any():
        x = torch.where(rot[:, None, None], x.transpose(1, 2).flip(2), x)
    b["tokens"] = x.reshape(B, -1).contiguous()


# ------------------------------------------------------------------------------------------
# optimisation helpers
# ------------------------------------------------------------------------------------------

def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_start = int(cfg.total_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    t = (step - decay_start) / max(1, cfg.total_steps - decay_start)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def build_optimizers(model: BitRossModel, cfg: TrainConfig):
    hidden = {id(p) for p in model.hidden_matrices()}
    decay, no_decay, muon = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in hidden and cfg.optimizer == "muon":
            muon.append(p)
        elif p.dim() >= 2 and "emb" not in n and "tok" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    opts = []
    if muon:
        try:
            opts.append(torch.optim.Muon(muon, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.95,
                                         adjust_lr_fn="match_rms_adamw"))
        except (AttributeError, TypeError):
            print("torch.optim.Muon unavailable -> using AdamW for all parameters")
            decay += muon
    fused = torch.cuda.is_available()
    opts.append(torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=cfg.lr, betas=tuple(cfg.betas), fused=fused))
    return opts


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ep = [p for p in ema.parameters()]
    mp = [p.detach() for p in model.parameters()]
    torch._foreach_lerp_(ep, mp, 1 - decay)


# ------------------------------------------------------------------------------------------
# checkpoints
# ------------------------------------------------------------------------------------------

def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def export_inference(ema: BitRossModel, palette: Palette, text_encoder: str, step: int, path: str,
                     extra: Optional[dict] = None) -> None:
    sd = {k: (v.half() if v.is_floating_point() and "palette" not in k else v) for k, v in ema.state_dict().items()}
    _atomic_save({"format": "bitross2", "model_config": ema.cfg.to_dict(), "palette": palette.to_json(),
                  "text_encoder": text_encoder, "step": step, "weights": sd, **(extra or {})}, path)


# ------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data: GPUData, te: TextEncoder, smooth, amp, n_batches: int = 4, B: int = 256) -> Dict[str, float]:
    model.eval()
    gen = torch.Generator(device=data.device).manual_seed(0)
    out = {}
    for ratio in (0.15, 0.5, 0.85, 1.0):
        nll, acc = [], []
        for i in range(n_batches):
            idx = torch.arange(i * B, min((i + 1) * B, data.n), device=data.device)
            if len(idx) == 0:
                break
            b = data.batch(len(idx), gen, idx=idx, vlm_prob=1.0)
            r = torch.full((len(idx),), ratio, device=data.device)
            m = sample_mask(b["tokens"], r, model.cfg.grid, blob_prob=0.0, generator=gen)
            x = torch.where(m, torch.full_like(b["tokens"], model.cfg.mask_id), b["tokens"])
            ts, tp = te.encode_ids(b["text_ids"])
            with amp:
                lg = model(x, ts, tp, b["type"], b["tile"], b["qual"])
            res = masked_loss(lg, b["tokens"], m)
            nll.append(res["nll"].item())
            acc.append(res["acc"].item())
        if nll:
            out[f"val_nll@{ratio}"] = float(np.mean(nll))
            out[f"val_acc@{ratio}"] = float(np.mean(acc))
    model.train()
    return out


@torch.no_grad()
def sample_grid(model, te: TextEncoder, palette: Palette, prompts, path: str, steps: int, cfg_scale: float,
                per_prompt: int = 8, amp_dtype=None, block_prompts=("cobblestone", "planks", "ore", "log", "stone", "block")):
    from PIL import Image
    model.eval()
    dev = next(model.parameters()).device
    n = len(prompts) * per_prompt
    texts = [p for p in prompts for _ in range(per_prompt)]
    ts, tp = te.encode(texts)
    is_block = torch.tensor([any(w in p for w in block_prompts) for p in texts], device=dev)
    typ = torch.where(is_block, ASSET_TYPES.index("block"), ASSET_TYPES.index("item")).long()
    tile = torch.where(is_block, TILE_MODES.index("tileable"), TILE_MODES.index("not_tileable")).long()
    qual = torch.full((n,), QUALITY_BUCKETS.index("high"), device=dev, dtype=torch.long)
    gen = torch.Generator(device=dev).manual_seed(0)
    x = sample(model, ts, tp, typ, tile, qual, steps=steps, cfg_scale=cfg_scale, generator=gen, amp_dtype=amp_dtype)
    rgba = palette.decode(x)
    S, pad = 64, 4
    sheet = Image.new("RGBA", (per_prompt * (S + pad) + pad, len(prompts) * (S + pad) + pad), (54, 57, 66, 255))
    for i, a in enumerate(rgba):
        im = Image.fromarray(a, "RGBA").resize((S, S), Image.NEAREST)
        sheet.alpha_composite(im, (pad + (i % per_prompt) * (S + pad), pad + (i // per_prompt) * (S + pad)))
    sheet.save(path)
    model.train()


# ------------------------------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "samples"), exist_ok=True)
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg.amp_dtype]
    if dev.type != "cuda":
        amp_dtype = None
    amp = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    palette = Palette.load(os.path.join(cfg.data_dir, "palette.json"))
    te = TextEncoder(cfg.text_encoder, length=preset_config(cfg.preset, **cfg.model_overrides).text_len, device=dev,
                     dtype=torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float32)
    mcfg = preset_config(cfg.preset, vocab_size=len(palette), text_dim=te.dim, **cfg.model_overrides)
    model = BitRossModel(mcfg, torch.from_numpy(palette.feats)).to(dev)
    print(f"model: {model.num_params() / 1e6:.1f}M params, vocab {mcfg.vocab_size}, device {dev}")

    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("weights")
        missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
        print(f"initialised from {cfg.init_from} (missing {len(missing)}, unexpected {len(unexpected)})")

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opts = build_optimizers(model, cfg)
    smooth = palette.neighbour_matrix(cfg.smooth_tau).to(dev) if cfg.label_smoothing > 0 else None

    step = 0
    last = os.path.join(cfg.out_dir, "last.pt")
    if cfg.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        for o, s in zip(opts, ck["optim"]):
            o.load_state_dict(s)
        step = ck["step"]
        print(f"resumed from step {step}")

    train_data = GPUData(cfg.data_dir, te, dev, "train")
    val_data = GPUData(cfg.data_dir, te, dev, "val")
    print(f"data: {train_data.n:,} train / {val_data.n:,} val")

    fwd = torch.compile(model) if (cfg.compile and dev.type == "cuda") else model
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + step)
    wandb = None
    if cfg.wandb_project:
        import wandb as _w
        wandb = _w
        wandb.init(project=cfg.wandb_project, config=cfg.to_dict(), resume="allow",
                   id=os.path.basename(os.path.normpath(cfg.out_dir)))
    log_f = open(os.path.join(cfg.out_dir, "metrics.jsonl"), "a")
    t_last, t_ckpt = time.time(), time.time()
    seen = 0
    model.train()
    B = cfg.batch_size
    while step < cfg.total_steps:
        lr = lr_at(step, cfg)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr
        stats = {}
        for _ in range(cfg.grad_accum):
            b = train_data.batch(B, gen)
            augment(b, mcfg.grid, cfg, gen)
            x0 = b["tokens"]
            u = (torch.arange(B, device=dev) + torch.rand(B, device=dev, generator=gen)) / B
            u = u[torch.randperm(B, device=dev, generator=gen)]
            m = sample_mask(x0, mask_ratio(u, cfg.mask_schedule), mcfg.grid, cfg.blob_mask_prob,
                            cfg.shape_mask_prob, generator=gen)
            x = torch.where(m, torch.full_like(x0, mcfg.mask_id), x0)
            ts, tp = te.encode_ids(b["text_ids"])
            drop = torch.rand(B, device=dev, generator=gen) < cfg.cfg_drop_text
            meta = []
            for key in ("type", "tile", "qual"):
                d = torch.rand(B, device=dev, generator=gen) < cfg.cfg_drop_meta
                meta.append(torch.where(d, torch.zeros_like(b[key]), b[key]))
            sc = None
            if mcfg.self_cond and torch.rand(1, generator=gen, device=dev).item() < cfg.self_cond_prob:
                with torch.no_grad(), amp:
                    sc = fwd(x, ts, tp, *meta, drop_text=drop).float().softmax(-1)
            with amp:
                logits = fwd(x, ts, tp, *meta, drop_text=drop, sc_probs=sc)
            res = masked_loss(logits, x0, m, smooth, cfg.label_smoothing)
            (res["loss"] / cfg.grad_accum).backward()
            for k_, v in res.items():
                stats[k_] = stats.get(k_, 0.0) + v.detach() / cfg.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        ema_update(ema, model, cfg.ema_decay if step > cfg.warmup_steps else 0.99)
        step += 1
        seen += B * cfg.grad_accum

        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t_last
            rec = {"step": step, "lr": lr, "grad_norm": float(gn), "img_per_s": seen / max(dt, 1e-9),
                   **{k_: float(v) for k_, v in stats.items()}}
            t_last, seen = time.time(), 0
            print(f"step {step:>7} | loss {rec['loss']:.4f} nll {rec['nll']:.4f} acc {rec['acc']:.3f} | "
                  f"lr {lr:.2e} gn {rec['grad_norm']:.2f} | {rec['img_per_s']:.0f} img/s", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if wandb:
                wandb.log(rec, step=step)
        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            ev = evaluate(ema, val_data, te, smooth, amp, cfg.eval_batches, cfg.eval_batch_size)
            ev["step"] = step
            print("eval", json.dumps(ev), flush=True)
            log_f.write(json.dumps(ev) + "\n")
            if wandb:
                wandb.log(ev, step=step)
        if step % cfg.sample_every == 0 or step == cfg.total_steps:
            p = os.path.join(cfg.out_dir, "samples", f"step_{step:07d}.png")
            sample_grid(ema, te, palette, cfg.sample_prompts, p, cfg.sample_steps, cfg.sample_cfg, amp_dtype=amp_dtype)
            if wandb:
                wandb.log({"samples": wandb.Image(p)}, step=step)
        due = (step % cfg.ckpt_every_steps == 0) or (time.time() - t_ckpt > cfg.ckpt_every_minutes * 60) \
            or step == cfg.total_steps
        if due:
            _atomic_save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                          "optim": [o.state_dict() for o in opts], "model_config": mcfg.to_dict(),
                          "train_config": cfg.to_dict(), "palette": palette.to_json(),
                          "text_encoder": cfg.text_encoder}, last)
            export_inference(ema, palette, cfg.text_encoder, step, os.path.join(cfg.out_dir, "bitross.pt"))
            if cfg.keep_every and step % cfg.keep_every == 0:
                export_inference(ema, palette, cfg.text_encoder, step,
                                 os.path.join(cfg.out_dir, f"bitross_step{step:07d}.pt"))
            t_ckpt = time.time()
    log_f.close()
    print("done.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with TrainConfig fields")
    ap.add_argument("--data", dest="data_dir")
    ap.add_argument("--out", dest="out_dir")
    ap.add_argument("--preset")
    ap.add_argument("--steps", dest="total_steps", type=int)
    ap.add_argument("--batch", dest="batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--init-from", dest="init_from")
    ap.add_argument("--no-compile", dest="compile", action="store_false", default=None)
    ap.add_argument("--wandb", dest="wandb_project")
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value overrides (JSON values)")
    a = ap.parse_args(argv)
    d = {}
    if a.config:
        with open(a.config) as f:
            d.update(json.load(f))
    for k in ("data_dir", "out_dir", "preset", "total_steps", "batch_size", "lr", "init_from", "compile", "wandb_project"):
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            d[k] = json.loads(v)
        except json.JSONDecodeError:
            d[k] = v
    train(TrainConfig.from_dict(d))


if __name__ == "__main__":
    main()
