import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

try:
    import wandb
except ImportError:
    class _WandbStub:
        @staticmethod
        def init(*_a, **_k):
            return None

        @staticmethod
        def log(*_a, **_k):
            return None

        @staticmethod
        def watch(*_a, **_k):
            return None

        @staticmethod
        def finish(*_a, **_k):
            return None

        class Image:
            def __init__(self, *a, **k):
                pass

    wandb = _WandbStub()

from model import (
    CLIP_ID,
    IMAGE_SIZE,
    TEXT_DIM,
    BitRoss,
    flow_matching_loss,
    sample_rectified_flow,
    tokenize_prompts,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_metadata(metadata_file):
    """Accept either a JSON array or JSONL (as written by label.py / labelOllama.py)."""
    with open(metadata_file, "r") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def item_prompt(item):
    """CLIP-style caption: name + description + tags, or a precomputed caption."""
    if item.get("caption"):
        return str(item["caption"])
    name = Path(str(item.get("file_name", ""))).stem.replace("_", " ").replace("-", " ")
    desc = str(item.get("description") or "").strip()
    tags = str(item.get("tags") or "").strip()
    mod = str(item.get("mod_slug") or "").replace("_", " ").replace("-", " ").strip()
    parts = ["pixel art minecraft item"]
    if name:
        parts.append(name)
    if desc and desc.lower() != name.lower():
        parts.append(desc)
    if tags:
        parts.append(tags)
    if mod and mod.lower() not in name.lower():
        parts.append(f"from {mod}")
    return ", ".join(parts)


class ProcessedHFDataset(Dataset):
    """Rows from prepare_dataset.py (HuggingFace save_to_disk cache)."""

    def __init__(self, hf_dataset):
        self.ds = hf_dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            import io

            if isinstance(image, dict) and image.get("bytes"):
                image = Image.open(io.BytesIO(image["bytes"]))
            else:
                image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
        return self.transform(image.convert("RGBA")), item_prompt(row)


class PackedSpriteDataset(Dataset):
    """In-RAM uint8 sprites + pre-tokenized CLIP ids. No PIL in the train loop."""

    def __init__(self, images_u8_nhwc, token_ids, attention_mask):
        if isinstance(images_u8_nhwc, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images_u8_nhwc))
        else:
            images = images_u8_nhwc
        if images.ndim != 4:
            raise ValueError(f"Expected NHWC or NCHW uint8 stack, got {tuple(images.shape)}")
        if images.shape[-1] == 4:
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got {tuple(images.shape)}")
        if images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {tuple(images.shape)}")
        self.images = images  # N,4,H,W uint8
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        if len(self.images) != len(self.token_ids):
            raise ValueError("Image / token length mismatch")

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx].to(dtype=torch.float32).mul(1.0 / 127.5).sub(1.0)
        return x, self.token_ids[idx], self.attention_mask[idx]


def _on_google_drive(path) -> bool:
    s = str(Path(path)).replace("\\", "/")
    return "/drive/" in s or "MyDrive" in s or s.startswith("/content/drive")


def localize_processed_dir(processed_dir: str, local_root: str | None = None) -> str:
    """Copy the packed cache off Drive FUSE onto local disk before training.

    Random HuggingFace/Arrow reads against Google Drive are often 10–50x slower
    than local SSD and will stall a 24h Colab session.
    """
    src = Path(processed_dir)
    if not _on_google_drive(src):
        return str(src)
    root = Path(local_root or os.environ.get("BITROSS_LOCAL_DATA", "/content"))
    dest = root / "BitRoss-data" / src.name
    dest.mkdir(parents=True, exist_ok=True)
    from prepare_dataset import PACK_CAPTIONS, PACK_IMAGES

    names = [PACK_IMAGES, PACK_CAPTIONS, "prepare_stats.json"]
    src_stats = src / "prepare_stats.json"
    dest_stats = dest / "prepare_stats.json"
    dest_img = dest / PACK_IMAGES
    need = not dest_img.exists()
    if src_stats.exists() and dest_stats.exists():
        need = need or src_stats.stat().st_mtime > dest_stats.stat().st_mtime
    if need:
        print(f"Copying dataset off Drive: {src} -> {dest}")
        copied_pack = False
        for name in names:
            sp = src / name
            if sp.exists():
                shutil.copy2(sp, dest / name)
                copied_pack = copied_pack or name == PACK_IMAGES
        if not copied_pack:
            shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        print(f"Using local dataset copy: {dest}")
    return str(dest)


def _cpu_state_dict(state):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}


def atomic_torch_save(obj, path):
    """Write to a sibling temp file then replace. Survives Drive/Colab kills mid-save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def prune_epoch_checkpoints(save_dir, model_name, keep=2):
    if keep is None or keep <= 0:
        return
    save_dir = Path(save_dir)
    ckpts = sorted(
        save_dir.glob(f"{model_name}_epoch_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in ckpts[:-keep]:
        try:
            old.unlink()
            print(f"Pruned {old.name}")
        except OSError as e:
            print(f"Could not prune {old}: {e}")


def load_or_build_packed_dataset(processed_dir, tokenizer):
    """Prefer images_u8.npy; otherwise decode the HF cache once and write the pack."""
    from prepare_dataset import (
        PACK_IMAGES,
        load_packed_cache,
        write_packed_cache,
    )

    packed = load_packed_cache(processed_dir)
    if packed is None:
        from datasets import load_from_disk

        print(f"No {PACK_IMAGES}; decoding {processed_dir} once (then cached)")
        hf_ds = load_from_disk(processed_dir)
        if hasattr(hf_ds, "keys") and "train" in list(hf_ds.keys()):
            hf_ds = hf_ds["train"]
        write_packed_cache(hf_ds, processed_dir)
        del hf_ds
        packed = load_packed_cache(processed_dir)
        if packed is None:
            raise SystemExit(f"Failed to build packed cache in {processed_dir}")
    images, captions = packed
    print(f"Packed sprites: {images.shape[0]:,}  {images.shape[1:]}  {images.nbytes / 1e6:.0f} MB")
    print(f"Tokenizing {len(captions):,} captions...")
    ids_parts, mask_parts = [], []
    chunk = 4096
    for i in range(0, len(captions), chunk):
        ids, mask = tokenize_prompts(tokenizer, captions[i : i + chunk], device="cpu")
        ids_parts.append(ids)
        mask_parts.append(mask)
    token_ids = torch.cat(ids_parts, dim=0)
    attention_mask = torch.cat(mask_parts, dim=0)
    return PackedSpriteDataset(images, token_ids, attention_mask)


def pack_image_caption_dataset(dataset, tokenizer):
    """Convert a (tensor, caption) Dataset into the packed RAM form."""
    n = len(dataset)
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    captions = []
    for i in range(n):
        x, cap = dataset[i]
        arr = ((x.detach().float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        if arr.shape[0] == 4:
            arr = arr.permute(1, 2, 0)
        images[i] = arr.cpu().numpy()
        captions.append(str(cap))
    ids_parts, mask_parts = [], []
    chunk = 4096
    for i in range(0, len(captions), chunk):
        ids, mask = tokenize_prompts(tokenizer, captions[i : i + chunk], device="cpu")
        ids_parts.append(ids)
        mask_parts.append(mask)
    return PackedSpriteDataset(images, torch.cat(ids_parts), torch.cat(mask_parts))


class Text2ImageDataset(Dataset):
    def __init__(self, image_dir, metadata_file):
        self.image_dir = image_dir
        raw = load_metadata(metadata_file)
        self.metadata = []
        skipped = 0
        for item in raw:
            path = os.path.join(image_dir, item.get("file_name", ""))
            if not item.get("file_name") or not os.path.isfile(path):
                skipped += 1
                continue
            self.metadata.append(item)
        if skipped:
            print(f"Skipped {skipped} metadata rows with missing files")
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.image_dir, item["file_name"])
        try:
            image = Image.open(image_path).convert("RGBA")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
        return self.transform(image), item_prompt(item)


def unwrap_module(model):
    return getattr(model, "_orig_mod", model)


def unwrap_state_dict(model):
    return unwrap_module(model).state_dict()


def _is_frozen_clip_key(key: str) -> bool:
    return "text_encoder.clip." in key


def drop_frozen_clip(state):
    return {k: v for k, v in state.items() if not _is_frozen_clip_key(k)}


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Frozen CLIP weights are reloaded from the Hub; tracking them doubles
        # checkpoint size and copies ~63M params every step for no gain.
        self.shadow = {
            k: v.detach().clone()
            for k, v in unwrap_state_dict(model).items()
            if not _is_frozen_clip_key(k)
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in unwrap_state_dict(model).items():
            dest = self.shadow.get(k)
            if dest is None:
                continue
            if not v.is_floating_point():
                dest.copy_(v)
                continue
            dest.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def apply_to(self, model):
        raw = unwrap_module(model)
        current = raw.state_dict()
        self._backup = {k: current[k].detach().clone() for k in self.shadow if k in current}
        for k, v in self.shadow.items():
            if k in current:
                current[k].copy_(v)

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        raw = unwrap_module(model)
        current = raw.state_dict()
        for k, v in self._backup.items():
            if k in current:
                current[k].copy_(v)
        self._backup = None


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    scaler,
    log_every=50,
    use_amp=True,
    amp_dtype=torch.float16,
    cfg_dropout=0.1,
    max_grad_norm=1.0,
    ema=None,
    log_wandb=False,
):
    model.train()
    train_loss = 0.0
    n_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler_enabled = scaler.is_enabled() if scaler is not None else False

    for batch_idx, batch in enumerate(train_loader):
        data, input_ids, attention_mask = batch
        data = data.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, t_mean = flow_matching_loss(
                model, data, input_ids, attention_mask, cfg_dropout=cfg_dropout
            )

        if scaler_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = data.size(0)
        train_loss += loss.item() * bs
        n_samples += bs

        if log_wandb and batch_idx % log_every == 0:
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "batch_t_mean": t_mean.item() if torch.is_tensor(t_mean) else float(t_mean),
                }
            )

    return train_loss / max(n_samples, 1)


def save_checkpoint(
    path,
    model,
    ema,
    epoch,
    config,
    optimizer=None,
    scheduler=None,
    scaler=None,
    include_optim=True,
):
    payload = {
        "model": drop_frozen_clip(_cpu_state_dict(unwrap_state_dict(model))),
        "ema": drop_frozen_clip(
            _cpu_state_dict(ema.state_dict() if ema is not None else unwrap_state_dict(model))
        ),
        "epoch": int(epoch),
        "config": config,
        "arch": "pixel-dit-flow",
    }
    if include_optim and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if include_optim and scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if include_optim and scaler is not None:
        payload["scaler"] = scaler.state_dict()
    atomic_torch_save(payload, path)


def tensor_to_pil(t):
    return transforms.ToPILImage()((t * 0.5 + 0.5).cpu().clamp(0, 1))


def find_latest_checkpoint(save_dir, model_name="BitRoss"):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    latest = save_dir / f"{model_name}_latest.pth"
    if latest.exists():
        return latest
    ckpts = list(save_dir.glob(f"{model_name}_epoch_*.pth"))
    if not ckpts:
        ckpts = list(save_dir.glob("*_epoch_*.pth")) + list(save_dir.glob("*_final.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def auto_batch_size(requested: int, device: torch.device) -> int:
    if requested and requested > 0:
        return requested
    if device.type != "cuda":
        return 32
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram >= 20 * 1024**3:
        return 512
    if vram >= 12 * 1024**3:
        return 256
    return 128


def make_scaler(enabled: bool):
    if not enabled:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train BitRoss (CLIP pixel DiT + rectified flow)")
    p.add_argument("--data_dir", type=str, default="./training-items/")
    p.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="HuggingFace save_to_disk cache from prepare_dataset.py",
    )
    p.add_argument("--metadata", type=str, default=None, help="Defaults to <data_dir>/metadata.json")
    p.add_argument("--save_dir", type=str, default="./models/BitRoss/")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 = auto from GPU VRAM (512 on L4, 256 on T4, 128 otherwise)",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0, help="0 is fastest for the packed RAM cache")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'auto' for latest in save_dir")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--sample_interval", type=int, default=10)
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--project", type=str, default="BitRoss")
    p.add_argument("--model_name", type=str, default="BitRoss")
    p.add_argument("--keep_checkpoints", type=int, default=2, help="Epoch .pth files to keep on disk; 0 keeps all")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the DiT (optional extra speed)")
    p.add_argument("--selftest", action="store_true", help="Save/resume/packed-cache smoke test, then exit")
    args = p.parse_args()
    if args.resume in ("", "none", "None"):
        args.resume = None
    return args
    p.add_argument("--data_dir", type=str, default="./training-items/")
    p.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="HuggingFace save_to_disk cache from prepare_dataset.py",
    )
    p.add_argument("--metadata", type=str, default=None, help="Defaults to <data_dir>/metadata.json")
    p.add_argument("--save_dir", type=str, default="./models/BitRoss/")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 = auto from GPU VRAM (512 on L4, 256 on T4, 128 otherwise)",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0, help="0 is fastest for the packed RAM cache")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'auto' for latest in save_dir")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--sample_interval", type=int, default=10)
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--project", type=str, default="BitRoss")
    p.add_argument("--model_name", type=str, default="BitRoss")
    p.add_argument("--keep_checkpoints", type=int, default=2, help="Epoch .pth files to keep on disk; 0 keeps all")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the DiT (optional extra speed)")
    p.add_argument("--selftest", action="store_true", help="Save/resume/packed-cache smoke test, then exit")
    args = p.parse_args()
    if args.resume in ("", "none", "None"):
        args.resume = None
    return args


def main():
    args = parse_args()
    if args.selftest:
        return run_selftest()

    LEARNING_RATE = args.lr
    CFG_DROPOUT = args.cfg_dropout
    CFG_SCALE = args.cfg_scale
    SAMPLE_STEPS = args.sample_steps
    EMA_DECAY = 0.999
    MAX_GRAD_NORM = 1.0
    USE_AMP = not args.no_amp
    LOG_EVERY = 50
    SAVE_INTERVAL = args.save_interval
    SAVE_INTERVAL_IMAGE = args.sample_interval
    PROJECT_NAME = args.project
    MODEL_NAME = args.model_name
    SAVE_DIR = args.save_dir.rstrip("/") + "/"
    DATA_DIR = args.data_dir
    METADATA_FILE = args.metadata or os.path.join(DATA_DIR, "metadata.json")
    NUM_EPOCHS = args.epochs

    os.makedirs(SAVE_DIR, exist_ok=True)
    num_workers = 0 if args.num_workers is None else args.num_workers
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU detected — training will be extremely slow.")
    BATCH_SIZE = auto_batch_size(args.batch_size, device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print(f"GPU: {torch.cuda.get_device_name(0)}  bf16={use_bf16}  batch={BATCH_SIZE}")

    if args.processed_dir:
        processed_dir = localize_processed_dir(args.processed_dir)
        dataset = load_or_build_packed_dataset(processed_dir, tokenizer)
    else:
        folder_ds = Text2ImageDataset(DATA_DIR, METADATA_FILE)
        if len(folder_ds) == 0:
            raise SystemExit("No training images found (check --processed_dir or --data_dir)")
        print(f"Packing folder dataset ({len(folder_ds):,} images) into RAM...")
        dataset = pack_image_caption_dataset(folder_ds, tokenizer)
        del folder_ds
    if len(dataset) == 0:
        raise SystemExit("No training images found (check --processed_dir or --data_dir)")
    print(f"Dataset size: {len(dataset)}")

    run_config = {
        "arch": "pixel-dit-flow",
        "TEXT_DIM": TEXT_DIM,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "SAVE_INTERVAL": SAVE_INTERVAL,
        "MODEL_NAME": MODEL_NAME,
        "CFG_DROPOUT": CFG_DROPOUT,
        "CFG_SCALE": CFG_SCALE,
        "SAMPLE_STEPS": SAMPLE_STEPS,
        "EMA_DECAY": EMA_DECAY,
        "CLIP_ID": CLIP_ID,
        "USE_AMP": USE_AMP,
        "num_workers": num_workers,
    }

    wandb.init(
        project=PROJECT_NAME,
        config=run_config,
        mode="disabled" if args.no_wandb else "online",
    )

    train_loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=len(dataset) > BATCH_SIZE,
    )

    model = BitRoss(clip_id=CLIP_ID).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = make_scaler(USE_AMP and device.type == "cuda" and not use_bf16)
    ema = EMA(model, decay=EMA_DECAY)

    if args.compile and device.type == "cuda":
        try:
            model.dit = torch.compile(model.dit)
            print("torch.compile enabled on PixelDiT")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    start_epoch = 1
    resume_path = args.resume
    if resume_path == "auto":
        found = find_latest_checkpoint(SAVE_DIR, MODEL_NAME)
        resume_path = str(found) if found else None
        if resume_path:
            print(f"Auto-resume: {resume_path}")
    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = unwrap_module(model).load_state_dict(state, strict=False)
        missing = [k for k in missing if not _is_frozen_clip_key(k)]
        unexpected = [k for k in unexpected if not _is_frozen_clip_key(k)]
        if missing:
            print(f"Resume missing keys: {missing[:6]}")
        if unexpected:
            print(f"Resume unexpected keys: {unexpected[:6]}")
        if ckpt.get("ema"):
            loaded_ema = ckpt["ema"]
            model_keys = {k for k in unwrap_state_dict(model) if not _is_frozen_clip_key(k)}
            ema_keys = {k for k in loaded_ema if not _is_frozen_clip_key(k)}
            if ema_keys == model_keys:
                ema.shadow = {k: v.to(device) for k, v in loaded_ema.items() if k in model_keys}
            else:
                print("EMA key mismatch — resetting EMA from current weights")
                ema = EMA(model, decay=EMA_DECAY)
        if ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"Optimizer state not restored ({e}); continuing with fresh AdamW")
        if ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"Scheduler state not restored ({e})")
        if ckpt.get("scaler") and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"GradScaler state not restored ({e})")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    if start_epoch > NUM_EPOCHS:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1} >= --epochs {NUM_EPOCHS}. "
            "Increase --epochs or train from scratch (RESUME empty)."
        )
        wandb.finish()
        return

    if not args.no_wandb:
        wandb.watch(unwrap_module(model.dit), log="gradients", log_freq=100)

    n_vis = min(4, train_loader.batch_size or 4)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            log_every=LOG_EVERY,
            use_amp=USE_AMP,
            amp_dtype=amp_dtype,
            cfg_dropout=CFG_DROPOUT,
            max_grad_norm=MAX_GRAD_NORM,
            ema=ema,
            log_wandb=not args.no_wandb,
        )
        print(
            f"Epoch {epoch}, Loss: {train_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        if SAVE_INTERVAL_IMAGE and epoch % SAVE_INTERVAL_IMAGE == 0:
            from generate import generate_image

            output_image = f"{SAVE_DIR}output_epoch_{epoch}.png"
            prompt = "pixel art minecraft item, diamond sword, blue crystal blade"
            model.eval()
            ema.apply_to(model)
            try:
                generated_image = generate_image(
                    model,
                    prompt,
                    device,
                    tokenizer=tokenizer,
                    cfg_scale=CFG_SCALE,
                    steps=SAMPLE_STEPS,
                )
                generated_image.save(output_image)
                if not args.no_wandb:
                    prompts = [
                        prompt,
                        "pixel art minecraft item, red apple food",
                        "pixel art minecraft item, iron pickaxe tool",
                        "pixel art minecraft item, potion bottle, glowing purple liquid",
                    ][:n_vis]
                    input_ids, attention_mask = tokenize_prompts(tokenizer, prompts, device)
                    pooled, seq = model.encode_text(input_ids, attention_mask, drop_p=0.0)
                    samples = sample_rectified_flow(
                        model,
                        pooled,
                        seq,
                        steps=SAMPLE_STEPS,
                        cfg_scale=CFG_SCALE,
                        device=device,
                    )
                    log = {
                        "generated_image": wandb.Image(
                            output_image,
                            caption=f"EMA cfg={CFG_SCALE} steps={SAMPLE_STEPS} epoch {epoch}",
                        )
                    }
                    log.update(
                        {
                            f"sample_{i}": wandb.Image(
                                tensor_to_pil(samples[i]), caption=prompts[i]
                            )
                            for i in range(samples.size(0))
                        }
                    )
                    wandb.log(log)
            finally:
                ema.restore(model)

        if SAVE_INTERVAL and epoch % SAVE_INTERVAL == 0:
            latest_path = f"{SAVE_DIR}{MODEL_NAME}_latest.pth"
            epoch_path = f"{SAVE_DIR}{MODEL_NAME}_epoch_{epoch}.pth"
            save_checkpoint(
                latest_path,
                model,
                ema,
                epoch,
                run_config,
                optimizer,
                scheduler,
                scaler,
                include_optim=True,
            )
            save_checkpoint(
                epoch_path,
                model,
                ema,
                epoch,
                run_config,
                include_optim=False,
            )
            prune_epoch_checkpoints(SAVE_DIR, MODEL_NAME, keep=args.keep_checkpoints)
            print(f"Saved {latest_path} and {epoch_path}")

        scheduler.step()

    final_path = f"{SAVE_DIR}{MODEL_NAME}_final.pth"
    save_checkpoint(
        final_path, model, ema, NUM_EPOCHS, run_config, optimizer, scheduler, scaler, include_optim=True
    )
    latest_path = f"{SAVE_DIR}{MODEL_NAME}_latest.pth"
    save_checkpoint(
        latest_path, model, ema, NUM_EPOCHS, run_config, optimizer, scheduler, scaler, include_optim=True
    )
    print(f"Final checkpoint saved to {final_path}")
    wandb.finish()


def run_selftest():
    """Catch save/resume and packed-cache bugs before a long Colab run."""
    import tempfile as _tmp

    from prepare_dataset import write_packed_cache, load_packed_cache, caption_from_row

    print("selftest: packed cache + atomic checkpoint")
    tmp = Path(_tmp.mkdtemp(prefix="bitross-selftest-"))
    try:
        n = 8
        images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
        images[:, 4:12, 4:12] = (40, 180, 220, 255)
        captions = [caption_from_row(f"diamond_sword_{i}.png", "testmod") for i in range(n)]
        from datasets import Dataset as HFDataset
        from PIL import Image as PILImage

        pil_rows = []
        for i in range(n):
            im = PILImage.fromarray(images[i], mode="RGBA")
            pil_rows.append(
                {
                    "image": im,
                    "file_name": f"diamond_sword_{i}.png",
                    "mod_slug": "testmod",
                    "caption": captions[i],
                    "type": "item",
                    "author": "",
                }
            )
        hf = HFDataset.from_list(pil_rows)
        write_packed_cache(hf, tmp)
        packed = load_packed_cache(tmp)
        assert packed is not None
        imgs, caps = packed
        assert imgs.shape == (n, IMAGE_SIZE, IMAGE_SIZE, 4), imgs.shape
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)
        ds = PackedSpriteDataset(
            imgs,
            tokenize_prompts(tokenizer, caps, device="cpu")[0],
            tokenize_prompts(tokenizer, caps, device="cpu")[1],
        )
        x, ids, mask = ds[0]
        assert x.shape == (4, IMAGE_SIZE, IMAGE_SIZE), x.shape
        assert x.min() >= -1.01 and x.max() <= 1.01

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert len(batch) == 3 and batch[0].shape[0] == 4

        device = torch.device("cpu")
        model = BitRoss(clip_id=CLIP_ID).to(device)
        ema = EMA(model, decay=0.9)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)
        ckpt_path = tmp / "BitRoss_latest.pth"
        save_checkpoint(ckpt_path, model, ema, 1, {"arch": "pixel-dit-flow"}, opt, sched, include_optim=True)
        assert ckpt_path.exists() and ckpt_path.stat().st_size > 1000
        # Corrupt-safe: sibling tmp must not remain
        leftovers = list(tmp.glob("*.tmp"))
        assert not leftovers, leftovers

        data, input_ids, attention_mask = batch
        loss, _ = flow_matching_loss(model, data, input_ids, attention_mask, cfg_dropout=0.0)
        loss.backward()
        opt.step()
        ema.update(model)
        save_checkpoint(ckpt_path, model, ema, 2, {"arch": "pixel-dit-flow"}, opt, sched, include_optim=True)

        model2 = BitRoss(clip_id=CLIP_ID).to(device)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(raw["model"], strict=False)
        assert raw["epoch"] == 2
        assert "optimizer" in raw and "ema" in raw
        assert not any("text_encoder.clip." in k for k in raw["model"]), "frozen CLIP leaked into checkpoint"
        found = find_latest_checkpoint(tmp, "BitRoss")
        assert found == ckpt_path, found
        print("selftest ok", {"cache": str(tmp / "images_u8.npy"), "ckpt_mb": round(ckpt_path.stat().st_size / 1e6, 1)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
