import argparse
import json
import os
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
import wandb

from model import (
    CLIP_ID,
    IMAGE_SIZE,
    TEXT_DIM,
    BitRoss,
    flow_matching_loss,
    sample_rectified_flow,
    tokenize_prompts,
)


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
    """CLIP-style caption: name + description + tags.

    CLIP was trained on natural image-text pairs; wrapping the item as a
    'pixel art minecraft item' phrase is a cheap alignment win over raw BERT
    token dumps.
    """
    name = Path(str(item.get("file_name", ""))).stem.replace("_", " ").replace("-", " ")
    desc = str(item.get("description") or "").strip()
    tags = str(item.get("tags") or "").strip()
    parts = ["pixel art minecraft item"]
    if name:
        parts.append(name)
    if desc and desc.lower() != name.lower():
        parts.append(desc)
    if tags:
        parts.append(tags)
    return ", ".join(parts)


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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in unwrap_state_dict(model).items()
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in unwrap_state_dict(model).items():
            if not v.is_floating_point():
                self.shadow[k].copy_(v)
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def apply_to(self, model):
        raw = unwrap_module(model)
        self._backup = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        raw.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        unwrap_module(model).load_state_dict(self._backup, strict=True)
        self._backup = None


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    tokenizer,
    scaler,
    log_every=50,
    use_amp=True,
    amp_dtype=torch.float16,
    cfg_dropout=0.1,
    max_grad_norm=1.0,
    ema=None,
):
    model.train()
    train_loss = 0.0
    n_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler_enabled = scaler.is_enabled() if scaler is not None else False

    for batch_idx, (data, prompt) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        input_ids, attention_mask = tokenize_prompts(tokenizer, prompt, device)

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

        if batch_idx % log_every == 0:
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "batch_t_mean": t_mean.item() if torch.is_tensor(t_mean) else float(t_mean),
                }
            )

    return train_loss / max(n_samples, 1)


def save_checkpoint(path, model, ema, epoch, config, optimizer=None, scheduler=None):
    payload = {
        "model": unwrap_state_dict(model),
        "ema": ema.state_dict() if ema is not None else unwrap_state_dict(model),
        "epoch": epoch,
        "config": config,
        "arch": "pixel-dit-flow",
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def tensor_to_pil(t):
    return transforms.ToPILImage()((t * 0.5 + 0.5).cpu().clamp(0, 1))


def find_latest_checkpoint(save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    ckpts = list(save_dir.glob("*_epoch_*.pth")) + list(save_dir.glob("*_final.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def parse_args():
    p = argparse.ArgumentParser(description="Train BitRoss (CLIP pixel DiT + rectified flow)")
    p.add_argument("--data_dir", type=str, default="./training-items/")
    p.add_argument("--metadata", type=str, default=None, help="Defaults to <data_dir>/metadata.json")
    p.add_argument("--save_dir", type=str, default="./models/BitRoss/")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'auto' for latest in save_dir")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--sample_interval", type=int, default=10)
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--project", type=str, default="BitRoss")
    p.add_argument("--model_name", type=str, default="BitRoss")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()
    if args.resume in ("", "none", "None"):
        args.resume = None
    return args


def main():
    args = parse_args()
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
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

    os.makedirs(SAVE_DIR, exist_ok=True)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) // 2)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU detected — training will be extremely slow.")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"GPU: {torch.cuda.get_device_name(0)}  bf16={use_bf16}")

    dataset = Text2ImageDataset(DATA_DIR, METADATA_FILE)
    if len(dataset) == 0:
        raise SystemExit(f"No training images found in {DATA_DIR} ({METADATA_FILE})")
    print(f"Dataset size: {len(dataset)}")

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
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP and device.type == "cuda" and not use_bf16
    )
    ema = EMA(model, decay=EMA_DECAY)

    start_epoch = 1
    resume_path = args.resume
    if resume_path == "auto":
        found = find_latest_checkpoint(SAVE_DIR)
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
        if missing:
            print(f"Resume missing keys: {missing[:6]}")
        if unexpected:
            print(f"Resume unexpected keys: {unexpected[:6]}")
        if ckpt.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    if not args.no_wandb:
        wandb.watch(model.dit, log="gradients", log_freq=100)

    n_vis = min(4, train_loader.batch_size or 4)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            tokenizer,
            scaler,
            log_every=LOG_EVERY,
            use_amp=USE_AMP,
            amp_dtype=amp_dtype,
            cfg_dropout=CFG_DROPOUT,
            max_grad_norm=MAX_GRAD_NORM,
            ema=ema,
        )
        print(
            f"Epoch {epoch}, Loss: {train_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        if epoch % SAVE_INTERVAL_IMAGE == 0:
            from generate import generate_image

            output_image = f"{SAVE_DIR}output_epoch_{epoch}.png"
            prompt = "pixel art minecraft item, diamond sword, a blue sword made of diamond, sword, diamond, blue, weapon"
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
            finally:
                ema.restore(model)
            generated_image.save(output_image)
            wandb.log(
                {
                    "generated_image": wandb.Image(
                        output_image,
                        caption=f"EMA cfg={CFG_SCALE} steps={SAMPLE_STEPS} epoch {epoch}",
                    )
                }
            )

        if epoch % SAVE_INTERVAL == 0:
            model_save_path = f"{SAVE_DIR}{MODEL_NAME}_epoch_{epoch}.pth"
            save_checkpoint(
                model_save_path, model, ema, epoch, run_config, optimizer, scheduler
            )
            print(f"Model saved to {model_save_path}")

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                prompts = [
                    "pixel art minecraft item, diamond sword, blue crystal blade",
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
                wandb.log(
                    {
                        f"sample_{i}": wandb.Image(
                            tensor_to_pil(samples[i]), caption=prompts[i]
                        )
                        for i in range(samples.size(0))
                    }
                )

        scheduler.step()

    final_path = f"{SAVE_DIR}{MODEL_NAME}_final.pth"
    save_checkpoint(
        final_path, model, ema, NUM_EPOCHS, run_config, optimizer, scheduler
    )
    print(f"Final EMA checkpoint saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
