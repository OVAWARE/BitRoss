import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer
import wandb

from model import (
    CVAE,
    IMAGE_SIZE,
    LATENT_DIM,
    TEXT_DIM,
    TextEncoder,
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
    name = Path(str(item.get("file_name", ""))).stem.replace("_", " ").replace("-", " ")
    desc = str(item.get("description") or "").strip()
    tags = str(item.get("tags") or "").strip()
    parts = []
    if name:
        parts.append(name)
    if desc and desc.lower() != name.lower():
        parts.append(desc)
    if tags:
        parts.append(f"tags: {tags}")
    return ". ".join(parts) if parts else name


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


def sobel_xy(img):
    channels = img.size(1)
    kx = img.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = img.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)
    gx = F.conv2d(img, kx, padding=1, groups=channels)
    gy = F.conv2d(img, ky, padding=1, groups=channels)
    return gx, gy


def loss_function(recon_x, x, mu, logvar, beta=1.0, alpha_weight=2.5, free_nats=0.5):
    """Mean-reduced losses so beta/free-bits stay stable across batch size.

    L1 + MSE for color, extra alpha occupancy BCE (sprites are mostly empty),
    Sobel matching for crisp pixel edges, and clamped KL to avoid collapse.
    """
    weights = torch.ones_like(x)
    weights[:, 3:4] = alpha_weight
    l1 = (weights * (recon_x - x).abs()).mean()
    mse = F.mse_loss(recon_x, x)

    alpha_t = ((x[:, 3] + 1) * 0.5).clamp(0, 1)
    alpha_p = ((recon_x[:, 3] + 1) * 0.5).clamp(1e-4, 1 - 1e-4)
    bce = F.binary_cross_entropy(alpha_p, alpha_t)

    gx_r, gy_r = sobel_xy(recon_x)
    gx_x, gy_x = sobel_xy(x)
    edge = (gx_r - gx_x).abs().mean() + (gy_r - gy_x).abs().mean()

    kl_pp = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kl_pp.clamp(min=free_nats).mean()

    total = l1 + 0.5 * mse + 0.5 * bce + 0.25 * edge + beta * kld
    return total, {
        "l1": l1.detach(),
        "mse": mse.detach(),
        "alpha_bce": bce.detach(),
        "edge": edge.detach(),
        "kld": kld.detach(),
    }


def effective_beta(base_beta, epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return base_beta
    return base_beta * min(1.0, epoch / warmup_epochs)


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
    beta=1.0,
    log_every=50,
    use_amp=True,
    amp_dtype=torch.float16,
    cfg_dropout=0.15,
    max_grad_norm=1.0,
    ema=None,
    free_nats=0.5,
):
    model.train()
    train_loss = 0.0
    n_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler_enabled = scaler.is_enabled() if scaler is not None else False

    for batch_idx, (data, prompt) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        encoded_input = tokenizer(
            list(prompt),
            padding=True,
            truncation=True,
            max_length=80,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded_input["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            text_encoding = model.text_encoder(input_ids, attention_mask)
            recon_batch, mu, logvar = model(data, text_encoding, cfg_dropout=cfg_dropout)
            loss, parts = loss_function(
                recon_batch, data, mu, logvar, beta=beta, free_nats=free_nats
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
                    "batch_l1": parts["l1"].item(),
                    "batch_mse": parts["mse"].item(),
                    "batch_alpha_bce": parts["alpha_bce"].item(),
                    "batch_edge": parts["edge"].item(),
                    "batch_kl_divergence": parts["kld"].item(),
                    "beta": beta,
                }
            )

    return train_loss / max(n_samples, 1)


def save_checkpoint(path, model, ema, epoch, config):
    torch.save(
        {
            "model": unwrap_state_dict(model),
            "ema": ema.state_dict() if ema is not None else unwrap_state_dict(model),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def tensor_to_pil(t):
    return transforms.ToPILImage()((t * 0.5 + 0.5).cpu().clamp(0, 1))


def main():
    NUM_EPOCHS = 600
    BATCH_SIZE = 128
    LEARNING_RATE = 2e-4
    BERT_LR = 2e-5
    BETA = 0.4
    BETA_WARMUP_EPOCHS = 80
    FREE_NATS = 0.5
    CFG_DROPOUT = 0.15
    CFG_SCALE = 2.0
    EMA_DECAY = 0.999
    MAX_GRAD_NORM = 1.0
    UNFREEZE_LAST_N = 4
    FREEZE_BERT = False
    USE_AMP = True
    LOG_EVERY = 50

    SAVE_INTERVAL = 25
    SAVE_INTERVAL_IMAGE = 10
    PROJECT_NAME = "BitRoss"
    MODEL_NAME = "BitRoss"
    SAVE_DIR = "./models/BitRoss/"

    os.makedirs(SAVE_DIR, exist_ok=True)

    num_workers = max(1, (os.cpu_count() or 2) // 2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    DATA_DIR = "./training-items/"
    METADATA_FILE = "./training-items/metadata.json"

    run_config = {
        "LATENT_DIM": LATENT_DIM,
        "TEXT_DIM": TEXT_DIM,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "BERT_LR": BERT_LR,
        "SAVE_INTERVAL": SAVE_INTERVAL,
        "MODEL_NAME": MODEL_NAME,
        "BETA": BETA,
        "BETA_WARMUP_EPOCHS": BETA_WARMUP_EPOCHS,
        "FREE_NATS": FREE_NATS,
        "CFG_DROPOUT": CFG_DROPOUT,
        "CFG_SCALE": CFG_SCALE,
        "EMA_DECAY": EMA_DECAY,
        "UNFREEZE_LAST_N": UNFREEZE_LAST_N,
        "FREEZE_BERT": FREEZE_BERT,
        "USE_AMP": USE_AMP,
        "num_workers": num_workers,
    }

    wandb.init(project=PROJECT_NAME, config=run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Full precision matmul — TF32 is a quality hit on small spatial maps
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset = Text2ImageDataset(DATA_DIR, METADATA_FILE)
    if len(dataset) == 0:
        raise SystemExit(f"No training images found in {DATA_DIR} ({METADATA_FILE})")

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

    text_encoder = TextEncoder(
        hidden_size=TEXT_DIM,
        output_size=TEXT_DIM,
        freeze_bert=FREEZE_BERT,
        unfreeze_last_n=UNFREEZE_LAST_N,
    )
    model = CVAE(text_encoder, latent_dim=LATENT_DIM, text_dim=TEXT_DIM).to(device)

    bert_params = [p for p in model.text_encoder.bert.parameters() if p.requires_grad]
    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("text_encoder.bert.")
    ]
    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": LEARNING_RATE},
            {"params": bert_params, "lr": BERT_LR},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP and device.type == "cuda" and not use_bf16
    )
    ema = EMA(model, decay=EMA_DECAY)

    wandb.watch(model.vae, log="gradients", log_freq=100)

    fixed_images, fixed_prompts = next(iter(train_loader))
    n_vis = min(4, fixed_images.size(0))
    fixed_images = fixed_images[:n_vis].to(device)
    fixed_prompts = list(fixed_prompts[:n_vis])

    for epoch in range(1, NUM_EPOCHS + 1):
        beta = effective_beta(BETA, epoch, BETA_WARMUP_EPOCHS)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            tokenizer,
            scaler,
            beta=beta,
            log_every=LOG_EVERY,
            use_amp=USE_AMP,
            amp_dtype=amp_dtype,
            cfg_dropout=CFG_DROPOUT,
            max_grad_norm=MAX_GRAD_NORM,
            ema=ema,
            free_nats=FREE_NATS,
        )
        print(
            f"Epoch {epoch}, Loss: {train_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}, Beta: {beta:.3f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "bert_lr": scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else BERT_LR,
                "beta": beta,
            }
        )

        if epoch % SAVE_INTERVAL_IMAGE == 0:
            from generate import generate_image

            output_image = f"{SAVE_DIR}output_epoch_{epoch}.png"
            prompt = "diamond sword. A blue sword made of diamond. tags: sword, diamond, blue, weapon"
            model.eval()
            ema.apply_to(model)
            try:
                generated_image = generate_image(
                    model, prompt, device, cfg_scale=CFG_SCALE
                )
            finally:
                ema.restore(model)
            generated_image.save(output_image)
            wandb.log(
                {
                    "generated_image": wandb.Image(
                        output_image,
                        caption=f"EMA cfg={CFG_SCALE} epoch {epoch}: {prompt}",
                    )
                }
            )

        if epoch % SAVE_INTERVAL == 0:
            model_save_path = f"{SAVE_DIR}{MODEL_NAME}_epoch_{epoch}.pth"
            save_checkpoint(model_save_path, model, ema, epoch, run_config)
            print(f"Model saved to {model_save_path}")

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                encoded_input = tokenizer(
                    fixed_prompts,
                    padding=True,
                    truncation=True,
                    max_length=80,
                    return_tensors="pt",
                )
                input_ids = encoded_input["input_ids"].to(device)
                attention_mask = encoded_input["attention_mask"].to(device)
                text_encoding = model.text_encoder(input_ids, attention_mask)
                recon_batch, _, _ = model(fixed_images, text_encoding, cfg_dropout=0.0)

                wandb.log(
                    {
                        f"original_vs_reconstructed_{i}": [
                            wandb.Image(
                                tensor_to_pil(fixed_images[i]), caption=f"Original {i}"
                            ),
                            wandb.Image(
                                tensor_to_pil(recon_batch[i]),
                                caption=f"Reconstructed {i}",
                            ),
                        ]
                        for i in range(n_vis)
                    }
                )

        scheduler.step()

    final_path = f"{SAVE_DIR}{MODEL_NAME}_final.pth"
    save_checkpoint(final_path, model, ema, NUM_EPOCHS, run_config)
    print(f"Final EMA checkpoint saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
