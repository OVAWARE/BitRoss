import json
import os

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer
import wandb

from model import CVAE, TextEncoder, HIDDEN_DIM, IMAGE_SIZE, LATENT_DIM


def load_metadata(metadata_file):
    """Accept either a JSON array or JSONL (as written by label.py / labelOllama.py)."""
    with open(metadata_file, "r") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class Text2ImageDataset(Dataset):
    def __init__(self, image_dir, metadata_file):
        self.image_dir = image_dir
        self.metadata = load_metadata(metadata_file)
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
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            # Return a blank sample rather than None (None breaks default collate)
            image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
            return self.transform(image), ""
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
            return self.transform(image), ""

        image = self.transform(image)
        prompt = str(item.get("description", ""))
        return image, prompt


def loss_function(recon_x, x, mu, logvar, beta=1.0, alpha_weight=2.0):
    # Emphasize alpha — Minecraft item sprites are mostly transparency
    weights = torch.ones_like(x)
    weights[:, 3:4] = alpha_weight
    RECON = (weights * (recon_x - x).abs()).sum()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON + beta * KLD, RECON, KLD


def effective_beta(base_beta, epoch, warmup_epochs):
    """Linear KL warmup — improves recon early, then brings in the prior."""
    if warmup_epochs <= 0:
        return base_beta
    return base_beta * min(1.0, epoch / warmup_epochs)


def unwrap_state_dict(model):
    """torch.compile wraps modules; save the underlying weights for generate.py."""
    raw = getattr(model, "_orig_mod", model)
    return raw.state_dict()


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
):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, prompt) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        encoded_input = tokenizer(
            list(prompt), padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded_input["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            text_encoding = model.text_encoder(input_ids, attention_mask)
            recon_batch, mu, logvar = model(data, text_encoding)
            loss, recon_loss, kld = loss_function(
                recon_batch, data, mu, logvar, beta=beta
            )

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_every == 0:
            wandb.log(
                {
                    "batch_loss": loss.item() / max(data.size(0), 1),
                    "batch_reconstruction_loss": (recon_loss / data.size(0)).item(),
                    "batch_kl_divergence": (kld / data.size(0)).item(),
                    "beta": beta,
                }
            )

    return train_loss / max(len(train_loader.dataset), 1)


def main():
    NUM_EPOCHS = 500
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    # Beta > 1 encourages disentanglement, Beta < 1 focuses on reconstruction
    BETA = 0.8
    BETA_WARMUP_EPOCHS = 50
    FREEZE_BERT = True
    USE_AMP = True
    LOG_EVERY = 50

    SAVE_INTERVAL = 25
    SAVE_INTERVAL_IMAGE = 10  # was 1 — generation/wandb every epoch is expensive
    PROJECT_NAME = "BitRoss"
    MODEL_NAME = "BitRoss"
    SAVE_DIR = "./models/BitRoss/"

    os.makedirs(SAVE_DIR, exist_ok=True)

    num_workers = max(1, (os.cpu_count() or 2) // 2)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    DATA_DIR = "./training-items/"
    METADATA_FILE = "./training-items/metadata.json"

    wandb.init(
        project=PROJECT_NAME,
        config={
            "LATENT_DIM": LATENT_DIM,
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_EPOCHS": NUM_EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "SAVE_INTERVAL": SAVE_INTERVAL,
            "MODEL_NAME": MODEL_NAME,
            "BETA": BETA,
            "BETA_WARMUP_EPOCHS": BETA_WARMUP_EPOCHS,
            "FREEZE_BERT": FREEZE_BERT,
            "USE_AMP": USE_AMP,
            "num_workers": num_workers,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = Text2ImageDataset(DATA_DIR, METADATA_FILE)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    text_encoder = TextEncoder(
        hidden_size=HIDDEN_DIM, output_size=HIDDEN_DIM, freeze_bert=FREEZE_BERT
    )
    model = CVAE(text_encoder).to(device)
    if torch.__version__.startswith("2."):
        print("Compiling model with torch.compile()")
        model = torch.compile(model)

    # Only optimize trainable params (projection + CVAE when BERT is frozen)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    wandb.watch(getattr(model, "_orig_mod", model), log="gradients", log_freq=100)

    # Fixed eval batch so we don't rebuild a DataLoader iterator every 10 epochs
    fixed_images, fixed_prompts = next(iter(train_loader))
    fixed_images = fixed_images[:4].to(device)
    fixed_prompts = list(fixed_prompts[:4])

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
                "beta": beta,
            }
        )

        if epoch % SAVE_INTERVAL_IMAGE == 0:
            from generate import generate_image

            output_image = f"{SAVE_DIR}output_epoch_{epoch}.png"
            prompt = "A blue sword made of diamond"
            generated_image = generate_image(model, prompt, device)
            generated_image.save(output_image)
            wandb.log(
                {
                    "generated_image": wandb.Image(
                        output_image,
                        caption=f"Generated at epoch {epoch} with prompt {prompt}",
                    )
                }
            )

        if epoch % SAVE_INTERVAL == 0:
            model_save_path = f"{SAVE_DIR}{MODEL_NAME}_epoch_{epoch}.pth"
            torch.save(unwrap_state_dict(model), model_save_path)
            print(f"Model saved to {model_save_path}")

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                encoded_input = tokenizer(
                    fixed_prompts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                input_ids = encoded_input["input_ids"].to(device)
                attention_mask = encoded_input["attention_mask"].to(device)
                text_encoding = model.text_encoder(input_ids, attention_mask)
                recon_batch, _, _ = model(fixed_images, text_encoding)

                original_images = [
                    transforms.ToPILImage()((fixed_images[i] * 0.5 + 0.5).cpu().clamp(0, 1))
                    for i in range(4)
                ]
                reconstructed_images = [
                    transforms.ToPILImage()((recon_batch[i] * 0.5 + 0.5).cpu().clamp(0, 1))
                    for i in range(4)
                ]

                wandb.log(
                    {
                        f"original_vs_reconstructed_{i}": [
                            wandb.Image(original_images[i], caption=f"Original {i}"),
                            wandb.Image(
                                reconstructed_images[i], caption=f"Reconstructed {i}"
                            ),
                        ]
                        for i in range(4)
                    }
                )

        scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()
