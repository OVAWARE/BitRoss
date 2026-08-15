import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 16
LATENT_DIM = 256
TEXT_DIM = 512
HIDDEN_DIM = TEXT_DIM  # alias used by train/generate
BASE_CH = 64


def group_norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class FiLM(nn.Module):
    """Feature-wise affine modulation from a text vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film1 = FiLM(cond_dim, channels)
        self.film2 = FiLM(cond_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(x))
        h = self.film1(h, cond)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.film2(h, cond)
        h = self.conv2(h)
        return x + h


class SelfAttention2d(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(q, q, q, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv — avoids ConvTranspose checkerboard on sprites."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TextEncoder(nn.Module):
    """BERT with masked mean pooling and an MLP projection.

    Fine-tunes the last transformer blocks by default so item-name language
    can actually move the sprite prior, without paying for a full BERT update.
    """

    def __init__(
        self,
        hidden_size=TEXT_DIM,
        output_size=TEXT_DIM,
        freeze_bert=False,
        unfreeze_last_n=4,
    ):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        bert_dim = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(bert_dim),
            nn.Linear(bert_dim, output_size * 2),
            nn.GELU(),
            nn.Linear(output_size * 2, output_size),
            nn.LayerNorm(output_size),
        )

        for param in self.bert.parameters():
            param.requires_grad = False

        if not freeze_bert and unfreeze_last_n > 0:
            for layer in self.bert.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if getattr(self.bert, "pooler", None) is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True

        self._bert_trainable = any(p.requires_grad for p in self.bert.parameters())

    def forward(self, input_ids, attention_mask):
        if self._bert_trainable:
            hidden = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        else:
            with torch.no_grad():
                hidden = self.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.proj(pooled)


class PixelCVAE(nn.Module):
    """16x16 RGBA CVAE with FiLM text conditioning at every residual block."""

    def __init__(self, latent_dim=LATENT_DIM, text_dim=TEXT_DIM, base_ch=BASE_CH):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        ch = base_ch
        ch2, ch3 = ch * 2, ch * 4

        self.null_embed = nn.Parameter(torch.zeros(1, text_dim))

        self.conv_in = nn.Conv2d(4, ch, 3, padding=1)
        self.enc_16 = nn.ModuleList([ResBlock(ch, text_dim), ResBlock(ch, text_dim)])
        self.down_8 = Downsample(ch, ch2)
        self.enc_8 = nn.ModuleList([ResBlock(ch2, text_dim), ResBlock(ch2, text_dim)])
        self.down_4 = Downsample(ch2, ch3)
        self.enc_4 = nn.ModuleList(
            [ResBlock(ch3, text_dim), SelfAttention2d(ch3), ResBlock(ch3, text_dim)]
        )

        enc_flat = ch3 * 4 * 4
        self.to_stats = nn.Sequential(
            nn.SiLU(),
            nn.Linear(enc_flat + text_dim, ch3),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(ch3, latent_dim)
        self.fc_logvar = nn.Linear(ch3, latent_dim)

        self.from_latent = nn.Linear(latent_dim + text_dim, enc_flat)
        self.dec_4 = nn.ModuleList(
            [ResBlock(ch3, text_dim), SelfAttention2d(ch3), ResBlock(ch3, text_dim)]
        )
        self.up_8 = Upsample(ch3, ch2)
        self.dec_8 = nn.ModuleList([ResBlock(ch2, text_dim), ResBlock(ch2, text_dim)])
        self.up_16 = Upsample(ch2, ch)
        self.dec_16 = nn.ModuleList([ResBlock(ch, text_dim), ResBlock(ch, text_dim)])
        self.out_norm = group_norm(ch)
        self.out_conv = nn.Conv2d(ch, 4, 3, padding=1)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def drop_condition(self, cond, p):
        if (not self.training) or p <= 0:
            return cond
        drop = torch.rand(cond.size(0), 1, device=cond.device) < p
        return torch.where(drop, self.null_embed.expand_as(cond), cond)

    def encode(self, x, cond):
        h = self.conv_in(x)
        for block in self.enc_16:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        h = self.down_8(h)
        for block in self.enc_8:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        h = self.down_4(h)
        for block in self.enc_4:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        stats = self.to_stats(torch.cat([h.flatten(1), cond], dim=1))
        return self.fc_mu(stats), self.fc_logvar(stats)

    def decode(self, z, cond):
        h = self.from_latent(torch.cat([z, cond], dim=1))
        h = h.view(z.size(0), -1, 4, 4)
        for block in self.dec_4:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        h = self.up_8(h)
        for block in self.dec_8:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        h = self.up_16(h)
        for block in self.dec_16:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
        h = F.silu(self.out_norm(h))
        return torch.tanh(self.out_conv(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(max=10.0))
        return mu + torch.randn_like(std) * std

    def forward(self, x, cond, cfg_dropout=0.0):
        cond = self.drop_condition(cond, cfg_dropout)
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar


class CVAE(nn.Module):
    """Full BitRoss model: text encoder + pixel CVAE.

    `forward` still takes a precomputed text vector so the training loop can
    tokenize once and apply classifier-free guidance dropout on the VAE body.
    """

    def __init__(self, text_encoder, latent_dim=LATENT_DIM, text_dim=TEXT_DIM, base_ch=BASE_CH):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = PixelCVAE(latent_dim=latent_dim, text_dim=text_dim, base_ch=base_ch)
        self.latent_dim = latent_dim
        self.text_dim = text_dim

    def encode(self, x, cond):
        return self.vae.encode(x, cond)

    def decode(self, z, cond):
        return self.vae.decode(z, cond)

    def reparameterize(self, mu, logvar):
        return self.vae.reparameterize(mu, logvar)

    def forward(self, x, cond, cfg_dropout=0.0):
        return self.vae(x, cond, cfg_dropout=cfg_dropout)


def decode_with_cfg(model, z, cond, cfg_scale=2.0):
    """Linear CFG in image space: uncond + scale * (cond - uncond)."""
    vae = model.vae if isinstance(model, CVAE) else model
    if cfg_scale is None or cfg_scale == 1.0:
        return vae.decode(z, cond)
    uncond = vae.null_embed.expand_as(cond)
    img_c = vae.decode(z, cond)
    img_u = vae.decode(z, uncond)
    return img_u + cfg_scale * (img_c - img_u)


if __name__ == "__main__":
    vae = PixelCVAE()
    x = torch.randn(2, 4, IMAGE_SIZE, IMAGE_SIZE)
    c = torch.randn(2, TEXT_DIM)
    recon, mu, logvar = vae(x, c, cfg_dropout=0.5)
    assert recon.shape == x.shape, recon.shape
    assert mu.shape == (2, LATENT_DIM) and logvar.shape == (2, LATENT_DIM)
    cfg = decode_with_cfg(vae, torch.randn(2, LATENT_DIM), c, cfg_scale=2.0)
    assert cfg.shape == x.shape
    print(
        f"PixelCVAE ok  params={sum(p.numel() for p in vae.parameters()):,}  "
        f"recon={tuple(recon.shape)}"
    )
