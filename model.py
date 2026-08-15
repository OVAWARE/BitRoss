"""BitRoss generative model (2023–2026 stack).

The original CVAE is the wrong family for a quality retrain:
VAEs blur discrete pixels (Pixel VQ-VAE, EXAG 2022; PixDiff-PIG 2026).
16x16 RGBA is small enough to drop the autoencoder and generate in pixel space.

This module is a CLIP-conditioned pixel DiT trained with rectified flow:
- Flow Matching / Rectified Flow (Lipman et al. ICLR 2023; Liu et al. 2022)
- DiT AdaLN-Zero (Peebles & Xie ICCV 2023) + PixArt-α cross-attention (Chen 2023)
- CLIP text instead of BERT (vision-language space, not masked LM)
- CoOp-style learnable prompt tokens (Zhou et al. 2022)
- SD3 logit-normal timestep sampling (Esser et al. 2024)
- Offset noise on RGB (Lin et al. 2023) so global tint is not stuck at 0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 16
CHANNELS = 4
DIT_DIM = 384
DIT_DEPTH = 10
DIT_HEADS = 6
TEXT_DIM = DIT_DIM
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_MAX_LEN = 77
N_CTX = 4  # CoOp prefix tokens inserted after BOS
OFFSET_NOISE = 0.1

# Kept so older imports don't explode; unused by the flow model.
LATENT_DIM = 256
HIDDEN_DIM = TEXT_DIM


def tokenize_prompts(tokenizer, prompts, device=None):
    """Tokenize for CLIP (learned DiT prefix tokens are added after encoding)."""
    encoded = tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=CLIP_MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def logit_normal_t(batch, device, dtype=torch.float32):
    """SD3 timestep distribution: sigmoid(N(0,1)), biased toward mid-t."""
    u = torch.randn(batch, device=device, dtype=dtype)
    return torch.sigmoid(u).clamp(1e-4, 1.0 - 1e-4)


def sinusoidal_embedding(t, dim, max_period=10000.0):
    """t is in [0, 1]; scaled to 0–1000 so frequencies match diffusion practice."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype=t.dtype if t.dtype.is_floating_point else torch.float32)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_2d(h, w, dim, device="cpu"):
    """ViT-style 2D sine-cosine positional embedding (DiT)."""
    assert dim % 4 == 0
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    y = y.flatten()[:, None] * omega[None]
    x = x.flatten()[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text tower + a short learned prefix in DiT space.

    CLIP runs under no_grad so CoOp-in-CLIP cannot blow VRAM at large batch
    (backprop through 12 CLIP layers at batch 512 OOM'd an L4). The prefix is
    concatenated onto the projected sequence for DiT cross-attention instead.
    """

    def __init__(self, clip_id=CLIP_ID, n_ctx=N_CTX, out_dim=TEXT_DIM):
        super().__init__()
        from transformers import CLIPTextModel
        from transformers.utils import logging as hf_logging

        prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.clip = CLIPTextModel.from_pretrained(clip_id)
        finally:
            hf_logging.set_verbosity(prev)
        self.clip.eval()
        self.clip.requires_grad_(False)
        clip_dim = self.clip.config.hidden_size
        self.n_ctx = n_ctx
        self.context = nn.Parameter(torch.randn(n_ctx, out_dim) * 0.02)
        self.proj_seq = nn.Linear(clip_dim, out_dim)
        self.proj_pool = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def to_inference_dtype(self, dtype):
        """Keep frozen CLIP in fp16/bf16 so it does not occupy fp32 VRAM."""
        if dtype in (torch.float16, torch.bfloat16):
            self.clip.to(dtype=dtype)
        return self

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.clip(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            pooled = out.pooler_output
        # Cast off the frozen tower so trainable projections stay in model dtype.
        proj_dtype = self.proj_seq.weight.dtype
        hidden = hidden.to(dtype=proj_dtype)
        pooled = pooled.to(dtype=proj_dtype)
        pooled = self.proj_pool(pooled)
        seq = self.proj_seq(hidden)
        prefix = self.context.unsqueeze(0).expand(seq.size(0), -1, -1)
        seq = torch.cat([prefix, seq], dim=1)
        return pooled, seq


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    """AdaLN-Zero self-attn + PixArt-style text cross-attn + MLP."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, x, cond, text):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(cond).chunk(
            6, dim=1
        )
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + self.cross_attn(self.norm_cross(x), text, text, need_weights=False)[0]
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_ch=CHANNELS):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_ch)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN(cond).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class PixelDiT(nn.Module):
    """Patch-size-1 DiT over 16x16 RGBA — 256 tokens, one per pixel."""

    def __init__(
        self,
        dim=DIT_DIM,
        depth=DIT_DEPTH,
        heads=DIT_HEADS,
        image_size=IMAGE_SIZE,
        channels=CHANNELS,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.dim = dim
        self.patch = nn.Linear(channels, dim)
        pos = sincos_2d(image_size, image_size, dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final = FinalLayer(dim, channels)
        self.grad_checkpoint = True

    def forward(self, x, t, pooled, text_seq):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.patch(tokens) + self.pos_embed
        cond = self.t_embed(sinusoidal_embedding(t, self.dim).to(dtype=tokens.dtype))
        cond = cond + pooled
        use_ckpt = self.training and self.grad_checkpoint and tokens.requires_grad
        for block in self.blocks:
            if use_ckpt:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, cond, text_seq, use_reentrant=False
                )
            else:
                tokens = block(tokens, cond, text_seq)
        out = self.final(tokens, cond)
        return out.transpose(1, 2).reshape(b, c, h, w)


class BitRoss(nn.Module):
    """CLIP text encoder + pixel DiT. Predicts rectified-flow velocity."""

    def __init__(self, clip_id=CLIP_ID):
        super().__init__()
        self.text_encoder = CLIPTextEncoder(clip_id=clip_id, n_ctx=N_CTX, out_dim=TEXT_DIM)
        self.dit = PixelDiT()
        self.null_pooled = nn.Parameter(torch.zeros(1, TEXT_DIM))
        self.null_token = nn.Parameter(torch.zeros(1, 1, TEXT_DIM))

    def encode_text(self, input_ids, attention_mask, drop_p=0.0):
        pooled, seq = self.text_encoder(input_ids, attention_mask)
        if self.training and drop_p > 0:
            drop = torch.rand(pooled.size(0), device=pooled.device) < drop_p
            pooled = torch.where(drop[:, None], self.null_pooled.expand_as(pooled), pooled)
            seq = torch.where(drop[:, None, None], self.null_token.expand_as(seq), seq)
        return pooled, seq

    def velocity(self, x, t, pooled, text_seq):
        return self.dit(x, t, pooled, text_seq)

    def cfg_velocity(self, x, t, pooled, text_seq, cfg_scale=2.5):
        v_c = self.velocity(x, t, pooled, text_seq)
        if cfg_scale is None or cfg_scale == 1.0:
            return v_c
        b = x.size(0)
        v_u = self.velocity(
            x,
            t,
            self.null_pooled.expand(b, -1),
            self.null_token.expand(b, text_seq.size(1), -1),
        )
        return v_u + cfg_scale * (v_c - v_u)

    def forward(self, x, t, input_ids, attention_mask, drop_p=0.0):
        pooled, seq = self.encode_text(input_ids, attention_mask, drop_p=drop_p)
        return self.velocity(x, t, pooled, seq)


@torch.no_grad()
def sample_rectified_flow(
    model,
    pooled,
    text_seq,
    steps=20,
    cfg_scale=2.5,
    temperature=1.0,
    sampler="heun",
    x_start=None,
    t_start=1.0,
    device=None,
):
    """Integrate dx/dt = v from t=t_start (noise) to t=0 (data)."""
    device = device or pooled.device
    b = pooled.size(0)
    if x_start is None:
        x = torch.randn(b, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device) * temperature
    else:
        x = x_start

    ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(b)
        dt = ts[i + 1] - ts[i]
        v = model.cfg_velocity(x, t, pooled, text_seq, cfg_scale=cfg_scale)
        if sampler == "heun":
            x_pred = x + v * dt
            t_next = ts[i + 1].expand(b)
            v_next = model.cfg_velocity(
                x_pred, t_next, pooled, text_seq, cfg_scale=cfg_scale
            )
            x = x + 0.5 * dt * (v + v_next)
        else:
            x = x + v * dt
    return x.clamp(-1, 1)


def flow_matching_loss(model, x, input_ids, attention_mask, cfg_dropout=0.1, alpha_weight=1.5):
    """OT-path rectified flow: x_t = (1-t) x + t ε, target v = ε - x."""
    b = x.size(0)
    t = logit_normal_t(b, x.device, dtype=x.dtype)
    noise = torch.randn_like(x)
    noise[:, :3] = noise[:, :3] + OFFSET_NOISE * torch.randn(
        b, 3, 1, 1, device=x.device, dtype=x.dtype
    )
    t4 = t[:, None, None, None]
    x_t = (1.0 - t4) * x + t4 * noise
    v_target = noise - x
    v_pred = model(x_t, t, input_ids, attention_mask, drop_p=cfg_dropout)
    weights = torch.ones_like(x)
    weights[:, 3:4] = alpha_weight
    mse = (weights * (v_pred - v_target).pow(2)).mean()
    return mse, t.detach().mean()


# Backward-compatible names used by the previous CVAE training loop.
TextEncoder = CLIPTextEncoder
CVAE = BitRoss


if __name__ == "__main__":
    dit = PixelDiT()
    x = torch.randn(2, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.rand(2)
    pooled = torch.randn(2, TEXT_DIM)
    seq = torch.randn(2, 16, TEXT_DIM)
    v = dit(x, t, pooled, seq)
    assert v.shape == x.shape, v.shape
    print(f"PixelDiT ok  params={sum(p.numel() for p in dit.parameters()):,}  v={tuple(v.shape)}")
