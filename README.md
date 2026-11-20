# BitRoss 2

Text → 16×16 Minecraft item and block textures.

BitRoss 2 is a full rewrite of [BitRoss](https://github.com/OVAWARE/BitRoss). The old model was a
conditional VAE that regressed continuous RGBA values, which gave blurry, off-palette sprites. BitRoss 2
treats a texture the way a language model treats a sentence. Every pixel is one **token** from a learned
1024-color palette, and a bidirectional transformer starts from a fully masked canvas. Over about 16
parallel steps it fills in the pixels it is most confident about first. This is masked (absorbing-state)
discrete diffusion, the approach popularised for pixel art by
[Tessera](https://huggingface.co/spaces/thejagstudio/tessera-pixel-studio). BitRoss 2 is an independent
implementation, specialised for Minecraft.

Because the output is chosen from real palette colors and never interpolated, every result is crisp,
hard-edged pixel art.

## What's new compared with BitRoss 1

| | BitRoss 1 | BitRoss 2 |
|---|---|---|
| Model | CVAE, conv decoder, Tanh RGBA | 50M-param masked-diffusion transformer over palette tokens |
| Text | BERT CLS vector | frozen CLIP token states in joint attention + pooled adaLN |
| Data | hand-labeled subset | 1.03M textures → 649K curated, VLM-captioned |
| Conditioning | prompt | prompt + item/block + seamless flag + quality bucket |
| Editing | img2img blend | variations, inpainting, silhouette-locked recolors |
| Colors | free RGBA | 1024-token OKLab palette (+64 translucent), palette lock, color steering, max-color reduction |
| Blocks | — | seamless/tileable generation (toroidal convs + tile conditioning) with seam scoring |
| Output | PNG | PNG, upscales, sprite sheets, **material sets**, mod asset folders, resource packs |
| UI | HF Space | BitRoss Studio (Gradio) with live unmasking preview |

## How it works

```
 prompt ──► CLIP text encoder (frozen) ──► 32 token states ─┐
                                    pooled ──┐               │
 item/block, tileable, quality ──────────────┼─► adaLN       │
 mask ratio ─────────────────────────────────┘   modulation  ▼
                                                 [registers | text | 256 pixel tokens]
                                                 2 blocks @16×16 → 8 blocks @8×8 → 2 blocks @16×16
                                                 ──► logits over 1024 palette colors per pixel
```

* **Palette tokenizer** (`bitross/palette.py`): frequency-weighted k-means in OKLab over the 4M distinct
  colors in the curated set. Centroids snap to real colors, and the most common exact colors are pinned.
  Token 0 is transparent and the last 64 tokens are translucent (glass, ice, slime). The mean
  quantization error is ΔE 0.013, which is below what you can see.
* **Model** (`bitross/model.py`): hourglass transformer with QK-RMSNorm, 2D RoPE, SwiGLU with a depthwise 3×3
  conv, adaLN-single conditioning, and self-conditioning. Token embeddings are grounded in each color's
  OKLab value. The conv wraps around (toroidal padding) when generating tileable blocks.
* **Training objective** (`bitross/diffusion.py`): cosine mask schedule. Some masks are blob-shaped, which
  teaches inpainting, and some are "shape" masks that reveal the transparent silhouette, which teaches
  recoloring. Loss is cross-entropy with perceptual label smoothing over OKLab neighbours. Classifier-free
  guidance dropout is applied to text and metadata. Items are never mirrored during augmentation, because
  Minecraft tools always point to the top right. Seamless blocks get random toroidal shifts.
* **Sampler** (`bitross/sampler.py`): confidence-ordered parallel unmasking with CFG, annealed Gumbel noise,
  and spatial spreading, so neighbouring uncertain pixels are not committed in the same step. Hard
  constraints are supported: palette lock, per-pixel allowed sets, and kept pixels. A soft color bias and
  optional polish passes are also available.

## Dataset curation

Source: [OVAWARE/16xModdedMinecraft](https://huggingface.co/datasets/OVAWARE/16xModdedMinecraft), with
1,034,057 textures from 13K mods. `bitross/data/build.py` measures every image (about 11K images/s on 22
cores) and filters it:

| Step | Removed |
|---|---|
| exact duplicates (canonical pixels) | 153,289 |
| recolor spam: >6 textures with the same silhouette + luminance structure | 135,164 |
| ≤2 colors (flat fills, glyphs) | 29,793 |
| tiny fragments (<20 px in a small box) | 19,930 |
| non-color maps by name (`_n`, `_s`, emissive, overlay, glowmask, …) | 19,322 |
| per-mod cap (4,000 per mod and type) | 12,116 |
| empty / featureless / scattered dots / translucent overlays / flat | 15,409 |
| **kept** | **649,030** (349K items, 300K blocks, 118K seamless) |

Each kept image gets a heuristic quality score (color count, shading, noise, coverage, outline,
fragmentation, name quality). The score becomes a `low/mid/high` bucket that the model is conditioned on,
so mediocre data still teaches shapes while you sample at `quality="high"`. Sampling weights soften
mods with thousands of textures and balance items against blocks (60/40). Run
`python -m bitross.data.build preview` to get contact sheets of every drop reason.

**Captions.** Filenames are parsed into names (`aluminum_pickaxe` → "aluminum pickaxe", `oak_log_top` →
"oak log (top)"), then combined with dominant color names and a category. `bitross/data/vlm_label.py`
then runs **Qwen2.5-VL-3B** over every image. It returns a clean name, a one-line caption, a validity
flag and a 1–5 quality score. Images with meaningless filenames are labeled first. Each image keeps
several caption variants, and training samples one at random.

## Quick start

```bash
pip install -e ".[app]"
python app.py --model path/to/bitross.pt          # BitRoss Studio
```

```python
from bitross import BitRoss
br = BitRoss.from_pretrained("runs/bitross-small/bitross.pt")

br.generate("ruby sword", n=8, seed=0)
br.generate("mossy cobblestone", asset_type="block", tileable=True)
br.generate("emerald sword", silhouette=sword)                 # same shape, new material
br.generate("ruby sword", init=sword, strength=0.4)            # variations
br.generate("sword with a golden hilt", init=sword, mask=m)    # inpaint white pixels of m
br.generate("magic staff", palette="#2b1d0e #7a4b23 #43e0ff")  # hard palette lock
br.generate("frost wand", prefer_colors=ice_texture)           # soft color steering
br.generate("apple", n=16, rerank=True, keep_top=4)            # CLIP re-ranking

from bitross.minecraft import material_set, mod_assets
pieces = material_set(br, "ruby")   # ingot/gem, nugget, tools, armor, ore, block in one palette
```

CLI:

```bash
bitross-generate "ruby sword" -n 8 --scale 16 --sheet
bitross-generate "oak planks" --block --tileable
bitross-generate --material-set ruby --modid gemmod --out ruby_set/
```

`--material-set` writes the PNGs, a ready `assets/<modid>/` zip and a resource pack. The assets zip
contains textures, item and block models, blockstates, 1.21.4+ item definitions and `en_us.json`.

## Training on Colab

| Notebook | Runtime | What it does |
|---|---|---|
| `notebooks/BitRoss_1_Data.ipynb` | any | download → scan → filter → palette → previews (≈10 min) |
| `notebooks/BitRoss_2_Label.ipynb` | L4 | VLM captions, resumable shards on Drive (≈3–4 h) |
| `notebooks/BitRoss_3_Train.ipynb` | L4 / A100 | train, samples every 2K steps, Studio with share link, Hub upload |

Add your Hugging Face token as the Colab secret `HF_TOKEN` (the dataset is private). Training resumes
automatically from `RUN_DIR/last.pt`. Presets: `configs/small_l4.json` (50M params, 30K × 384, ≈4 h
on L4), `small_a100.json`, `base_a100.json` (155M), `nano_debug.json`.

Local equivalent:

```bash
python -m bitross.data.build all --parquet "data/raw/data/*.parquet" --work data/work --out data/cache
python -m bitross.data.vlm_label --work data/work --out data/labels            # needs a GPU + vLLM
python -m bitross.data.build cache --work data/work --out data/cache --vlm-labels "data/labels/*.jsonl"
python -m bitross.train --config configs/small_l4.json --data data/cache --out runs/bitross-small
pytest tests
```

## Repository layout

```
bitross/
  palette.py      OKLab palette tokenizer (build, encode/decode, locks, smoothing matrix)
  model.py        the denoiser
  diffusion.py    masking + loss
  sampler.py      parallel decoding with constraints
  pipeline.py     BitRoss high-level API (generate / stream / edit / rerank)
  minecraft.py    material sets, resource packs, mod asset export, sprite sheets
  train.py        GPU-resident training loop (Muon + AdamW, WSD schedule, EMA, bf16, compile)
  text.py         frozen CLIP text encoder
  data/           features, filename parsing, dataset build, VLM labeling
app.py            BitRoss Studio (Gradio)
configs/          training presets
notebooks/        Colab workflow
tools/            the original Modrinth scraper used to build the dataset
```

## Credits and licensing

* Method inspired by **Tessera** by Jagrat Patel (TheJagStudio), which showed that masked discrete
  diffusion over palette tokens works well for 16×16 sprites. BitRoss 2 contains no Tessera code or
  weights. Tessera itself is CC BY-NC 4.0. The underlying ideas come from MaskGIT (Chang et al. 2022),
  MDLM (Sahoo et al. 2024), DiT/PixArt-α (adaLN), and Muon (Jordan et al. 2024).
* Text encoder: OpenAI CLIP ViT-B/32 (MIT). Captioner: Qwen2.5-VL (Qwen license).
* Code: MIT.
* **Data:** the source textures come from mods under many licenses, including about 40% "All Rights
  Reserved". If you plan commercial use of the model or its outputs, consider retraining with
  `--licenses MIT Apache-2.0 CC0-1.0 ...` in the filter stage.
