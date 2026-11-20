"""Publish a trained run to the Hugging Face Hub (inference checkpoint + configs + samples + card).

    python -m bitross.publish --run runs/bitross-small --repo OVAWARE/BitRoss-2 --private
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile

CARD = """---
license: mit
pipeline_tag: text-to-image
library_name: pytorch
tags: [pixel-art, minecraft, text-to-image, masked-diffusion, discrete-diffusion]
datasets: [OVAWARE/16xModdedMinecraft]
---

# BitRoss 2

Text to 16×16 Minecraft item and block textures. Every pixel is one token from a {vocab}-color
palette (token 0 = transparent, plus translucent colors); a {params:.0f}M-parameter bidirectional
transformer unmasks the whole canvas in parallel in ~16 steps, conditioned on a frozen CLIP text
encoder, the asset type (item / block), a tileable flag and a quality bucket.

Trained for {step:,} steps on a curated subset of
[16xModdedMinecraft](https://huggingface.co/datasets/OVAWARE/16xModdedMinecraft).

```python
from bitross import BitRoss
br = BitRoss.from_pretrained("{repo}")
br.generate("ruby sword", n=8, seed=0)
br.generate("mossy cobblestone", asset_type="block", tileable=True)
```

Code: https://github.com/OVAWARE/BitRoss

![samples](samples.png)
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from huggingface_hub import HfApi

    ck_path = os.path.join(a.run, "bitross.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    params = sum(v.numel() for k, v in ck["weights"].items() if "palette" not in k) / 1e6
    tmp = tempfile.mkdtemp()
    shutil.copy(ck_path, os.path.join(tmp, "bitross.pt"))
    for f in ("train_config.json", "metrics.jsonl"):
        if os.path.exists(os.path.join(a.run, f)):
            shutil.copy(os.path.join(a.run, f), tmp)
    with open(os.path.join(tmp, "model_config.json"), "w") as f:
        json.dump(ck["model_config"], f, indent=2)
    samples = sorted(glob.glob(os.path.join(a.run, "samples", "*.png")))
    if samples:
        shutil.copy(samples[-1], os.path.join(tmp, "samples.png"))
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write(CARD.format(vocab=ck["model_config"]["vocab_size"], params=params, step=ck.get("step", 0), repo=a.repo))
    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=tmp, repo_id=a.repo, commit_message=f"BitRoss 2 step {ck.get('step', 0)}")
    print(f"uploaded to https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
