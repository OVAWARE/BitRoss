"""Caption the curated textures with a vision-language model (default: Qwen2.5-VL-3B via vLLM).

For each kept image the model sees the texture upscaled 14x on a neutral background, plus the
file name / mod / asset type as a (possibly misleading) hint, and returns one line of JSON:

    {"name": "ruby pickaxe", "caption": "a pickaxe with a red ruby head and wooden handle",
     "valid": true, "quality": 4}

Work is split into fixed shards written as `<out>/labels_<shard>.jsonl`; finished shards are
skipped, so the job can be resumed after a Colab disconnect. Write --out to Google Drive.

    python -m bitross.data.vlm_label --work data/work --out /content/drive/MyDrive/BitRoss/labels
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

SYSTEM = ("You label small game textures (items and blocks from Minecraft mods) for a text-to-image dataset. "
          "Be concrete and visual. Never mention Minecraft, pixel art, textures, sprites, icons or resolution.")

QUESTION = (
    "Hint: the file is named '{file}' from the mod '{mod}' and is a {kind} texture. The hint can be wrong or "
    "meaningless; trust the picture.\n"
    "Reply with ONE line of JSON and nothing else, with keys:\n"
    '"name": what it is in 1-4 lowercase words (e.g. "ruby pickaxe", "mossy stone bricks", "cooked salmon"),\n'
    '"caption": one short phrase of 5-14 words describing the object, its colors and material,\n'
    '"valid": false only if this is not a usable standalone picture (a UI element, a fragment of a larger '
    'image, an overlay, random noise, a letter or number), otherwise true,\n'
    '"quality": 1-5 rating of how clean and readable the art is.'
)

_JSON = re.compile(r"\{.*\}", re.S)


def render(rgba: np.ndarray, size: int = 224, bg=(150, 150, 150)) -> "Image.Image":
    from PIL import Image
    im = Image.fromarray(np.asarray(rgba, np.uint8), "RGBA").resize((size, size), Image.NEAREST)
    base = Image.new("RGBA", (size, size), (*bg, 255))
    base.alpha_composite(im)
    return base.convert("RGB")


def parse_reply(text: str) -> Optional[Dict]:
    m = _JSON.search(text or "")
    if not m:
        return None
    s = m.group(0)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        try:
            d = json.loads(s.replace("'", '"').replace("True", "true").replace("False", "false"))
        except json.JSONDecodeError:
            return None
    out = {}
    name = str(d.get("name", "")).strip().lower()
    cap = str(d.get("caption", "")).strip()
    if name and len(name) <= 48:
        out["name"] = re.sub(r"[^a-z0-9 '\-]", "", name).strip()
    if cap and len(cap) <= 160:
        cap = re.sub(r"(?i)\b(minecraft|pixel[- ]art|pixelated|texture|sprite|16x16|icon)\b", "", cap)
        out["caption"] = re.sub(r"\s+", " ", cap).strip(" ,.")
    v = d.get("valid", True)
    out["valid"] = not (v is False or str(v).lower() == "false")
    try:
        out["quality"] = int(np.clip(int(float(d.get("quality", 3))), 1, 5))
    except (TypeError, ValueError):
        out["quality"] = 3
    return out


class VLLMBackend:
    def __init__(self, model: str, max_batch: int = 256, gpu_mem: float = 0.88):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.proc = AutoProcessor.from_pretrained(model)
        self.llm = LLM(model=model, max_model_len=1024, limit_mm_per_prompt={"image": 1},
                       gpu_memory_utilization=gpu_mem, max_num_seqs=max_batch, dtype="bfloat16")
        self.sp = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=96, stop=["\n"])

    def prompt(self, question: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, images, questions) -> List[str]:
        reqs = [{"prompt": self.prompt(q), "multi_modal_data": {"image": im}} for im, q in zip(images, questions)]
        outs = self.llm.generate(reqs, self.sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    """Slow fallback when vLLM is not available (plain transformers generate)."""

    def __init__(self, model: str, max_batch: int = 32):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def __call__(self, images, questions) -> List[str]:
        msgs = [[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}] for q in questions]
        prompts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = self.proc(text=prompts, images=list(images), return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=96, do_sample=False)
        out = out[:, inp["input_ids"].shape[1]:]
        return [t.split("\n")[0] for t in self.proc.batch_decode(out, skip_special_tokens=True)]


def run(work: str, out: str, model: str = DEFAULT_MODEL, shard_size: int = 20000, batch: int = 512,
        backend: str = "vllm", limit: Optional[int] = None, only_unnamed_first: bool = True,
        shards: Optional[List[int]] = None) -> None:
    os.makedirs(out, exist_ok=True)
    m = pd.read_parquet(os.path.join(work, "filtered.parquet"),
                        columns=["source_index", "file_name", "mod_slug", "type", "keep", "meaningful"])
    m = m[m["keep"]]
    if only_unnamed_first:   # label the images whose file names say nothing first
        m = m.sort_values("meaningful", kind="stable")
    m = m.reset_index(drop=True)
    if limit:
        m = m.iloc[:limit]
    px = np.load(os.path.join(work, "pixels.npy"), mmap_mode="r")
    n_shards = (len(m) + shard_size - 1) // shard_size
    todo = [s for s in range(n_shards) if (shards is None or s in shards)
            and not os.path.exists(os.path.join(out, f"labels_{s:04d}.jsonl"))]
    print(f"{len(m):,} images, {n_shards} shards, {len(todo)} to do")
    if not todo:
        return
    vlm = VLLMBackend(model, max_batch=min(batch, 512)) if backend == "vllm" else HFBackend(model)
    for s in todo:
        part = m.iloc[s * shard_size:(s + 1) * shard_size]
        t0 = time.time()
        recs, n_ok = [], 0
        for b in range(0, len(part), batch):
            rows = part.iloc[b:b + batch]
            imgs = [render(px[i]) for i in rows["source_index"]]
            qs = [QUESTION.format(file=r.file_name, mod=r.mod_slug or "unknown", kind=r.type or "item")
                  for r in rows.itertuples()]
            replies = vlm(imgs, qs)
            for r, rep in zip(rows.itertuples(), replies):
                d = parse_reply(rep)
                if d is None:
                    continue
                n_ok += 1
                d["source_index"] = int(r.source_index)
                recs.append(d)
        tmp = os.path.join(out, f"labels_{s:04d}.jsonl.tmp")
        with open(tmp, "w") as f:
            for d in recs:
                f.write(json.dumps(d) + "\n")
        os.replace(tmp, os.path.join(out, f"labels_{s:04d}.jsonl"))
        dt = time.time() - t0
        print(f"shard {s}: {n_ok}/{len(part)} parsed in {dt:.0f}s ({len(part) / dt:.1f} img/s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="data/labels")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shards", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    run(a.work, a.out, a.model, a.shard_size, a.batch, a.backend, a.limit, shards=a.shards)


if __name__ == "__main__":
    main()
