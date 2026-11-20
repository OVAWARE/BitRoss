"""Benchmark / profile one training step with random data (no dataset needed).

    python -m bitross.bench --preset small --batch 256            # img/s with torch.compile
    python -m bitross.bench --preset small --batch 256 --profile  # top CUDA kernels (eager)
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch

from .config import preset_config
from .diffusion import mask_ratio, masked_loss, sample_mask
from .model import BitRossModel


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--self-cond", type=float, default=0.5)
    ap.add_argument("--set", nargs="*", default=[], help="model overrides key=value")
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    over = {}
    for kv in a.set:
        k, v = kv.split("=")
        over[k] = type(getattr(preset_config(a.preset), k))(eval(v))
    cfg = preset_config(a.preset, vocab_size=1024, text_dim=512, **over)
    model = BitRossModel(cfg, torch.rand(1024, 4)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    fwd = model if (a.no_compile or a.profile) else torch.compile(model)
    B = a.batch
    smooth = torch.softmax(torch.randn(1024, 1024, device=dev), -1)
    x0 = torch.randint(0, 1024, (B, 256), device=dev)
    ts, tp = torch.randn(B, cfg.text_len, 512, device=dev, dtype=torch.bfloat16), torch.randn(B, 512, device=dev, dtype=torch.bfloat16)
    meta = [torch.randint(0, 3, (B,), device=dev), torch.randint(0, 3, (B,), device=dev), torch.randint(0, 4, (B,), device=dev)]
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"{model.num_params() / 1e6:.1f}M params, batch {B}")

    def step(i):
        m = sample_mask(x0, mask_ratio(torch.rand(B, device=dev)), 16, 0.25, 0.1)
        x = torch.where(m, torch.full_like(x0, cfg.mask_id), x0)
        sc = None
        if i % 2 == 0 and a.self_cond > 0:
            with torch.no_grad(), amp:
                sc = fwd(x, ts, tp, *meta).float().softmax(-1)
        with amp:
            lg = fwd(x, ts, tp, *meta, sc_probs=sc)
        masked_loss(lg, x0, m, smooth, 0.1)["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for i in range(4):
        step(i)
    torch.cuda.synchronize()
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                  record_shapes=True) if a.profile else contextlib.nullcontext()
    with prof:
        t = time.time()
        n = 6 if a.profile else a.steps
        for i in range(n):
            step(i)
        torch.cuda.synchronize()
        dt = time.time() - t
    print(f"{n * B / dt:.0f} img/s, {dt / n * 1000:.0f} ms/step, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if a.profile:
        rows = [e for e in prof.key_averages(group_by_input_shape=True) if e.key.startswith("aten::")]
        rows.sort(key=lambda e: -e.device_time_total)
        tot = sum(e.self_device_time_total for e in prof.key_averages())
        for e in rows[:30]:
            print(f"{e.device_time_total / 1e3:8.1f} ms {100 * e.device_time_total / tot:5.1f}%  x{e.count:<4d} {e.key:28s} {str(e.input_shapes)[:90]}")


if __name__ == "__main__":
    main()
