"""Fast CPU tests for the core pieces (no downloads, no GPU)."""

import numpy as np
import pytest
import torch

from bitross.config import preset_config
from bitross.data.features import measure, structure_hash
from bitross.data.names import category_of, parse_name
from bitross.diffusion import mask_ratio, masked_loss, sample_mask
from bitross.model import BitRossModel
from bitross.palette import Palette, canonical_rgba, pack_rgba
from bitross.sampler import sample


def _toy_palette(k=64, seed=0):
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 256, (5000, 4), dtype=np.uint8)
    cols[:, 3] = np.where(rng.random(5000) < 0.9, 255, 128)
    keys, counts = np.unique(pack_rgba(canonical_rgba(cols.reshape(-1, 1, 1, 4)).reshape(-1, 4)), return_counts=True)
    return Palette.build(keys, counts, size=k, n_translucent=4, iters=5)


def _tiny_model(k=64):
    cfg = preset_config("nano", vocab_size=k, text_dim=32, text_len=8, dim=64, heads=4)
    pal = _toy_palette(k)
    return BitRossModel(cfg, torch.from_numpy(pal.feats)), pal


def test_palette_roundtrip_exact_colors():
    pal = _toy_palette()
    img = pal.rgba[np.random.default_rng(1).integers(0, len(pal), 256)].reshape(1, 16, 16, 4)
    toks = pal.encode(img)
    assert toks.shape == (1, 256)
    back = pal.decode(toks)
    assert np.array_equal(canonical_rgba(back), canonical_rgba(img))


def test_palette_transparent_is_zero():
    pal = _toy_palette()
    img = np.zeros((1, 16, 16, 4), np.uint8)
    img[0, :, :, :3] = 200      # color with alpha 0 must still be token 0
    assert (pal.encode(img) == 0).all()


def test_neighbour_matrix_rows_sum_to_one():
    M = _toy_palette().neighbour_matrix()
    assert torch.allclose(M.sum(1), torch.ones(M.shape[0]), atol=1e-5)
    assert M[0, 0] == 1


def test_mask_counts_and_shape_mask():
    x = torch.randint(1, 10, (8, 256))
    x[:, :100] = 0
    r = torch.full((8,), 0.5)
    m = sample_mask(x, r, 16, blob_prob=0.5)
    assert (m.sum(1) == 128).all()
    m2 = sample_mask(x, torch.ones(8), 16, shape_prob=1.0)
    assert not m2[:, :100].any() and m2[:, 100:].all()
    assert (mask_ratio(torch.rand(100)) > 0).all()


def test_model_forward_backward_and_wrap():
    model, _ = _tiny_model()
    B = 3
    x = torch.randint(0, 65, (B, 256))
    t, p = torch.randn(B, 8, 32), torch.randn(B, 32)
    meta = [torch.tensor([1, 2, 0]), torch.tensor([1, 2, 0]), torch.tensor([3, 0, 1])]
    out = model(x, t, p, *meta, drop_text=torch.tensor([True, False, False]), sc_probs=torch.softmax(torch.randn(B, 256, 64), -1))
    assert out.shape == (B, 256, 64)
    m = x == 64
    m[:, 0] = True
    res = masked_loss(out, torch.randint(0, 64, (B, 256)), m, torch.eye(64), 0.1)
    res["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampler_respects_constraints():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    model.eval()
    B = 2
    t, p = torch.randn(B, 8, 32), torch.randn(B, 32)
    meta = [torch.ones(B, dtype=torch.long)] * 3
    allowed = torch.zeros(64, dtype=torch.bool)
    allowed[[0, 5, 7]] = True
    init = torch.full((B, 256), 9)
    keep = torch.zeros(B, 256, dtype=torch.bool)
    keep[:, :50] = True
    x = sample(model, t, p, *meta, steps=6, cfg_scale=2.0, allowed=allowed, init=init, keep=keep, polish_steps=2)
    assert (x[:, :50] == 9).all()                                  # kept pixels untouched
    assert torch.isin(x[:, 50:], torch.tensor([0, 5, 7])).all()    # palette lock
    assert (x < 64).all()                                          # nothing left masked


def test_features_and_names():
    rgba = np.zeros((2, 16, 16, 4), np.uint8)
    rgba[0, 4:12, 4:12] = [200, 30, 30, 255]
    rgba[0, 4, 4:12] = [20, 20, 20, 255]
    rgba[1] = [100, 100, 100, 255]
    f = measure(rgba)
    assert f["n_visible"][0] == 64 and f["n_visible"][1] == 256
    assert f["n_colors"][1] == 1
    assert len(set(structure_hash(rgba))) == 2
    assert parse_name("ruby_pickaxe.png")[0] == "ruby pickaxe"
    assert parse_name("stone_n.png")[3] is True
    assert parse_name("1.png")[2] is False
    assert category_of("ruby pickaxe") == "tool"


def test_reduce_colors():
    from PIL import Image
    from bitross.pipeline import reduce_colors
    a = np.random.default_rng(0).integers(0, 256, (16, 16, 4), dtype=np.uint8)
    a[..., 3] = 255
    out = np.asarray(reduce_colors(Image.fromarray(a, "RGBA"), 6))
    assert len(np.unique(out.reshape(-1, 4), axis=0)) <= 6


def test_snap_to_palette_only_uses_palette_colors():
    from PIL import Image
    from bitross.pipeline import parse_colors, snap_to_palette
    a = np.random.default_rng(0).integers(0, 256, (16, 16, 4), dtype=np.uint8)
    a[..., 3] = np.where(a[..., 3] > 100, 255, 0)
    pal = "#1e1b2e #6a4fb3 #f2e9ff #8a6236"
    out = np.asarray(snap_to_palette(Image.fromarray(a, "RGBA"), pal)).reshape(-1, 4)
    vis = out[out[:, 3] > 0][:, :3]
    allowed = {tuple(c[:3]) for c in parse_colors(pal)}
    assert {tuple(c) for c in vis} <= allowed
