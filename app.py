"""BitRoss Studio: a Gradio UI for the BitRoss 2 Minecraft texture model.

    python app.py --model runs/bitross-small/bitross.pt          # local checkpoint
    python app.py --model OVAWARE/BitRoss-2                       # Hugging Face repo
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time
import zipfile
from typing import List, Optional

import gradio as gr
import numpy as np
from PIL import Image

from bitross import BitRoss
from bitross.minecraft import MATERIAL_SET, material_set, mod_assets, resource_pack, slug, sprite_sheet
from bitross.pipeline import parse_colors, seam_score, tile_preview, upscale

CSS = """
.pixel img, .pixel canvas { image-rendering: pixelated !important; image-rendering: crisp-edges !important; }
#title { text-align: center; margin-bottom: 0; }
.tex16 img { width: 256px !important; height: 256px !important; object-fit: contain; }
#subtitle { text-align: center; opacity: .75; margin-top: 0; }
"""
SHOW = 16  # display upscale factor
CHECKER = None


def checker(size: int = 256, cell: int = 16) -> Image.Image:
    global CHECKER
    if CHECKER is None or CHECKER.width != size:
        a = (np.add.outer(np.arange(size) // cell, np.arange(size) // cell) % 2)
        g = np.where(a[..., None] == 0, 68, 58).astype(np.uint8)
        CHECKER = Image.fromarray(np.concatenate([g, g, g + 6, np.full_like(g, 255)], -1), "RGBA")
    return CHECKER


def show(img: Image.Image) -> Image.Image:
    big = upscale(img, SHOW)
    bg = checker(big.width).copy()
    bg.alpha_composite(big)
    return bg


def zip_pngs(images: List[Image.Image], names: Optional[List[str]] = None, scale: int = 1) -> str:
    path = os.path.join(tempfile.mkdtemp(), "bitross_textures.zip")
    with zipfile.ZipFile(path, "w") as z:
        for i, im in enumerate(images):
            b = io.BytesIO()
            (upscale(im, scale) if scale > 1 else im).save(b, "PNG")
            z.writestr(f"{(names[i] if names else f'texture_{i:02d}')}.png", b.getvalue())
    return path


def save_bytes(data: bytes, name: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def build_ui(br: BitRoss) -> gr.Blocks:
    with gr.Blocks(title="BitRoss Studio") as demo:
        gr.Markdown("# BitRoss Studio", elem_id="title")
        gr.Markdown(f"Text → 16×16 Minecraft textures with a masked-diffusion pixel transformer "
                    f"({br.model.num_params() / 1e6:.0f}M params, {len(br.palette)}-color palette, step {br.step:,})",
                    elem_id="subtitle")
        last = gr.State([])

        # ------------------------------------------------------------------ Create
        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", value="ruby sword", lines=2)
                    with gr.Row():
                        kind = gr.Radio(["item", "block", "any"], value="item", label="Type")
                        tile = gr.Radio(["auto", "seamless", "standalone"], value="auto", label="Tiling")
                    with gr.Row():
                        n = gr.Slider(1, 16, 8 if br.device.type == "cuda" else 4, step=1,
                                      label="Images" + ("" if br.device.type == "cuda" else " (CPU: ~6 s each)"))
                        seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                    with gr.Accordion("Sampling", open=False):
                        steps = gr.Slider(4, 48, 16, step=1, label="Steps")
                        cfg = gr.Slider(1, 8, 3.0, step=0.25, label="Guidance (CFG)")
                        temp = gr.Slider(0, 1.5, 1.0, step=0.05, label="Temperature (0 = greedy)")
                        rnd = gr.Slider(0, 6, 2.0, step=0.25, label="Order randomness")
                        polish = gr.Slider(0, 8, 0, step=1, label="Polish passes")
                        quality = gr.Radio(["high", "mid", "any"], value="high", label="Quality conditioning")
                        rerank = gr.Checkbox(False, label="Sort results by CLIP score")
                    with gr.Accordion("Colors", open=False):
                        pal = gr.Textbox(label="Palette lock (hex colors, space separated)", placeholder="#1b1b1b #6b2f2f #c4453f")
                        pal_img = gr.Image(label="…or lock to the colors of an image", type="pil", image_mode="RGBA", height=140)
                        pal_mode = gr.Radio(["soft", "hard"], value="soft", label="Palette mode (soft = steer then snap, hard = strict)")
                        pal_tol = gr.Slider(0, 0.1, 0.0, step=0.005, label="Palette tolerance (OKLab, hard mode)")
                        prefer = gr.Textbox(label="Prefer colors (soft)", placeholder="#3fbf5f #1e7a3a")
                        maxc = gr.Slider(0, 32, 0, step=1, label="Max colors (0 = off)")
                    go = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="Results", columns=4, height=560, elem_classes="pixel", object_fit="contain")
                    status = gr.Markdown()
                    with gr.Row():
                        dl16 = gr.DownloadButton("16×16 PNGs (.zip)")
                        dlbig = gr.DownloadButton("Upscaled ×16 (.zip)")
                        dlsheet = gr.DownloadButton("Sprite sheet")

            def run(prompt, kind, tile, n, seed, steps, cfg, temp, rnd, polish, quality, rerank, pal, pal_img,
                    pal_mode, pal_tol, prefer, maxc):
                lock = None
                if pal and pal.strip():
                    lock = pal
                elif pal_img is not None:
                    lock = pal_img
                kw = dict(n=int(n), asset_type=kind, tileable=None if tile == "auto" else tile == "seamless",
                          quality=quality, steps=int(steps), cfg_scale=float(cfg), temperature=float(temp),
                          randomness=float(rnd), polish=int(polish), seed=None if seed is None or seed < 0 else int(seed),
                          palette=lock, palette_tolerance=float(pal_tol), palette_mode=pal_mode,
                          prefer_colors=prefer if prefer and prefer.strip() else None)
                t0 = time.time()
                frame = None
                for st in br.stream(prompt, **kw):
                    frame = st
                    imgs = br.decode(st.tokens)
                    yield [show(i) for i in imgs], f"step {st.index + 1}/{st.total} ({st.phase})", [], None, None, None
                imgs = br.decode(frame.tokens)
                if lock is not None:
                    from bitross.pipeline import snap_to_palette
                    imgs = [snap_to_palette(i, lock) for i in imgs]
                if maxc and maxc > 0:
                    from bitross.pipeline import reduce_colors
                    imgs = [reduce_colors(i, int(maxc)) for i in imgs]
                if rerank:
                    sc = br.clip_scores(imgs, [prompt] * len(imgs))
                    order = np.argsort(-np.array(sc))
                    imgs = [imgs[i] for i in order]
                base = slug(prompt)
                names = [f"{base}_{i:02d}" for i in range(len(imgs))]
                msg = f"done in {time.time() - t0:.1f}s"
                if kind == "block":
                    msg += " · seam scores: " + ", ".join(f"{seam_score(i):.2f}" for i in imgs[:8]) + " (<1.3 tiles cleanly)"
                yield ([show(i) for i in imgs], msg, imgs, zip_pngs(imgs, names), zip_pngs(imgs, names, 16),
                       save_bytes(_png(sprite_sheet(imgs, cols=min(8, len(imgs)))), f"{base}_sheet.png"))

            go.click(run, [prompt, kind, tile, n, seed, steps, cfg, temp, rnd, polish, quality, rerank, pal, pal_img,
                           pal_mode, pal_tol, prefer, maxc], [gallery, status, last, dl16, dlbig, dlsheet])

        # ------------------------------------------------------------------ Edit
        with gr.Tab("Edit"):
            gr.Markdown("Pick a result from **Create** (or upload a 16×16 texture), then: **Variations** redraws a random "
                        "share of pixels, **Recolor** keeps the exact silhouette and redraws the inside from the prompt, "
                        "and **Inpaint** regenerates only the pixels you paint over.")
            with gr.Row():
                with gr.Column():
                    pick = gr.Number(0, label="Use result # from Create", precision=0)
                    use = gr.Button("Load result")
                    src = gr.Image(label="Source (16×16)", type="pil", image_mode="RGBA", elem_classes=["pixel", "tex16"], height=280)
                    eprompt = gr.Textbox(label="Prompt", value="emerald sword")
                    ekind = gr.Radio(["item", "block", "any"], value="item", label="Type")
                    strength = gr.Slider(0.05, 1.0, 0.4, step=0.05, label="Variation strength")
                    en = gr.Slider(1, 16, 8, step=1, label="Images")
                    eseed = gr.Number(-1, label="Seed", precision=0)
                    with gr.Row():
                        bvar = gr.Button("Variations")
                        brec = gr.Button("Recolor (keep shape)", variant="primary")
                    editor = gr.ImageEditor(label="Inpaint: paint the pixels to regenerate", type="pil", image_mode="RGBA",
                                            elem_classes="pixel", height=320,
                                            brush=gr.Brush(colors=["#ffffff"], default_size=16, color_mode="fixed"))
                    binp = gr.Button("Inpaint")
                with gr.Column():
                    egal = gr.Gallery(label="Results", columns=4, height=560, elem_classes="pixel")
                    edl = gr.DownloadButton("Download (.zip)")

            def load(results, i):
                if not results:
                    raise gr.Error("Generate something in the Create tab first.")
                im = results[int(np.clip(i, 0, len(results) - 1))]
                return im, show(im)

            use.click(load, [last, pick], [src, editor])
            src.upload(lambda im: show(im.convert("RGBA").resize((16, 16), Image.NEAREST)) if im else None, src, editor)

            def _seed(s):
                return None if s is None or s < 0 else int(s)

            def variations(im, p, k, s, n, seed):
                if im is None:
                    raise gr.Error("Load or upload a source texture.")
                imgs = br.generate(p, n=int(n), asset_type=k, init=im, strength=float(s), seed=_seed(seed))
                return [show(i) for i in imgs], zip_pngs(imgs)

            def recolor(im, p, k, n, seed):
                if im is None:
                    raise gr.Error("Load or upload a source texture.")
                imgs = br.generate(p, n=int(n), asset_type=k, silhouette=im, seed=_seed(seed))
                return [show(i) for i in imgs], zip_pngs(imgs)

            def inpaint(im, ed, p, k, n, seed):
                if im is None or ed is None:
                    raise gr.Error("Load a source texture and paint a mask.")
                layers = ed.get("layers") or []
                if not layers:
                    raise gr.Error("Paint over the pixels you want to regenerate.")
                a = np.zeros((16, 16), bool)
                for L in layers:
                    al = np.asarray(L.convert("RGBA").resize((16, 16), Image.BOX))[..., 3]
                    a |= al > 40
                if not a.any():
                    raise gr.Error("The mask is empty.")
                mask = Image.fromarray(np.where(a[..., None], 255, 0).repeat(4, -1).astype(np.uint8), "RGBA")
                imgs = br.generate(p, n=int(n), asset_type=k, init=im, mask=mask, seed=_seed(seed))
                return [show(i) for i in imgs], zip_pngs(imgs)

            bvar.click(variations, [src, eprompt, ekind, strength, en, eseed], [egal, edl])
            brec.click(recolor, [src, eprompt, ekind, en, eseed], [egal, edl])
            binp.click(inpaint, [src, editor, eprompt, ekind, en, eseed], [egal, edl])

        # ------------------------------------------------------------------ Material set
        with gr.Tab("Material Set"):
            gr.Markdown("Generate a full, color-consistent family for a new material — ingot/gem, tools, armor, ore and "
                        "storage block — and export it as a mod asset folder or a resource pack.")
            with gr.Row():
                with gr.Column():
                    mat = gr.Textbox(label="Material", value="ruby")
                    kinds = gr.CheckboxGroup([k for k, *_ in MATERIAL_SET], value=[k for k, *_ in MATERIAL_SET], label="Pieces")
                    cand = gr.Slider(1, 8, 4, step=1, label="Candidates per piece (best by CLIP)")
                    steer = gr.Slider(0, 5, 2.5, step=0.25, label="Color consistency")
                    mseed = gr.Number(0, label="Seed", precision=0)
                    modid = gr.Textbox(label="Mod id (for mod export)", value="mymod")
                    packfmt = gr.Number(46, label="Resource pack format", precision=0)
                    mgo = gr.Button("Generate set", variant="primary")
                with gr.Column():
                    mgal = gr.Gallery(label="Set", columns=5, height=520, elem_classes="pixel")
                    with gr.Row():
                        mdl_mod = gr.DownloadButton("Mod assets (.zip)")
                        mdl_rp = gr.DownloadButton("Resource pack (.zip)")

            def make_set(mat, kinds, cand, steer, seed, modid, packfmt):
                res = material_set(br, mat, kinds=kinds, seed=int(seed or 0), candidates=int(cand), steer=float(steer))
                names = {f"{slug(mat)}_{k}" if k not in ("block",) else f"{slug(mat)}_block": im for k, im in res.items()}
                kinds_map = {n_: ("block" if n_.endswith(("_ore", "_block")) else "item") for n_ in names}
                gal = [(show(im), n_) for n_, im in names.items()]
                mod_zip = save_bytes(mod_assets(names, kinds_map, modid=modid), f"{slug(modid)}_{slug(mat)}_assets.zip")
                rp_zip = save_bytes(resource_pack(names, kinds_map, description=f"BitRoss {mat} set",
                                                  pack_format=int(packfmt)), f"{slug(mat)}_resource_pack.zip")
                return gal, mod_zip, rp_zip

            mgo.click(make_set, [mat, kinds, cand, steer, mseed, modid, packfmt], [mgal, mdl_mod, mdl_rp])

        # ------------------------------------------------------------------ Tiling
        with gr.Tab("Tiling check"):
            with gr.Row():
                timg = gr.Image(label="Texture", type="pil", image_mode="RGBA", height=280, elem_classes=["pixel", "tex16"])
                tout = gr.Image(label="3×3 preview", elem_classes="pixel", height=420)
            tmsg = gr.Markdown()
            tpick = gr.Number(0, label="Use result # from Create", precision=0)
            tuse = gr.Button("Load result")

            def tcheck(im):
                if im is None:
                    return None, ""
                im = im.convert("RGBA").resize((16, 16), Image.NEAREST)
                s = seam_score(im)
                return tile_preview(im), f"Seam score **{s:.2f}** — {'tiles cleanly' if s < 1.3 else 'visible seams'}"

            timg.change(tcheck, timg, [tout, tmsg])
            tuse.click(lambda r, i: r[int(np.clip(i, 0, len(r) - 1))] if r else None, [last, tpick], timg)

    return demo


def _png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("BITROSS_MODEL", "runs/bitross-small/bitross.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    br = BitRoss.from_pretrained(a.model, device=a.device)
    build_ui(br).queue().launch(share=a.share, server_port=a.port, css=CSS)


if __name__ == "__main__":
    main()
