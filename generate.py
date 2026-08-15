import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer

from model import (
    CLIP_ID,
    IMAGE_SIZE,
    BitRoss,
    sample_rectified_flow,
    tokenize_prompts,
)

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)
    return _tokenizer


def clean_image(image, threshold=0.75):
    """
    Clean up the image by setting pixels with opacity <= threshold to 0% opacity
    and pixels above the threshold to 100% visibility.
    """
    np_image = np.array(image)
    alpha_channel = np_image[:, :, 3]
    alpha_channel[alpha_channel <= int(threshold * 255)] = 0
    alpha_channel[alpha_channel > int(threshold * 255)] = 255
    return Image.fromarray(np_image)


def quantize_u8(image_01):
    """Snap decoder output onto the 8-bit grid pixel art actually lives on."""
    return (image_01 * 255.0).round().clamp(0, 255) / 255.0


def generate_image(
    model,
    text_prompt,
    device,
    input_image=None,
    img_control=0.5,
    cfg_scale=2.5,
    temperature=1.0,
    steps=20,
    tokenizer=None,
    sampler="heun",
):
    tokenizer = tokenizer or get_tokenizer()
    input_ids, attention_mask = tokenize_prompts(tokenizer, [text_prompt], device)

    with torch.no_grad():
        pooled, seq = model.encode_text(input_ids, attention_mask, drop_p=0.0)
        x_start = None
        t_start = 1.0
        if input_image is not None:
            x = input_image.convert("RGBA").resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST
            )
            x = transforms.ToTensor()(x).unsqueeze(0).to(device)
            x = x * 2 - 1
            t_start = float(min(max(1.0 - img_control, 1e-3), 1.0))
            noise = torch.randn_like(x) * temperature
            x_start = (1.0 - t_start) * x + t_start * noise

        generated = sample_rectified_flow(
            model,
            pooled,
            seq,
            steps=steps,
            cfg_scale=cfg_scale,
            temperature=temperature,
            sampler=sampler,
            x_start=x_start,
            t_start=t_start,
            device=device,
        )

    generated = generated.squeeze(0).float().cpu()
    generated = quantize_u8((generated + 1) / 2).clamp(0, 1)
    return transforms.ToPILImage()(generated)


def _strip_compile_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def load_model(model_path, device):
    model = BitRoss(clip_id=CLIP_ID).to(device)
    try:
        raw = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(model_path, map_location=device)
    if isinstance(raw, dict) and ("ema" in raw or "model" in raw):
        state = raw.get("ema") or raw["model"]
    else:
        state = raw
    state = _strip_compile_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading {model_path}: {missing[:8]}...")
    if unexpected:
        print(f"Warning: unexpected keys when loading {model_path}: {unexpected[:8]}...")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 16x16 item sprite from a text prompt (rectified-flow DiT)."
    )
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--prompt_file", type=str, help="File containing prompts, one per line"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_images",
        help="Output directory or file for generated images",
    )
    parser.add_argument(
        "--model_paths", type=str, nargs="*", help="Paths to the trained model(s)"
    )
    parser.add_argument("--model_path", type=str, help="Path to a single trained model")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up the image by removing low opacity pixels",
    )
    parser.add_argument(
        "--size", type=int, default=16, help="Size of the generated image"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        help="Path to the input image for img2img generation",
    )
    parser.add_argument(
        "--img_control",
        type=float,
        default=0.5,
        help="How much the input image is preserved (0 = ignore, 1 = keep)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (1 = off)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Initial noise scale (<1 typical, >1 diverse)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Rectified-flow integration steps (Heun uses 2 NFE each except the last)",
    )
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        parser.error("Either --prompt or --prompt_file must be provided")

    if args.model_paths and args.model_path:
        parser.error("Specify either --model_paths or --model_path, not both")

    model_paths = args.model_paths if args.model_paths else [args.model_path]
    if not model_paths or model_paths == [None]:
        parser.error("Provide --model_path or --model_paths")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    is_folder_output = os.path.isdir(args.output) or not args.output.endswith(
        (".png", ".jpg", ".jpeg", ".webp")
    )
    if is_folder_output:
        os.makedirs(args.output, exist_ok=True)

    input_image = None
    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGBA")

    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    for model_path in model_paths:
        model = load_model(model_path, device)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        for i, prompt in enumerate(prompts):
            start_time = time.time()
            generated_image = generate_image(
                model,
                prompt,
                device,
                input_image,
                args.img_control,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                steps=args.steps,
                tokenizer=tokenizer,
            )
            generation_time = time.time() - start_time

            if args.clean:
                generated_image = clean_image(generated_image)

            generated_image = generated_image.resize(
                (args.size, args.size), resample=Image.NEAREST
            )

            if not is_folder_output:
                output_file = args.output
            else:
                safe_prompt = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in prompt
                )[:80]
                output_file = os.path.join(
                    args.output, f"{model_name}_{safe_prompt}_{i:03d}.png"
                )

            generated_image.save(output_file)
            print(
                f"Generated image for prompt '{prompt}' using model '{model_name}' saved as {output_file}"
            )
            print(f"Generation time: {generation_time:.6f} seconds")


if __name__ == "__main__":
    main()
