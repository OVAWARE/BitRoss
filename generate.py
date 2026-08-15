import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from model import (
    CVAE,
    IMAGE_SIZE,
    LATENT_DIM,
    TEXT_DIM,
    TextEncoder,
    decode_with_cfg,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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
    cfg_scale=2.0,
    temperature=1.0,
):
    encoded_input = tokenizer(
        text_prompt, padding=True, truncation=True, max_length=80, return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        text_encoding = model.text_encoder(input_ids, attention_mask)
        z = torch.randn(1, LATENT_DIM, device=device) * temperature

        if input_image is not None:
            x = input_image.convert("RGBA").resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST
            )
            x = transforms.ToTensor()(x).unsqueeze(0).to(device)
            x = x * 2 - 1
            mu, logvar = model.encode(x, text_encoding)
            z_img = model.reparameterize(mu, logvar)
            z = img_control * z_img + (1.0 - img_control) * z

        generated_image = decode_with_cfg(model, z, text_encoding, cfg_scale=cfg_scale)

    generated_image = generated_image.squeeze(0).float().cpu()
    generated_image = quantize_u8((generated_image + 1) / 2).clamp(0, 1)
    return transforms.ToPILImage()(generated_image)


def _strip_compile_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def load_model(model_path, device, freeze_bert=True):
    text_encoder = TextEncoder(
        hidden_size=TEXT_DIM,
        output_size=TEXT_DIM,
        freeze_bert=freeze_bert,
        unfreeze_last_n=0,
    )
    model = CVAE(text_encoder, latent_dim=LATENT_DIM, text_dim=TEXT_DIM).to(device)
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
        description="Generate an image from a text prompt using the trained CVAE model(s)."
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
        help="Control how much the input image influences the output (0 to 1)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale (1 = off)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Latent noise temperature (<1 is more typical, >1 is more diverse)",
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
