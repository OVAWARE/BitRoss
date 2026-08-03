import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from model import CVAE, TextEncoder, HIDDEN_DIM, LATENT_DIM

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


def generate_image(model, text_prompt, device, input_image=None, img_control=0.5):
    encoded_input = tokenizer(
        text_prompt, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        text_encoding = model.text_encoder(input_ids, attention_mask)
        z = torch.randn(1, LATENT_DIM, device=device)
        generated_image = model.decode(z, text_encoding)

    if input_image is not None:
        input_image = input_image.convert("RGBA").resize((16, 16), resample=Image.NEAREST)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
        # Match training normalization range [-1, 1]
        input_image = input_image * 2 - 1
        generated_image = img_control * input_image + (1 - img_control) * generated_image

    generated_image = generated_image.squeeze(0).cpu()
    generated_image = (generated_image + 1) / 2
    generated_image = generated_image.clamp(0, 1)
    return transforms.ToPILImage()(generated_image)


def load_model(model_path, device, freeze_bert=True):
    text_encoder = TextEncoder(
        hidden_size=HIDDEN_DIM, output_size=HIDDEN_DIM, freeze_bert=freeze_bert
    )
    model = CVAE(text_encoder).to(device)
    state = torch.load(model_path, map_location=device)
    # Allow checkpoints saved under torch.compile (_orig_mod. prefix)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
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
                model, prompt, device, input_image, args.img_control
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
                safe_prompt = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt)[
                    :80
                ]
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
