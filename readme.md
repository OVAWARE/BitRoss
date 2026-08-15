# BitRoss

Donate ETH: 0xe0D9431D2404CF5cAeb16610626475003A59D427

## Installation

To use BitRoss, you need to have Python installed. Clone the repository and install the required dependencies.

```bash
git clone https://github.com/OVAWARE/BitRoss.git
cd BitRoss
pip install -r requirements.txt
```

## Usage

### `generate.py`

Use the trained model to generate assets

**Usage:**

```bash
python generate.py [options]
```

**Options:**

- `--prompt`: **Text prompt for image generation**  
  Example: `"a beautiful sunset"`

- `--prompt_file`: **File containing prompts, one per line**  
  Example: `prompts.txt`

- `--output`: **Output directory or file for generated images**  
  Default: `generated_images`  
  Example: `output_folder`

- `--model_paths`: **Paths to the trained model(s)**  
  Example: `model1.pth model2.pth`

- `--model_path`: **Path to a single trained model**  
  Example: `model.pth`

- `--clean`: **Clean up the image by removing low opacity pixels**  
  Example: `--clean`

- `--size`: **Resize the generated image after generation**  
  Default: `16`  
  Example: `1024`

- `--input_image`: **Path to the input image for img2img generation**  
  Example: `input.jpg`

- `--img_control`: **Control how much the input image influences the output (0 to 1)**  
  Default: `0.5`  
  Example: `0.7`

### `train.py`

```bash
python train.py --data_dir ./training-items --save_dir ./models/BitRoss
```

### Colab Pro

This Cursor environment has no GPU. Train on Colab instead:

[Open BitRoss_train.ipynb in Colab](https://colab.research.google.com/github/OVAWARE/BitRoss/blob/cursor/hq-cvae-architecture-5a9f/BitRoss_train.ipynb)

Set **Runtime → Change runtime type → L4** (T4 if L4 is unavailable; skip A100). Point it at a `training-items/` folder with `metadata.json`. Checkpoints save to Google Drive.

## Demo

You can try out BitRoss on the live demo available at [Hugging Face Spaces](https://huggingface.co/spaces/OVAWARE/BitRoss).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please open an issue on the GitHub repository or contact me at [OVAWARE@proton.me](mailto:OVAWARE@proton.me).
