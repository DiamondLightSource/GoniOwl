"""
Adjust brightness and contrast for images in a directory.

Example:
  python adjust_brightness_contrast.py \
    --input-dir ./outputs/disagreements \
    --output-dir ./outputs/adjusted \
    --brightness-delta 20 \
    --contrast-delta 0.1
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjust brightness and contrast for images in a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save adjusted images.")
    parser.add_argument(
        "--brightness-delta",
        type=float,
        default=0.0,
        help="Brightness delta to add (can be negative).",
    )
    parser.add_argument(
        "--contrast-delta",
        type=float,
        default=0.0,
        help="Contrast delta to add to 1.0 (e.g. 0.1 means 10%% more contrast; can be negative).",
    )
    return parser.parse_args()


def adjust_image(img: Image.Image, brightness_delta: float, contrast_delta: float) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    contrast_factor = 1.0 + contrast_delta
    adjusted = (arr - 128.0) * contrast_factor + 128.0 + brightness_delta
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def average_brightness(arr: np.ndarray) -> int:
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    return int(round(gray.mean()))


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            adjusted_arr = adjust_image(img, args.brightness_delta, args.contrast_delta)
            avg_val = average_brightness(adjusted_arr)

            new_name = f"{image_path.stem}_{avg_val}{image_path.suffix}"
            output_path = output_dir / new_name

            Image.fromarray(adjusted_arr).save(output_path)


if __name__ == "__main__":
    main()
