"""
Generate a histogram of average image intensity for all images in a directory tree.
Also writes a CSV with per-image averages and summary statistics.
Useful for determining high/low values for over/under exposed images

Example:
  python intensity_histogram.py \
    --input-dir ./outputs/disagreements \
    --output-dir ./outputs/intensity_report \
    --bins 50 --workers 8
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate histogram and stats for average image intensity."
    )
    parser.add_argument("--input-dir", required=True, help="Root directory of images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker threads (0 uses max CPUs - 1).")
    return parser.parse_args()


def average_intensity(img: Image.Image) -> float:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    return float(gray.mean())


def process_image(image_path: Path) -> tuple[str, float]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        avg_val = average_intensity(img)
    return str(image_path), avg_val


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    image_paths = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    per_image = []
    total = len(image_paths)
    if total == 0:
        raise SystemExit(f"No images found under {input_dir}")

    max_cpus = os.cpu_count() or 1
    default_workers = max(max_cpus - 1, 1)
    workers = default_workers if args.workers <= 0 else args.workers

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_image, p) for p in image_paths]
        for future in as_completed(futures):
            per_image.append(future.result())
            completed += 1
            percent = (completed / total) * 100
            print(f"Processed {completed}/{total} images ({percent:.1f}%)", end="\r")

    print()

    values = np.array([v for _, v in per_image], dtype=np.float32)
    stats = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "stdev": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=args.bins, color="#4c72b0", edgecolor="black")
    plt.title("Histogram of Average Image Intensity")
    plt.xlabel("Average Intensity (0-255)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    hist_path = output_dir / "average_intensity_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=120)
    plt.close()

    # CSV output
    csv_path = output_dir / "average_intensity_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key in ["count", "mean", "stdev", "min", "max", "median", "p10", "p90"]:
            writer.writerow([key, stats[key]])
        writer.writerow([])
        writer.writerow(["image_path", "average_intensity"])
        for path, avg_val in per_image:
            writer.writerow([path, avg_val])


if __name__ == "__main__":
    main()
