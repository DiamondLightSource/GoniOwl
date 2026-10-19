"""
Read all CSV log files from /dls_sw/i23/logs/GoniOwl, run fresh GoniOwl
inference on each ECAM_6 image, and produce:
  - Annotated images saved as {confidence}_{decision}_{original}.png
  - A CSV summary of all results
  - Confidence histograms split by human decision (pin on vs pin off)

Example:
  python infer_all.py \
    --model-path ./outputs/checkpoints/model.pt
"""

import os
import csv
import glob
import argparse
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GoniOwl.model import load_checkpoint, predict_proba, select_device
from GoniOwl.preprocess import load_image


LOG_DIR = "/dls_sw/i23/logs/GoniOwl"
START_DATE_CUTOFF = "2023-01-01"
DODGY_DATA = ["2024-08-01-09-02-27"]
OUTPUT_DIR = "outputs/inferall"


def setup_logging() -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    return logging.getLogger(__name__)


logger = setup_logging()


def load_log_data(log_dir: str = LOG_DIR) -> pd.DataFrame:
    """Load all CSV logs, apply filters, and normalise labels."""
    csvs = glob.glob(os.path.join(log_dir, "*.csv"))
    csvs = [f for f in csvs if os.path.basename(f)[:10] >= START_DATE_CUTOFF]

    if not csvs:
        logger.warning("No CSV files found in %s", log_dir)
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(f, header=None) for f in csvs), ignore_index=True)
    df.columns = ["DateTime", "Histogram", "GoniOwl", "Human", "Image1", "Image2"]

    df["Histogram"] = df["Histogram"].replace({
        "histogram_pin_detected": "on",
        "histogram_pin_not_detected": "off",
        "histogram_not_matched": "fail",
    })
    df["GoniOwl"] = df["GoniOwl"].replace({
        "goniowl_pin_off": "off", "goniowl_pin_on": "on",
        "ERROR_READING_PV": "error", "goniowl_dark": "dark",
        "goniowl_light": "light", "PIN_ON": "on", "PIN_OFF": "off",
    })
    df["Human"] = df["Human"].replace({"human_no": "off", "human_yes": "on"})

    df = df[~df["DateTime"].isin(DODGY_DATA)]
    df = df[~df["GoniOwl"].isin(["error", "light", "dark"])]

    logger.info("Loaded %d records after filtering.", len(df))
    return df


def save_annotated_image(
    image_path: str,
    output_path: str,
    human: str,
    hist_decision: str,
    goniowl_decision: str,
    confidence_pct: int,
) -> None:
    """Save the original image with decision info annotated below it."""
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis("off")

    info_text = (
        f"Human: {human}    |    Histogram: {hist_decision}    |    "
        f"GoniOwl: {goniowl_decision}    |    Confidence: {confidence_pct:.2f}%"
    )
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_annotated_image_worker(args: tuple) -> None:
    """Worker function for parallel image saving (runs in separate process)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_path, out_path, human, hist_decision, goniowl_decision, confidence_pct = args
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis("off")

    info_text = (
        f"Human: {human}    |    Histogram: {hist_decision}    |    "
        f"GoniOwl: {goniowl_decision}    |    Confidence: {confidence_pct:.2f}%"
    )
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_inference_all(
    df: pd.DataFrame,
    model_path: str,
    img_height: int | None = None,
    img_width: int | None = None,
    batch_size: int = 32,
    max_workers: int | None = None,
    crop_border: int = 0,
    device: str = "auto",
) -> list[dict]:
    logger.info("Loading model from: %s", model_path)
    model, info = load_checkpoint(model_path, select_device(device))
    img_height = img_height or info["img_height"]
    img_width = img_width or info["img_width"]
    logger.info("Model loaded successfully (input %dx%d).", img_height, img_width)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter to existing images and collect metadata
    rows = []
    for _, row in df.iterrows():
        img_path = row["Image1"]
        if not os.path.exists(img_path):
            logger.warning("Image not found, skipping: %s", img_path)
            continue
        rows.append(row)

    total = len(rows)
    logger.info("Running batched inference on %d images (batch_size=%d).", total, batch_size)

    results = []
    save_tasks = []

    # Process in batches for inference
    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch_rows)
        logger.info("[%d-%d/%d] Loading and predicting batch...", batch_start + 1, batch_end, total)

        # Load all images in this batch
        img_arrays = []
        valid_rows = []
        for row in batch_rows:
            try:
                img_arrays.append(load_image(row["Image1"], (img_height, img_width), border=crop_border))
                valid_rows.append(row)
            except Exception as e:
                logger.error("Error loading %s: %s", row["Image1"], e)

        if not img_arrays:
            continue

        # Batch predict
        batch_tensor = np.stack(img_arrays, axis=0)
        predictions = predict_proba(model, batch_tensor, batch_size=batch_size)

        for row, prediction in zip(valid_rows, predictions):
            pred = float(prediction)
            img_path = row["Image1"]
            human_label = row["Human"]
            histogram_label = row["Histogram"]

            if pred > 0.5:
                goniowl_decision = "on"
            elif pred < 0.5:
                goniowl_decision = "off"
            else:
                goniowl_decision = "undetermined"

            confidence = pred if pred > 0.5 else 1 - pred
            confidence_pct = round(confidence * 100, 2)

            original_name = os.path.basename(img_path)
            out_name = f"{confidence_pct:.2f}_{goniowl_decision}_{original_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            save_tasks.append((
                img_path, out_path, human_label, histogram_label,
                goniowl_decision, confidence_pct,
            ))

            results.append({
                "image_name": original_name,
                "histogram_decision": histogram_label,
                "human_decision": human_label,
                "goniowl_decision": goniowl_decision,
                "goniowl_confidence": confidence_pct,
            })

    # Save annotated images in parallel
    logger.info("Saving %d annotated images in parallel...", len(save_tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_save_annotated_image_worker, task): task[1] for task in save_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving images"):
            try:
                future.result()
            except Exception as e:
                logger.error("Error saving %s: %s", futures[future], e)

    return results


def save_csv(results: list[dict]) -> None:
    if not results:
        logger.warning("No results to save.")
        return

    csv_path = os.path.join(OUTPUT_DIR, "infer_all_results.csv")
    fieldnames = ["image_name", "histogram_decision", "human_decision",
                  "goniowl_decision", "goniowl_confidence"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved to: %s", csv_path)


def generate_confidence_histograms(results: list[dict], output_dir: str = OUTPUT_DIR) -> None:
    """Generate separate confidence histograms split by human decision."""
    os.makedirs(output_dir, exist_ok=True)
    pin_on_conf = [r["goniowl_confidence"] for r in results if r["human_decision"] == "on"]
    pin_off_conf = [r["goniowl_confidence"] for r in results if r["human_decision"] == "off"]

    for label, confidences, color in [
        ("human_pin_on", pin_on_conf, "#4c72b0"),
        ("human_pin_off", pin_off_conf, "#dd8452"),
    ]:
        if not confidences:
            logger.warning("No images for %s, skipping histogram.", label)
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(confidences, bins=20, range=(50, 100), color=color, edgecolor="black", alpha=0.85)
        ax.set_title(f"GoniOwl Confidence Distribution - {label}", fontsize=14)
        ax.set_xlabel("Confidence (%)")
        ax.set_yscale("log")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        hist_path = os.path.join(output_dir, f"confidence_histogram_{label}.png")
        plt.savefig(hist_path, dpi=120)
        plt.close(fig)
        logger.info("Confidence histogram saved: %s", hist_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fresh GoniOwl inference on all logged ECAM_6 images."
    )
    parser.add_argument("--model-path", default=None, help="Path to the trained model.")
    parser.add_argument("--histograms-only", action="store_true",
                        help="Regenerate histograms from existing CSV without running inference.")
    parser.add_argument("--run-path", default=None,
                        help="Path to a run directory (e.g. outputs/inferall_modelid). Used with --histograms-only.")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Directory of GoniOwl CSV logs.")
    parser.add_argument("--img-height", type=int, default=None, help="Image height for inference (default: from the checkpoint).")
    parser.add_argument("--img-width", type=int, default=None, help="Image width for inference (default: from the checkpoint).")
    parser.add_argument("--crop-border", type=int, default=0, help="Pixels to crop from each side before resizing (use 100 for raw, uncropped camera frames).")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps or cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers for saving images.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.histograms_only:
        run_dir = args.run_path or OUTPUT_DIR
        csv_path = os.path.join(run_dir, "infer_all_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No existing results CSV found at: {csv_path}")
        results = pd.read_csv(csv_path).to_dict("records")
        logger.info("Loaded %d results from %s", len(results), csv_path)
        generate_confidence_histograms(results, output_dir=run_dir)
    else:
        if not args.model_path or not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model not found: {args.model_path}")

        df = load_log_data(args.log_dir)
        if df.empty:
            logger.info("No data found. Nothing to do.")
        else:
            results = run_inference_all(
                df,
                model_path=args.model_path,
                img_height=args.img_height,
                img_width=args.img_width,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                crop_border=args.crop_border,
                device=args.device,
            )
            save_csv(results)
            generate_confidence_histograms(results)
            logger.info("Done. Total processed: %d images.", len(results))
