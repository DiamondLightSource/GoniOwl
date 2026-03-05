"""
Read all CSV log files from /dls_sw/i23/logs/GoniOwl, run fresh GoniOwl
inference on each ECAM_6 image, and produce:
  - Annotated images saved as {confidence}_{decision}_{original}.png
  - A CSV summary of all results
  - Confidence histograms split by human decision (pin on vs pin off)

Example:
  python infer_all.py \
    --model-path ./outputs/checkpoints/model.keras \
    --img-height 152 --img-width 218
"""

import os
import csv
import glob
import json
import shutil
import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import Model
from typing import cast


LOG_DIR = "/dls_sw/i23/logs/GoniOwl"
START_DATE_CUTOFF = "2023-01-01"
DODGY_DATA = ["2024-08-01-09-02-27"]
OUTPUT_DIR = "outputs/inferall"


def load_model(model_path: str) -> Model:
    """Load a Keras model, patching legacy .h5 files for Keras 3 compatibility."""
    if model_path.endswith(".h5"):
        import h5py

        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()
        shutil.copy(model_path, tmp.name)

        with h5py.File(tmp.name, "r+") as f:
            config = json.loads(f.attrs["model_config"])

            def fix_dtypes(obj: object) -> None:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "dtype":
                            if isinstance(v, str):
                                obj[k] = {"class_name": "DTypePolicy", "config": {"name": v}}
                            elif isinstance(v, dict) and v.get("class_name") == "Policy":
                                v["class_name"] = "DTypePolicy"
                        else:
                            fix_dtypes(v)
                elif isinstance(obj, list):
                    for item in obj:
                        fix_dtypes(item)

            fix_dtypes(config)
            f.attrs["model_config"] = json.dumps(config)

        model = cast(Model, keras.models.load_model(tmp.name, compile=False))
        os.remove(tmp.name)
        return model
    else:
        return cast(Model, keras.models.load_model(model_path, compile=False))


def setup_logging() -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    return logging.getLogger(__name__)


logger = setup_logging()


def load_log_data() -> pd.DataFrame:
    """Load all CSV logs, apply filters, and normalise labels."""
    csvs = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    csvs = [f for f in csvs if os.path.basename(f)[:10] >= START_DATE_CUTOFF]

    if not csvs:
        logger.warning("No CSV files found in %s", LOG_DIR)
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


def run_inference_all(
    df: pd.DataFrame,
    model_path: str,
    img_height: int,
    img_width: int,
) -> list[dict]:
    logger.info("Loading model from: %s", model_path)
    model = load_model(model_path)
    logger.info("Model loaded successfully.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        img_path = row["Image1"]
        histogram_label = row["Histogram"]
        human_label = row["Human"]

        if not os.path.exists(img_path):
            logger.warning("[%d/%d] Image not found, skipping: %s", idx, total, img_path)
            continue

        logger.info("[%d/%d] Processing: %s", idx, total, img_path)

        try:
            img = keras.utils.load_img(img_path, target_size=(img_height, img_width))
            img_array = keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = float(model.predict(img_array, verbose=0)[0][0])

            if prediction > 0.85:
                goniowl_decision = "on"
            elif prediction < 0.15:
                goniowl_decision = "off"
            else:
                goniowl_decision = "undetermined"

            confidence = prediction if prediction > 0.5 else 1 - prediction
            confidence_pct = round(confidence * 100, 2)

            # Save annotated image
            original_name = os.path.basename(img_path)
            out_name = f"{confidence_pct:.2f}_{goniowl_decision}_{original_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            save_annotated_image(
                img_path, out_path,
                human=human_label,
                hist_decision=histogram_label,
                goniowl_decision=goniowl_decision,
                confidence_pct=confidence_pct,
            )

            results.append({
                "image_name": original_name,
                "histogram_decision": histogram_label,
                "human_decision": human_label,
                "goniowl_decision": goniowl_decision,
                "goniowl_confidence": confidence_pct,
            })

            logger.info(
                "  Human=%s, Histogram=%s, GoniOwl=%s (confidence=%.2f%%)",
                human_label, histogram_label, goniowl_decision, confidence_pct,
            )

        except Exception as e:
            logger.error("Error processing %s: %s", img_path, e)

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


def generate_confidence_histograms(results: list[dict]) -> None:
    """Generate separate confidence histograms split by human decision."""
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
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        hist_path = os.path.join(OUTPUT_DIR, f"confidence_histogram_{label}.png")
        plt.savefig(hist_path, dpi=120)
        plt.close(fig)
        logger.info("Confidence histogram saved: %s", hist_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fresh GoniOwl inference on all logged ECAM_6 images."
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained model.")
    parser.add_argument("--img-height", type=int, default=152, help="Image height for inference.")
    parser.add_argument("--img-width", type=int, default=218, help="Image width for inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    df = load_log_data()
    if df.empty:
        logger.info("No data found. Nothing to do.")
    else:
        results = run_inference_all(
            df,
            model_path=args.model_path,
            img_height=args.img_height,
            img_width=args.img_width,
        )
        save_csv(results)
        generate_confidence_histograms(results)
        logger.info("Done. Total processed: %d images.", len(results))
