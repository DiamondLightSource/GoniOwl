"""
Find all instances where GoniOwl disagrees with the human decision,
run inference on those images, and save results with confidence-prefixed filenames.

Example:
  python disagreement_inference.py \
    --model-path ./outputs/checkpoints/20260219_094846_epoch99_binary_batch16.keras \
    --img-height 152 --img-width 218
"""

import os
import csv
import glob
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Model
from typing import cast


LOG_DIR = "/dls_sw/i23/logs/GoniOwl"
START_DATE_CUTOFF = "2023-01-01"
DODGY_DATA = ["2024-08-01-09-02-27"]
OUTPUT_DIR = "outputs/disagreements"


def setup_logging() -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    return logging.getLogger(__name__)


logger = setup_logging()


def load_disagreements() -> pd.DataFrame:
    """Load CSVs from the log directory and return rows where GoniOwl != Human."""
    csvs = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    csvs = [f for f in csvs if os.path.basename(f)[:10] >= START_DATE_CUTOFF]

    if not csvs:
        logger.warning("No CSV files found in %s", LOG_DIR)
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(f, header=None) for f in csvs), ignore_index=True)
    df.columns = ["DateTime", "Histogram", "GoniOwl", "Human", "Image1", "Image2"]

    df["GoniOwl"] = df["GoniOwl"].replace({
        "goniowl_pin_off": "off", "goniowl_pin_on": "on",
        "ERROR_READING_PV": "error", "goniowl_dark": "dark",
        "goniowl_light": "light", "PIN_ON": "on", "PIN_OFF": "off",
    })
    df["Human"] = df["Human"].replace({"human_no": "off", "human_yes": "on"})

    df = df[~df["DateTime"].isin(DODGY_DATA)]
    df = df[~df["GoniOwl"].isin(["error", "light", "dark"])]

    disagreements = df[df["GoniOwl"] != df["Human"]].copy()
    logger.info("Found %d disagreements out of %d total records.", len(disagreements), len(df))
    return disagreements


def run_inference_on_disagreements(
    disagreements: pd.DataFrame,
    model_path: str,
    img_height: int,
    img_width: int,
) -> list[dict]:
    """Run model inference on each disagreement image and return results."""
    logger.info("Loading model from: %s", model_path)
    model = cast(Model, keras.models.load_model(model_path))
    logger.info("Model loaded successfully.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    total = len(disagreements)

    for idx, (_, row) in enumerate(disagreements.iterrows(), 1):
        img_path = row["Image1"]
        goniowl_value = row["GoniOwl"]
        human_value = row["Human"]
        date_time = row["DateTime"]

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
                model_label = "on"
            elif prediction < 0.15:
                model_label = "off"
            else:
                model_label = "undetermined"

            confidence = prediction if prediction > 0.5 else 1 - prediction
            confidence_pct = int(round(confidence * 100))

            # Save image with confidence prefix
            original_name = os.path.basename(img_path)
            new_name = f"{confidence_pct}_{original_name}"
            dest_path = os.path.join(OUTPUT_DIR, new_name)
            shutil.copy(img_path, dest_path)

            results.append({
                "DateTime": date_time,
                "OriginalImage": img_path,
                "SavedAs": new_name,
                "GoniOwl": goniowl_value,
                "Human": human_value,
                "ModelPrediction": model_label,
                "RawScore": round(prediction, 4),
                "Confidence": f"{confidence_pct}%",
            })

            logger.info(
                "  GoniOwl=%s, Human=%s, Model=%s (confidence=%d%%)",
                goniowl_value, human_value, model_label, confidence_pct,
            )

        except Exception as e:
            logger.error("Error processing %s: %s", img_path, e)

    return results


def save_results(results: list[dict]) -> None:
    """Save the results table to CSV."""
    if not results:
        logger.warning("No results to save.")
        return

    csv_path = os.path.join(OUTPUT_DIR, "disagreement_inference_results.csv")
    fieldnames = [
        "DateTime", "OriginalImage", "SavedAs",
        "GoniOwl", "Human", "ModelPrediction", "RawScore", "Confidence",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results table saved to: %s", csv_path)
    logger.info("Images saved to: %s", OUTPUT_DIR)
    logger.info("Total processed: %d", len(results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on GoniOwl/Human disagreement images."
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained model.")
    parser.add_argument("--img-height", type=int, default=152, help="Image height for inference.")
    parser.add_argument("--img-width", type=int, default=218, help="Image width for inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    disagreements = load_disagreements()
    if disagreements.empty:
        logger.info("No disagreements found. Nothing to do.")
    else:
        results = run_inference_on_disagreements(
            disagreements,
            model_path=args.model_path,
            img_height=args.img_height,
            img_width=args.img_width,
        )
        save_results(results)
