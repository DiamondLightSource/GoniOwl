"""
Deployment-style evaluation of a model on a held-out test set.

Applies the same confidence gating as the IOC: a prediction is confident iff
max(score, 1 - score) > 0.85, otherwise it is the fail-safe "undetermined"
state. Reports coverage, accuracy on the confident subset and the confusion
matrix, and writes per-image results to CSV.

Example:
  python evaluate_test_set.py \
    --model-path ./outputs/20261019_112139_binary_batch16_div5.pt \
    --test-dir ./outputs/data/test_19102026_binary
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from GoniOwl.model import load_checkpoint, predict_proba, select_device
from GoniOwl.preprocess import CONFIDENCE_THRESHOLD, load_image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def evaluate(model_path: str, test_dir: str, device: str = "auto", workers: int = 8) -> tuple[dict, list[dict]]:
    model, info = load_checkpoint(model_path, select_device(device))
    size = (info["img_height"], info["img_width"])
    class_names = info["class_names"]

    paths, labels = [], []
    for label, name in enumerate(class_names):
        files = sorted(p for p in (Path(test_dir) / name).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        paths += files
        labels += [label] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found under {test_dir}/{{{','.join(class_names)}}}")
    labels = np.array(labels)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        images = np.stack(list(ex.map(lambda p: load_image(p, size), paths)))
    scores = predict_proba(model, images)

    pred = (scores > 0.5).astype(int)
    confident = np.maximum(scores, 1 - scores) > CONFIDENCE_THRESHOLD
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(labels[confident], pred[confident]):
        cm[t, p] += 1
    summary = {
        "model": os.path.basename(model_path),
        "images": int(len(labels)),
        "confident": int(confident.sum()),
        "coverage": float(confident.mean()),
        "accuracy_confident": float((pred[confident] == labels[confident]).mean()) if confident.any() else float("nan"),
        "accuracy_ungated": float((pred == labels).mean()),
        "confusion_matrix_confident": cm.tolist(),
        "classes": class_names,
    }
    rows = [
        {
            "filename": str(p),
            "true": class_names[t],
            "score": round(float(s), 6),
            "predicted": class_names[pr] if c else "undetermined",
            "correct": bool(c and pr == t),
        }
        for p, t, s, pr, c in zip(paths, labels, scores, pred, confident)
    ]
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GoniOwl model on a held-out test set.")
    parser.add_argument("--model-path", required=True, help="Path to the trained .pt model.")
    parser.add_argument("--test-dir", required=True, help="Directory with pinoff/ and pinon/ subfolders.")
    parser.add_argument("--output-csv", default=None, help="Optional per-image results CSV.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps or cpu.")
    args = parser.parse_args()

    summary, rows = evaluate(args.model_path, args.test_dir, args.device)
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
