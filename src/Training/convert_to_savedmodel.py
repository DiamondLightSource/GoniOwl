#!/usr/bin/env python3
"""Convert .keras or .h5 model to SavedModel format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert .keras or .h5 model to SavedModel format."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to input .keras or .h5 model file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: input path without extension).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    if model_path.suffix.lower() not in {".h5", ".keras"}:
        print(
            "Model file must have .h5 or .keras extension.",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully.")
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output or str(model_path.with_suffix(""))
    print(f"Saving model to {output_dir} (SavedModel format)...")

    try:
        # Use TensorFlow's SavedModel format directly (compatible with Keras 3)
        tf.saved_model.save(model, output_dir)
        print(f"Model converted and saved successfully to {output_dir}")
        return 0
    except Exception as exc:
        print(f"Failed to save model: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
