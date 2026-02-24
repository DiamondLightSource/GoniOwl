#!/usr/bin/env python3
"""
Make a .keras model compatible with older Keras 3 versions by stripping
unsupported keys like 'quantization_config' from the saved config.

A .keras file is a zip archive containing config.json and weight files.
This script patches config.json in-place so the model loads on older Keras 3.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path


KEYS_TO_STRIP = {"quantization_config"}


def strip_keys(obj):
    """Recursively remove unsupported keys from a config dict."""
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in KEYS_TO_STRIP}
    if isinstance(obj, list):
        return [strip_keys(item) for item in obj]
    return obj


def patch_keras_file(input_path: Path, output_path: Path) -> int:
    """Read a .keras zip, patch config.json, write to output."""
    if not zipfile.is_zipfile(input_path):
        print(f"Not a valid .keras (zip) file: {input_path}", file=sys.stderr)
        return 1

    buf = io.BytesIO()
    with zipfile.ZipFile(input_path, "r") as zin, zipfile.ZipFile(buf, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                config = strip_keys(config)
                data = json.dumps(config, indent=2).encode("utf-8")
                print(f"Patched config.json (stripped {KEYS_TO_STRIP})")
            zout.writestr(item, data)

    output_path.write_bytes(buf.getvalue())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Make a .keras model compatible with older Keras 3 by removing unsupported config keys."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to input .keras model file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .keras file path (default: <input>_compatible.keras).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    if model_path.suffix.lower() != ".keras":
        print("Input must be a .keras file.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else model_path.with_stem(model_path.stem + "_compatible")

    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")

    rc = patch_keras_file(model_path, output_path)
    if rc == 0:
        print(f"Done. Compatible model saved to {output_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
