import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKAGE_DIR = SRC / "GoniOwl"

sys.path.insert(0, str(SRC / "Training"))


@pytest.fixture(scope="session")
def parity():
    """Cropped held-out test images (as written by import_crop_save.py) and the
    original Keras models' scores for them.

    The last image is a pin-on frame that the 2025 model is unsure about.
    """
    from GoniOwl.preprocess import load_image

    ref = json.loads((FIXTURES / "keras_scores.json").read_text())
    paths = [FIXTURES / "images" / name for name in ref["images"]]
    return {
        "paths": paths,
        "x": np.stack([load_image(p, (152, 218)) for p in paths]),
        "y": np.array([1 if name.startswith("pinon/") else 0 for name in ref["images"]]),
        "keras": {name: np.array(scores, dtype=np.float32) for name, scores in ref["scores"].items()},
    }


def synthetic_image(rng: np.random.Generator, pin: bool, size=(60, 80)) -> np.ndarray:
    """Grey noisy frame; 'pin on' images have a dark vertical bar in the middle."""
    h, w = size
    img = rng.normal(130, 8, size=(h, w, 3)).clip(0, 255)
    if pin:
        img[h // 4 : 3 * h // 4, w // 2 - 4 : w // 2 + 4] = 30
    return img.astype(np.uint8)


def write_class_folders(root: Path, n_per_class: int, size=(60, 80), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    for name, pin in (("pinoff", False), ("pinon", True)):
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            Image.fromarray(synthetic_image(rng, pin, size)).save(root / name / f"{name}_{i:03d}.jpg", quality=95)
    return root
