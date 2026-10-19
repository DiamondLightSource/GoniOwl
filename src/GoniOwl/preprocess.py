"""Image preprocessing and decision logic shared by training, evaluation and metrics.

Every consumer of the model must prepare images the same way:

1. Crop ``CROP_BORDER`` pixels from every side of the raw camera frame
   (``import_crop_save.py`` does this once for the training set; ``crop_jpeg``
   does the same for a single live frame).
2. Resize to the model's input size with non-antialiased bilinear filtering,
   matching tf.image.resize (``load_image``).
3. Feed raw 0-255 RGB pixels; the model rescales to 0-1 itself.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CROP_BORDER = 100
IMAGE_DIVIDER = 5
CLASS_NAMES = ("pinoff", "pinon")  # alphabetical, so pinoff=0 and pinon=1

# Mean grey intensity outside this range means the frame is too dark/bright to trust.
LOW_INTENSITY = 80
HIGH_INTENSITY = 180

# A decision needs at least this much confidence, otherwise the IOC reports "unsure".
CONFIDENCE_THRESHOLD = 0.85

# GONIOSTATUS values read by GDA.
STATUS_PIN_OFF = 1
STATUS_PIN_ON = 2
STATUS_UNSURE = 3

ImageSource = Union[str, Path, bytes, BytesIO, Image.Image]


def crop_border(img: np.ndarray, border: int = CROP_BORDER) -> np.ndarray:
    """Crop ``border`` pixels from each side of an HxW(xC) array."""
    if border <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(f"Image too small to crop {border}px borders (h={h}, w={w})")
    return img[border:-border, border:-border]


def crop_jpeg(jpeg_bytes: bytes, border: int = CROP_BORDER) -> bytes:
    """Decode a camera JPEG, crop its border and re-encode it.

    Uses OpenCV with its default JPEG quality so live frames go through exactly
    the same crop/re-encode as the training images written by import_crop_save.py.
    """
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode camera image")
    ok, buffer = cv2.imencode(".jpg", crop_border(img, border))
    if not ok:
        raise ValueError("Could not re-encode cropped camera image")
    return buffer.tobytes()


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize without antialiasing, identical to tf.image.resize.

    The Keras models were trained on tf.image.resize output; antialiased
    resizing (e.g. PIL's BILINEAR) smooths the image enough to flip the 2025
    model's decisions on some pin-on frames.
    """
    if image.shape[:2] == tuple(size):
        return image.astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False, antialias=False)
    return x.squeeze(0).permute(1, 2, 0).numpy()


def load_image(source: ImageSource, size: tuple[int, int], border: int = 0) -> np.ndarray:
    """Load an image as an RGB float32 array (0-255) of shape (height, width, 3).

    ``size`` is (height, width). ``border`` crops that many pixels from each side
    before resizing; leave it at 0 for images that are already cropped.
    """
    if isinstance(source, bytes):
        source = BytesIO(source)
    img = source if isinstance(source, Image.Image) else Image.open(source)
    arr = crop_border(np.asarray(img.convert("RGB")), border)
    return resize(arr, size)


def mean_intensity(rgb: np.ndarray) -> float:
    """Mean ITU-R 601 luma of an RGB image, with per-pixel rounding like cv2/PIL greyscale."""
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int64)
    grey = (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114 + 500) // 1000
    return float(np.mean(grey))


def is_image_valid(rgb: np.ndarray, low: float = LOW_INTENSITY, high: float = HIGH_INTENSITY) -> bool:
    return low <= mean_intensity(rgb) <= high


def decide(score: float, threshold: float = CONFIDENCE_THRESHOLD) -> tuple[int, str]:
    """Map P(pin on) to a (GONIOSTATUS, label) pair."""
    if score < round(1.0 - threshold, 12):
        return STATUS_PIN_OFF, CLASS_NAMES[0]
    if score > threshold:
        return STATUS_PIN_ON, CLASS_NAMES[1]
    return STATUS_UNSURE, "unknown"
