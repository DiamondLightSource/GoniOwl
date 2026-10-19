import logging
import urllib.request
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .model import load_checkpoint
from .preprocess import (
    CROP_BORDER,
    STATUS_UNSURE,
    crop_jpeg,
    decide,
    is_image_valid,
    load_image,
    mean_intensity,
)

PACKAGE_DIR = Path(__file__).resolve().parent

# The 2025 production model, converted from 20250714-111015_epoch20_binary_batch4.h5.
DEFAULT_MODEL = PACKAGE_DIR / "20250714-111015_epoch20_binary_batch4.pt"
DEFAULT_CAMERA_URL = "http://bl23i-di-serv-04.diamond.ac.uk:8080/ECAM6.mjpg.jpg"
DEFAULT_LOG_PATH = "/dls/science/groups/i23/scripts/chris/GoniOwl/src/GoniOwl/logs/GoniOwl_binary_controller.log"


def _file_logger(log_path: Optional[Union[str, Path]]) -> logging.Logger:
    log = logging.getLogger("GoniOwl.controller")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
    return log


class GoniOwlController:
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL,
        camera_url: str = DEFAULT_CAMERA_URL,
        log_path: Optional[Union[str, Path]] = DEFAULT_LOG_PATH,
        device: str = "cpu",
        last_image_path: Optional[Union[str, Path]] = "tmp.jpg",
        timeout: float = 5.0,
    ):
        self.camera_url = camera_url
        self.last_image_path = last_image_path
        self.timeout = timeout
        self.log = _file_logger(log_path)
        self.log.info("GoniOwl binary controller started.")

        self.model_name = Path(model_path).name
        self.model, info = load_checkpoint(model_path, device)
        self.img_size = (info["img_height"], info["img_width"])
        self.classes = info["class_names"]
        self.log.info(f"Model {self.model_name} loaded successfully.")
        print(self.model)

        self.predicted_label = None
        self.score = 0.0

    def fetch_frame(self) -> bytes:
        """Grab the latest frame from the camera and crop it like the training images."""
        with urllib.request.urlopen(self.camera_url, timeout=self.timeout) as resp:
            cropped = crop_jpeg(resp.read(), CROP_BORDER)
        if self.last_image_path:
            Path(self.last_image_path).write_bytes(cropped)
        return cropped

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> float:
        """P(pin on) for one RGB uint8 image at the model's input size."""
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        x = x.to(next(self.model.parameters()).device)
        return float(torch.sigmoid(self.model(x)).item())

    def infer(self) -> int:
        """Run one inference on the live camera and return the GONIOSTATUS value."""
        try:
            image = load_image(self.fetch_frame(), self.img_size)
        except Exception as e:
            print(f"Could not read camera image: {e}")
            self.log.error(f"Could not read camera image from {self.camera_url}: {e}")
            self.predicted_label = "error"
            self.score = 0.0
            return STATUS_UNSURE
        return self.infer_image(image)

    def infer_image(self, image: np.ndarray) -> int:
        if not is_image_valid(image):
            print(f"Image rejected: mean intensity {mean_intensity(image):.2f}")
            self.predicted_label = "invalid"
            self.score = 0.0
            self.printscore()
            return STATUS_UNSURE

        score = self.predict(image)
        status, self.predicted_label = decide(score)
        print(f"{score} is {self.predicted_label}")
        # Report confidence in the predicted class.
        self.score = 1 - score if self.predicted_label == self.classes[0] else score
        self.printscore()
        return status

    def printscore(self):
        rounded_score = np.round((self.score * 100), 3)
        msg = f"Status is {self.predicted_label} with {str(rounded_score)} % conf."
        print(msg)
        self.log.info(msg)
