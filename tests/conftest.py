import http.server
import json
import socket
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKAGE_DIR = SRC / "GoniOwl"

sys.path.insert(0, str(SRC / "Training"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def raw_frame(cropped_jpeg: Path, border: int = 100) -> bytes:
    """Rebuild a full-size camera frame by adding back a border around a cropped image."""
    img = cv2.imread(str(cropped_jpeg))
    big = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".jpg", big)
    assert ok
    return buf.tobytes()


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


class CameraServer:
    """Serves whatever JPEG bytes the test last set, like the beamline MJPEG snapshot URL."""

    def __init__(self):
        self.frame = b""
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(server.frame)))
                self.end_headers()
                self.wfile.write(server.frame)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/ECAM6.mjpg.jpg"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def camera():
    server = CameraServer()
    yield server
    server.close()
