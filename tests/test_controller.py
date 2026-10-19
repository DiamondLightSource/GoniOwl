import cv2
import numpy as np
import pytest

from conftest import raw_frame
from GoniOwl.GoniOwl_controller import DEFAULT_MODEL, GoniOwlController
from GoniOwl.preprocess import STATUS_PIN_OFF, STATUS_PIN_ON, STATUS_UNSURE


@pytest.fixture
def controller(camera, tmp_path):
    return GoniOwlController(
        model_path=DEFAULT_MODEL,
        camera_url=camera.url,
        log_path=tmp_path / "logs" / "controller.log",
        last_image_path=tmp_path / "tmp.jpg",
    )


def test_default_model_is_shipped():
    assert DEFAULT_MODEL.exists()


def test_infer_live_frames(controller, camera, parity, tmp_path):
    statuses = []
    for path in parity["paths"]:
        camera.frame = raw_frame(path)
        statuses.append(controller.infer())
    expected = [STATUS_PIN_ON] * 3 + [STATUS_PIN_OFF] * 3 + [STATUS_UNSURE]
    assert statuses == expected

    saved = cv2.imread(str(tmp_path / "tmp.jpg"))
    assert saved.shape == (764, 1092, 3)  # 100px border cropped from a 964x1292 frame
    log = (tmp_path / "logs" / "controller.log").read_text()
    assert "Model 20250714-111015_epoch20_binary_batch4.pt loaded successfully." in log
    assert "Status is pinon with" in log and "Status is pinoff with" in log


@pytest.mark.parametrize("value", [10, 250])
def test_too_dark_or_bright_is_unsure(controller, camera, value):
    ok, buf = cv2.imencode(".jpg", np.full((964, 1292, 3), value, dtype=np.uint8))
    camera.frame = buf.tobytes()
    assert controller.infer() == STATUS_UNSURE
    assert controller.predicted_label == "invalid"


def test_camera_failure_is_unsure(controller, camera):
    camera.frame = b"not a jpeg"
    assert controller.infer() == STATUS_UNSURE
    controller.camera_url = "http://127.0.0.1:1/nothing.jpg"
    assert controller.infer() == STATUS_UNSURE
    assert controller.predicted_label == "error"
