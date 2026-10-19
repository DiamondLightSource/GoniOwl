import cv2
import numpy as np
import pytest
from PIL import Image

from GoniOwl.preprocess import (
    STATUS_PIN_OFF,
    STATUS_PIN_ON,
    STATUS_UNSURE,
    crop_border,
    crop_jpeg,
    decide,
    is_image_valid,
    load_image,
    mean_intensity,
    resize,
)


@pytest.mark.parametrize(
    "score, status, label",
    [
        (0.0, STATUS_PIN_OFF, "pinoff"),
        (0.149, STATUS_PIN_OFF, "pinoff"),
        (0.15, STATUS_UNSURE, "unknown"),
        (0.5, STATUS_UNSURE, "unknown"),
        (0.85, STATUS_UNSURE, "unknown"),
        (0.851, STATUS_PIN_ON, "pinon"),
        (1.0, STATUS_PIN_ON, "pinon"),
    ],
)
def test_decide_matches_controller_thresholds(score, status, label):
    assert decide(score) == (status, label)


def test_crop_border():
    img = np.zeros((964, 1292, 3), dtype=np.uint8)
    assert crop_border(img).shape == (764, 1092, 3)
    assert crop_border(img, 0) is img
    with pytest.raises(ValueError):
        crop_border(np.zeros((150, 300, 3), dtype=np.uint8))


def test_crop_jpeg_matches_import_crop_save():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (964, 1292, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    cropped = cv2.imdecode(np.frombuffer(crop_jpeg(buf.tobytes()), np.uint8), cv2.IMREAD_COLOR)
    # import_crop_save.croppit: cv2.imread -> [100:-100, 100:-100] -> cv2.imwrite
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    ok, expected = cv2.imencode(".jpg", decoded[100:-100, 100:-100])
    np.testing.assert_array_equal(cropped, cv2.imdecode(expected, cv2.IMREAD_COLOR))


def test_load_image_resizes_and_crops(tmp_path):
    arr = np.full((964, 1292, 3), 50, dtype=np.uint8)
    arr[100:-100, 100:-100] = 200
    path = tmp_path / "frame.png"
    Image.fromarray(arr).save(path)
    out = load_image(path, (152, 218), border=100)
    assert out.shape == (152, 218, 3) and out.dtype == np.float32
    assert np.all(out == 200)
    assert load_image(path.read_bytes(), (152, 218)).shape == (152, 218, 3)


def test_load_image_converts_greyscale_to_rgb(tmp_path):
    Image.fromarray(np.full((40, 60), 99, dtype=np.uint8)).save(tmp_path / "g.png")
    out = load_image(tmp_path / "g.png", (20, 30))
    assert out.shape == (20, 30, 3) and np.all(out == 99)


def test_resize_matches_tf_image_resize():
    # Values from tf.image.resize(x, (2, 2)) (bilinear, half-pixel centres, no antialias).
    x = np.arange(16, dtype=np.float32).reshape(4, 4, 1).repeat(3, axis=2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_allclose(resize(x, (2, 2))[..., 0], expected)

    # Non-integer scale factor.
    x = (np.random.default_rng(0).random((7, 9, 1)).astype(np.float32) * 255).repeat(3, axis=2)
    expected = np.array(
        [
            [180.9309, 129.2617, 186.6006, 182.3135],
            [203.0506, 174.3431, 53.1408, 113.3952],
            [99.5878, 89.7514, 130.335, 110.299],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(resize(x, (3, 4))[..., 0], expected, atol=1e-3)


def test_intensity_gate():
    assert mean_intensity(np.full((4, 4, 3), 100, dtype=np.uint8)) == pytest.approx(100)
    assert not is_image_valid(np.full((4, 4, 3), 79, dtype=np.uint8))
    assert is_image_valid(np.full((4, 4, 3), 80, dtype=np.uint8))
    assert is_image_valid(np.full((4, 4, 3), 180, dtype=np.uint8))
    assert not is_image_valid(np.full((4, 4, 3), 181, dtype=np.uint8))
