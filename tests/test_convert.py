"""The PyTorch port must reproduce the original Keras models' predictions."""

import numpy as np
import pytest

from conftest import PACKAGE_DIR
from convert_keras_to_torch import convert
from GoniOwl.model import GoniOwlNet, load_checkpoint, predict_proba
from GoniOwl.preprocess import decide

# The 2025 model was trained with Keras' mixed_float16 policy, so Keras itself
# computed it in half precision; PyTorch runs it in float32 and lands within ~1e-3.
CASES = [
    ("20250714-111015_epoch20_binary_batch4.h5", True, 5e-3),
    ("20250714-111015_epoch20_binary_batch4.keras", True, 5e-3),
    ("20260626_140916_epoch30_binary_batch16.keras", False, 1e-5),
]


@pytest.mark.parametrize("name, use_bias, atol", CASES)
def test_converted_model_matches_keras(tmp_path, parity, name, use_bias, atol):
    out = tmp_path / "model.pt"
    convert(PACKAGE_DIR / name, out)
    model, info = load_checkpoint(out)
    assert model.hp.use_bias is use_bias
    assert (info["img_height"], info["img_width"]) == (152, 218)

    keras_scores = parity["keras"][name]
    torch_scores = predict_proba(model, parity["x"])
    np.testing.assert_allclose(torch_scores, keras_scores, atol=atol)
    assert [decide(s)[0] for s in torch_scores] == [decide(s)[0] for s in keras_scores]


@pytest.mark.parametrize("name", ["20250714-111015_epoch20_binary_batch4", "20260626_140916_epoch30_binary_batch16"])
def test_committed_checkpoints_are_current(tmp_path, parity, name):
    """The .pt files shipped in the package are conversions of the Keras files next to them."""
    source = next(PACKAGE_DIR.glob(f"{name}.[hk]*"))
    convert(source, tmp_path / "fresh.pt")
    fresh, _ = load_checkpoint(tmp_path / "fresh.pt")
    shipped, _ = load_checkpoint(PACKAGE_DIR / f"{name}.pt")
    np.testing.assert_array_equal(predict_proba(shipped, parity["x"]), predict_proba(fresh, parity["x"]))


def test_converted_models_classify_fixture_images(parity):
    for name in ["20250714-111015_epoch20_binary_batch4", "20260626_140916_epoch30_binary_batch16"]:
        model, _ = load_checkpoint(PACKAGE_DIR / f"{name}.pt")
        assert isinstance(model, GoniOwlNet)
        statuses = [decide(s)[0] for s in predict_proba(model, parity["x"])]
        assert statuses == [2, 2, 2, 1, 1, 1, 3]
