import numpy as np
import pytest
import torch
from torch import nn

from GoniOwl.model import (
    GoniOwlNet,
    Hyperparameters,
    count_parameters,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
)

LEGACY_HP = Hyperparameters(20, 44, 24, 48, 176, 64, use_bias=True)


def test_parameter_count_matches_keras():
    # Keras model.summary() totals, including BatchNorm moving statistics.
    assert count_parameters(GoniOwlNet()) == 51313
    assert count_parameters(GoniOwlNet(LEGACY_HP)) == 49989


def test_forward_returns_one_logit_per_image():
    model = GoniOwlNet().eval()
    out = model(torch.rand(3, 3, 152, 218) * 255)
    assert out.shape == (3,)
    probs = predict_proba(model, np.random.randint(0, 255, (3, 152, 218, 3), dtype=np.uint8))
    assert probs.shape == (3,) and np.all((probs >= 0) & (probs <= 1))


def test_batchnorm_uses_keras_defaults():
    for m in GoniOwlNet().modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            assert m.eps == pytest.approx(1e-3)
            assert m.momentum == pytest.approx(0.01)


def test_use_bias_flag():
    biases = [m.bias for m in GoniOwlNet().modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert all(b is None for b in biases[:-1]) and biases[-1] is not None
    biases = [m.bias for m in GoniOwlNet(LEGACY_HP).modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert all(b is not None for b in biases)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = GoniOwlNet(Hyperparameters(conv1=24)).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, (152, 218), ["pinoff", "pinon"], epoch=3)
    loaded, info = load_checkpoint(path)
    assert info["img_height"] == 152 and info["img_width"] == 218
    assert info["class_names"] == ["pinoff", "pinon"]
    assert info["metadata"]["epoch"] == 3
    assert loaded.hp == model.hp
    x = np.random.randint(0, 255, (2, 152, 218, 3), dtype=np.uint8)
    np.testing.assert_allclose(predict_proba(loaded, x), predict_proba(model, x), atol=1e-6)


def test_checkpoint_roundtrip_unwraps_dataparallel(tmp_path):
    model = nn.DataParallel(GoniOwlNet())
    save_checkpoint(tmp_path / "m.pt", model, (152, 218))
    loaded, _ = load_checkpoint(tmp_path / "m.pt")
    assert isinstance(loaded, GoniOwlNet)


def test_load_rejects_other_files(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "other.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "other.pt")
