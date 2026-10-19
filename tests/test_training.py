import csv
import subprocess
import sys

import numpy as np
import pytest
import torch

import train_binary_classification_model as tbcm
from conftest import SRC, write_class_folders
from GoniOwl.model import Hyperparameters, load_checkpoint


def smooth_images(n=8, h=60, w=80):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = (60 + 120 * xx + 40 * yy).expand(n, 3, h, w)
    return base.clone()


def test_augmentations_are_gentle_and_per_image():
    torch.manual_seed(0)
    x = smooth_images()
    out = tbcm.augmentations(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 255
    diff = (out - x).abs().mean(dim=(1, 2, 3))
    assert torch.all(diff > 0.5) and torch.all(diff < 40)
    # Each image gets its own brightness offset.
    assert len(set(np.round((out - x).mean(dim=(1, 2, 3)).numpy(), 1))) > 1


def test_plateau_lr_matches_keras_semantics():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    plateau = tbcm.PlateauLR(opt, patience=2)
    lrs = []
    for epoch, value in enumerate([0.5, 0.6, 0.6, 0.6, 0.6, 0.6]):
        plateau.step(value, epoch)
        lrs.append(opt.param_groups[0]["lr"])
    # Keras cuts after `patience` non-improving epochs, then waits another `patience`.
    assert lrs == pytest.approx([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5])


def test_early_stopping_restores_best_weights():
    model = torch.nn.Linear(1, 1)
    stopper = tbcm.EarlyStopping(patience=2, restore_best_weights=True)
    stops = []
    for epoch, value in enumerate([0.90, 0.95, 0.94, 0.93]):
        with torch.no_grad():
            model.weight.fill_(value)
        stops.append(stopper.step(value, epoch, model))
    assert stops == [False, False, False, True]
    assert model.weight.item() == pytest.approx(0.95)


def test_dataset_split_and_labels(tmp_path):
    data = write_class_folders(tmp_path / "data", 10)
    train, val, class_names = tbcm.build_datasets(str(data), 60, 80, seed=0, tmpdir=str(tmp_path), val_split=0.3)
    assert class_names == ["pinoff", "pinon"]
    assert len(train) == 14 and len(val) == 6
    assert train.images.shape[1:] == (60, 80, 3)
    # Pin-on images have the dark bar, so they are darker on average.
    on = train.images[train.labels == 1].mean()
    off = train.images[train.labels == 0].mean()
    assert on < off

    explicit_val = write_class_folders(tmp_path / "val", 3, seed=1)
    train, val, _ = tbcm.build_datasets(str(data), 60, 80, seed=0, tmpdir=str(tmp_path), val_dir=str(explicit_val))
    assert len(train) == 20 and len(val) == 6


@pytest.mark.slow
def test_training_cli_end_to_end(tmp_path):
    data = write_class_folders(tmp_path / "data", 24)
    out = tmp_path / "out"
    cmd = [
        sys.executable, str(SRC / "Training" / "train_binary_classification_model.py"),
        "--train-dir", str(data), "--tmpdir", str(out),
        "--img-height", "60", "--img-width", "80", "--image-divider", "1",
        "--epochs", "3", "--batch-size", "8", "--device", "cpu",
        "--early-stopping", "--log-file", str(out / "run.log"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    finals = list(out.glob("*_binary_batch8_div1.pt"))
    assert len(finals) == 1
    model, info = load_checkpoint(finals[0])
    assert (info["img_height"], info["img_width"]) == (60, 80)
    assert model.hp == Hyperparameters()

    assert len(list((out / "checkpoints").glob("*_epoch0[1-3]_binary_batch8.pt"))) == 3
    rows = list(csv.DictReader(open(next((out / "logs" / "fit").glob("*/training_*.csv")))))
    assert len(rows) == 3 and {"loss", "val_auc", "learning_rate"} <= set(rows[0])
    assert list(out.glob("training_curves_*.png")) and list(out.glob("confusionmatrix_*.png"))
    assert list((out / "logs" / "fit").glob("*/train/events.out.tfevents.*"))
    assert len(list((out / "sample_images").glob("*.png"))) == 20
    assert "Validation confusion matrix" in (out / "run.log").read_text()


@pytest.mark.slow
def test_learns_separable_data(tmp_path):
    data = write_class_folders(tmp_path / "data", 48)
    args = tbcm.parse_args([
        "--train-dir", str(data), "--tmpdir", str(tmp_path / "out"),
        "--img-height", "60", "--img-width", "80", "--image-divider", "1",
        "--epochs", "8", "--batch-size", "8", "--device", "cpu",
    ])
    path = tbcm.run(args)
    model, _ = load_checkpoint(path)
    train, val, _ = tbcm.build_datasets(str(data), 60, 80, seed=0, tmpdir=str(tmp_path), val_split=0.3)
    metrics, _ = tbcm.evaluate(model, val, 16, torch.device("cpu"))
    # Only the ranking is checked: on a toy set this small, the dropout->BatchNorm
    # variance shift (same layer order as the Keras model) offsets eval-mode logits,
    # so thresholded accuracy is not meaningful here. Real-data accuracy is ~0.99.
    assert metrics["auc"] > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("tuner", tbcm.TUNERS)
def test_tuning(tmp_path, tuner):
    data = write_class_folders(tmp_path / "data", 12)
    out = tmp_path / "out"
    args = tbcm.parse_args([
        "--train-dir", str(data), "--tmpdir", str(out),
        "--img-height", "60", "--img-width", "80", "--image-divider", "1",
        "--epochs", "1", "--batch-size", "8", "--device", "cpu",
        "--tune", "--tuner", tuner, "--tune-trials", "2",
        "--tune-max-epochs", "2", "--tune-search-epochs", "1",
    ])
    path = tbcm.run(args)
    assert path.endswith("_tuned.pt")
    trials = list(csv.DictReader(open(next(out.glob("tune_b8_*/trials.csv")))))
    assert len(trials) == 2
    model, _ = load_checkpoint(path)
    best = max(trials, key=lambda t: float(t["value"] or 0))
    assert model.hp.conv1 == int(best["params_conv1"])
