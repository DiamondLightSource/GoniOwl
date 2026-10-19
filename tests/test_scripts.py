"""Run the data-prep and metrics scripts as the README describes."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXTURES, PACKAGE_DIR, SRC, synthetic_image

MANUSCRIPT_MODEL = PACKAGE_DIR / "20260626_140916_epoch30_binary_batch16.pt"


def run_script(path: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path), *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=600
    )


def test_console_script_version():
    exe = Path(sys.executable).parent / "GoniOwl"
    out = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True)
    assert out.stdout.strip()


def test_import_crop_save(tmp_path):
    rng = np.random.default_rng(0)
    snapshots = tmp_path / "snapshots"
    for sub, pin in (("pin_ON/Pin_gripper_on_gonio", True), ("pin_OFF/Before_sample_load", False)):
        (snapshots / sub).mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(snapshots / sub / f"{i:02d}.jpg"), synthetic_image(rng, pin, (300, 400)))
    run_script(SRC / "Training" / "import_crop_save.py", "--snapshots", str(snapshots), "--imgdir", str(tmp_path / "out"), cwd=tmp_path)

    train = next((tmp_path / "out").glob("goniopin_auto_*_binary"))
    test = next((tmp_path / "out").glob("test_*_binary"))
    for cls in ("pinon", "pinoff"):
        assert len(list((train / cls).glob("*.jpg"))) == 9
        assert len(list((test / cls).glob("*.jpg"))) == 1
    assert cv2.imread(str(next((train / "pinon").glob("*.jpg")))).shape == (100, 200, 3)


def test_evaluate_test_set(tmp_path):
    out = run_script(
        SRC / "Metrics" / "evaluate_test_set.py",
        "--model-path", str(MANUSCRIPT_MODEL), "--test-dir", str(FIXTURES / "images"),
        "--output-csv", str(tmp_path / "results.csv"), "--device", "cpu",
        cwd=tmp_path,
    )
    summary = json.loads(out.stdout)
    assert summary["images"] == 7 and summary["confident"] == 6
    assert summary["accuracy_confident"] == 1.0
    assert summary["confusion_matrix_confident"] == [[3, 0], [0, 3]]
    rows = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert sum(r["predicted"] == "undetermined" for r in rows) == 1


@pytest.fixture
def beamline_logs(tmp_path, parity):
    """A GoniOwl CSV log as written on the beamline, pointing at the fixture images."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    images = tmp_path / "ecam6"
    images.mkdir()
    rows = []
    for i, (path, label) in enumerate(zip(parity["paths"], parity["y"])):
        img = images / f"2024-05-01-10-00-{i:02d}.jpg"
        shutil.copy(path, img)
        human = "human_yes" if label else "human_no"
        goniowl = "goniowl_pin_on" if label else "goniowl_pin_off"
        if i in (0, 3):  # two logged disagreements
            goniowl = "goniowl_pin_off" if label else "goniowl_pin_on"
        hist = "histogram_pin_detected" if label else "histogram_pin_not_detected"
        rows.append([f"2024-05-01-10-00-{i:02d}", hist, goniowl, human, str(img), str(img)])
    rows.append(["2024-05-01-10-01-00", "histogram_not_matched", "goniowl_dark", "human_no", str(img), str(img)])
    with open(log_dir / "2024-05-01.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return log_dir


def test_infer_all(tmp_path, beamline_logs):
    script = SRC / "Metrics" / "infer_all.py"
    run_script(script, "--model-path", str(MANUSCRIPT_MODEL), "--log-dir", str(beamline_logs), "--device", "cpu", cwd=tmp_path)
    out = tmp_path / "outputs" / "inferall"
    rows = list(csv.DictReader(open(out / "infer_all_results.csv")))
    assert len(rows) == 7  # the "dark" row is filtered out
    assert [r["goniowl_decision"] for r in rows] == ["on"] * 3 + ["off"] * 3 + ["off"]
    assert len(list(out.glob("*_2024-05-01-10-00-*.jpg"))) == 7
    for label in ("human_pin_on", "human_pin_off"):
        (out / f"confidence_histogram_{label}.png").unlink()

    run_script(script, "--histograms-only", "--run-path", str(out), cwd=tmp_path)
    assert (out / "confidence_histogram_human_pin_on.png").exists()


def test_disagreement_inference(tmp_path, beamline_logs):
    run_script(
        SRC / "Metrics" / "disagreement_inference.py",
        "--model-path", str(MANUSCRIPT_MODEL), "--log-dir", str(beamline_logs), "--device", "cpu",
        cwd=tmp_path,
    )
    out = tmp_path / "outputs" / "disagreements"
    rows = list(csv.DictReader(open(out / "disagreement_inference_results.csv")))
    assert len(rows) == 2
    assert [r["ModelPrediction"] for r in rows] == ["on", "off"]
    assert all((out / r["SavedAs"]).exists() for r in rows)


def test_inference_on_disagreements(tmp_path, parity):
    images = tmp_path / "disagreements"
    images.mkdir()
    for path in parity["paths"]:
        shutil.copy(path, images / path.name)
    run_script(
        SRC / "Metrics" / "inference_on_disagreements.py",
        "--model-path", str(MANUSCRIPT_MODEL), "--disagreement-dir", str(images),
        "--output-dir", str(tmp_path / "results"), "--device", "cpu",
        cwd=tmp_path,
    )
    rows = list(csv.DictReader(open(tmp_path / "results" / "inference_results.csv")))
    assert len(rows) == 7
    assert sorted(r["label"] for r in rows) == ["pin off"] * 3 + ["pin on"] * 3 + ["undetermined"]
    assert len(list((tmp_path / "results").glob("*_pred.png"))) == 7
