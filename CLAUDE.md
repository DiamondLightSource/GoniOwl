# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GoniOwl is a small binary CNN (~51k params) that tells whether a sample pin is on the goniometer at beamline I23, Diamond Light Source. It stops equipment collisions during automated sample handling. The repo has three independent parts that share no imports:

- `src/Training/`: dataset prep and model training (standalone scripts, usually run on the Diamond SLURM cluster).
- `src/GoniOwl/`: the deployed inference service, a Python EPICS soft IOC (`softioc` + `cothread`) that GDA calls.
- `src/Metrics/`: offline evaluation against the beamline's historical decision logs.

## Commands

```bash
# Environment (pyproject.toml pins the training/TF stack)
uv venv && source .venv/bin/activate && uv pip install -e .

# 1. Crop raw snapshots (expects <snapshots>/pin_ON/Pin_gripper_on_gonio and <snapshots>/pin_OFF/Before_sample_load)
python src/Training/import_crop_save.py --snapshots /path/to/snapshots --imgdir /path/to/out
#    -> <imgdir>/goniopin_auto_<ddmmyyyy>_binary/{pinon,pinoff} plus a 10% held-out <imgdir>/test_<ddmmyyyy>_binary

# 2. Train (outputs go to --tmpdir, default ./outputs, which is gitignored)
python src/Training/train_binary_classification_model.py --train-dir /path/to/out/goniopin_auto_..._binary
#    add --tune [--tuner hyperband|bayesianoptimization|randomsearch|gridsearch] for KerasTuner, --parallel for MirroredStrategy

# 3. Run the IOC (must be run from src/GoniOwl/, because the model is loaded from a relative path)
cd src/GoniOwl && python -m GoniOwl        # README uses `pipenv run python -m GoniOwl`; run inside screen/tmux

# Smoke-test the running IOC
caget GONIOWL-TEST:WHOAMI
caput GONIOWL-TEST:INFER 1 && caget GONIOWL-TEST:GONIOSTATUS

# Offline evaluation (reads /dls_sw/i23/logs/GoniOwl/*.csv, writes under ./outputs/)
python src/Metrics/infer_all.py --model-path model.keras --img-height 152 --img-width 218
python src/Metrics/inference_on_disagreements.py --model-path model.keras --disagreement-dir DIR
```

The repo has no tests. The pytest, tox, flake8 and black config in `setup.cfg` is left over from the DLS project template, and those tools are not installed by `pyproject.toml`. Do not run `pytest` as configured: `--doctest-modules` over `src` would import `GoniOwl/__main__.py`, which starts the IOC and loads TensorFlow as a side effect.

## Architecture and cross-file invariants

**Preprocessing must match between training and inference.** Raw camera frames get a 100 px crop on every side (`import_crop_save.croppit` and `GoniOwlController.urltoimage`), then both height and width are divided by `image_divider = 5`. Training defaults are 764×1092, which gives 152×218 inputs. The model has its own `Rescaling(1/255)` layer, so feed it raw 0–255 pixels. The Metrics scripts take `--img-height/--img-width` directly and do **not** apply the 100 px crop.

**Label semantics.** `image_dataset_from_directory` assigns classes alphabetically, so `pinoff = 0` and `pinon = 1`, and the sigmoid output is P(pin on). `GoniOwlController.infer()` maps the score to `GONIOSTATUS` like this:
- `3`: image rejected (mean grey intensity outside [80, 180]) or score in [0.15, 0.85] ("not sure, stop")
- `1`: score < 0.15 → `pinoff`
- `2`: score > 0.85 → `pinon`

The README's "Beamline Implementation" section describes the opposite mapping (score 0 = pin on, status 1 = pin on). The code and the Metrics scripts (`> 0.85` = "on") agree with each other but not with the README. Confirm with the user before changing either.

**IOC flow** (`src/GoniOwl/__main__.py`): everything runs at module import time, and there is no `main()` function, even though the `setup.cfg` console_script points at `GoniOwl.__main__:main`. It builds PVs under the `GONIOWL-TEST` prefix and spawns a cothread loop that polls `INFER` once per second. When `INFER` is 1, it fetches the latest MJPEG frame from the hard-coded ECAM6 URL, runs `infer()` and writes `GONIOSTATUS`. The model filename, camera URL and log path (`/dls/science/...`) are all hard-coded in `GoniOwl_controller.py`. The deployed model is still the legacy `.h5` file; the `.keras` files next to it are newer training outputs.

**Training script** (`train_binary_classification_model.py`). Several choices here were tuned on purpose; don't revert them:
- `_ensure_cuda_libpath()` must stay above every TF/cv2 import. It prepends the pip `nvidia/*/lib` dirs to `LD_LIBRARY_PATH` and re-execs the process, because cv2 otherwise hides the CUDA libs and TF falls back to CPU.
- Dataset pipeline order is `cache() → shuffle(reshuffle_each_iteration) → map(augmentations)`. Caching after augmentation froze the augmentations from epoch to epoch.
- `ReduceLROnPlateau` and `EarlyStopping` monitor `val_auc` (mode max), not `val_loss`. On this small dataset val_loss is noisy enough to collapse training to the majority class.
- Keep augmentation gentle (±3° rotation, 5% translation, small brightness/contrast changes, light noise). Stronger photometric augmentation stops learning.
- Batch size ≥ 8 (default 16, seed 0, which matches the manuscript model). Batch 2–4 often collapses to ~0.53.
- `predefined_model()` (the manuscript architecture) and `model_builder()` (the KerasTuner search space) duplicate the same layer stack. Keep them in sync.
- After saving, `_strip_unsupported_keys` removes `quantization_config` from the `.keras` zip so older Keras 3 versions can load it. `src/Training/convert_to_savedmodel.py` does the same thing standalone (despite its name, it does not produce a SavedModel). `src/Metrics/infer_all.load_model` separately patches legacy `.h5` dtype configs for Keras 3.

**Metrics scripts** all parse the same beamline log CSVs: headerless, with columns `DateTime, Histogram, GoniOwl, Human, Image1, Image2`. They normalise labels to `on`/`off` using the same replace maps and exclude `DODGY_DATA` timestamps. That loading logic is copy-pasted across `GoniOwlMetrics.py`, `infer_all.py` and `disagreement_inference.py`.

## Packaging quirks

- Dependencies are split. `pyproject.toml` pins the TF/CUDA training stack but leaves out `softioc`, `cothread` and `epicscorelibs`. Those IOC deps appear only in `setup.cfg` `install_requires`.
- Versioning is configured twice: `setuptools_scm` in `pyproject.toml` writes `_version.py`, while `setup.py` and `__init__.py` use `_version_git.py`. `Pipfile` refers to `pymalcolm`, another template leftover.
- The Singularity/SLURM files in `src/Training/` refer to an older TF 2.8.2 image and script names (`img_classification*.py`) that no longer exist.
- The README gives `src/Training/inference_on_disagreements.py` as the path, but the file actually lives in `src/Metrics/`.
