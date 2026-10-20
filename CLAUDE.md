# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GoniOwl is a small binary CNN (~51k params, PyTorch) that tells whether a sample pin is on the goniometer at beamline I23, Diamond Light Source. It stops equipment collisions during automated sample handling. The code is in four places:

- `src/GoniOwl/`: the installable package. It holds the shared PyTorch model (`model.py`) and the preprocessing and decision logic (`preprocess.py`). It also holds the deployed inference service, a Python EPICS soft IOC (`__main__.py`, `GoniOwl_controller.py`) that GDA calls. **The IOC is out of scope and has not been ported:** it still loads the Keras `.h5` model with TensorFlow on `cothread`, and its dependencies are not in `pyproject.toml`. Don't modify it unless asked.
- `src/Training/`: standalone scripts for dataset prep, training, and Keras→PyTorch conversion.
- `src/Metrics/`: offline evaluation, both against a held-out test set and against the beamline's historical decision logs.
- `tests/`: the pytest suite. `tests/fixtures/` holds 7 real cropped images and the original Keras models' scores for them.

The scripts in `src/Training/` and `src/Metrics/` import from the `GoniOwl` package, so install it first.

## Commands

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"   # or: uv sync --extra dev

pytest                         # full suite (~30 s), includes small training runs
pytest -m "not slow"           # skip training tests
pytest tests/test_convert.py::test_converted_model_matches_keras   # single test

# Data prep: expects <snapshots>/pin_ON/Pin_gripper_on_gonio and <snapshots>/pin_OFF/Before_sample_load
python src/Training/import_crop_save.py --snapshots training_images --imgdir outputs/data
python src/Training/train_binary_classification_model.py --train-dir outputs/data/goniopin_auto_<ddmmyyyy>_binary
python src/Metrics/evaluate_test_set.py --model-path outputs/<model>.pt --test-dir outputs/data/test_<ddmmyyyy>_binary
python src/Training/convert_keras_to_torch.py -m model.keras -o model.pt
```

The real training images live in `training_images/` and outputs in `outputs/`. Both are gitignored.

## Architecture and cross-file invariants

**Preprocessing must match everywhere** (all of it is in `GoniOwl/preprocess.py`):
1. Raw camera frames get a 100 px crop on every side. `import_crop_save.py` does this once for the training set; `crop_jpeg` does the same cv2 decode/crop/re-encode for a single frame.
2. `load_image` → `resize` is bilinear **without antialiasing**, equivalent to `tf.image.resize`, and returns float32. Do not switch to PIL `BILINEAR` or any other antialiased resize: it makes the 2025 model confidently call pin-on frames pin-off.
3. Images go into the model as raw 0–255 float NCHW. The model divides by 255 itself, and `forward` returns logits; use `predict_proba` to get P(pin on).

Training defaults are 764×1092 ÷ `--image-divider 5`, which gives 152×218. Every `.pt` checkpoint stores its input size and class names, and the evaluation and metrics scripts read them from it. The Metrics scripts default to `--crop-border 0` on logged images.

**Labels and status.** Classes come from alphabetical folder names: `pinoff = 0`, `pinon = 1`. `preprocess.decide` maps the score to `GONIOSTATUS`:
- `1`: score < 0.15 (pin off)
- `2`: score > 0.85 (pin on)
- `3`: unsure, or an image outside mean grey [80, 180]

This is the same mapping as the IOC code. The README's "Beamline Implementation" section documents 1 = pin on, which is the opposite; it's unresolved, so check with the user before changing either.

**Model** (`GoniOwl/model.py`): `GoniOwlNet(Hyperparameters)` is a port of the Keras Sequential model. It uses Keras BatchNorm defaults (eps 1e-3, momentum 0.01 in PyTorch terms) and Glorot init. `use_bias=True` exists only for the 2025 architecture. Checkpoints are dicts tagged `format: goniowl-torch-v1`; use `save_checkpoint` and `load_checkpoint`, not raw `torch.save`. `count_parameters` counts BatchNorm buffers so the totals match Keras' (51,313).

**Shipped models** in `src/GoniOwl/`: `20250714-…_batch4.pt` is a conversion of the deployed `.h5` (whose `.keras` twin is really HDF5 too). `20260626-…_batch16.pt` is the manuscript model. `test_committed_checkpoints_are_current` fails if a `.pt` drifts from its Keras source. The 2025 Keras model ran in `mixed_float16`, so its PyTorch port differs from Keras by about 1e-3. The manuscript model matches to about 1e-6.

**Training script**: deliberate choices that shouldn't be reverted:
- Images are cached in memory, then each epoch reshuffles and applies fresh per-image GPU augmentation (`augmentations`).
- `PlateauLR` and `EarlyStopping` re-implement Keras semantics on `val_auc` (mode max). `val_loss` is too noisy on this dataset.
- Augmentation stays gentle (±3°, ±5%, brightness ±25.5, contrast 0.9–1.1, noise σ=3). Stronger augmentation stops learning.
- Batch size ≥ 8 (default 16, seed 0). `--parallel` (DataParallel) splits the batch per GPU with separate BatchNorm stats.
- Adam uses eps=1e-7 (the Keras default).
- Tuning uses Optuna; `--tuner hyperband` means random sampling plus the HyperbandPruner. Models sit at chance for 3–7 epochs before learning suddenly. A learning rate of 2e-3 never learns, and wider nets need ≤2e-4. So the learning rates are capped at 1e-3 and trials can't be pruned before `--tune-min-epochs` (8). For the same reason, `--plateau-warmup` (10) keeps ReduceLROnPlateau from cutting the LR before learning starts.
- On tiny synthetic data, eval-mode accuracy can collapse even though AUC is perfect. That's the dropout→BatchNorm variance shift in this architecture, not a bug (see `test_learns_separable_data`).

**Metrics scripts** parse headerless beamline CSVs with columns `DateTime, Histogram, GoniOwl, Human, Image1, Image2`. The label-normalisation code is copy-pasted across `GoniOwlMetrics.py`, `infer_all.py` and `disagreement_inference.py`.
