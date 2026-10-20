# GoniOwl - Sample Pin Detection for Beamline Collision Prevention

![IMG_0781(1)](https://github.com/user-attachments/assets/8b08de35-6e63-461e-a933-2129fb25e74c)

This repository contains the training and deployment code for a convolutional neural network (CNN) that detects the presence or absence of sample pins on a goniometer, in this case beamline I23 at Diamond Light Source. The model is designed to prevent equipment collisions during automated sample handling by providing real-time classification from beamline camera feeds.

The workflow is implemented in **PyTorch**. Models trained with the earlier TensorFlow/Keras version can be converted (see [Converting Keras models](#converting-keras-models)).

---
# Usage

## Installation

### Using UV (Recommended)
- Create UV virtual environment: `uv venv`
- Activate virtual environment: `source .venv/bin/activate`
- Install the package and its dependencies: `uv pip install -e ".[dev]"`
- Or reproduce the exact locked environment: `uv sync --extra dev`

### Using pip
- Create Python virtual environment: `python -m venv .venv`
- Activate virtual environment: `source .venv/bin/activate`
- Install the package and its dependencies: `pip install -e ".[dev]"`

Python 3.12+ is required. The `-e` flag installs the package in editable mode, allowing you to modify the code without reinstalling. On Linux the PyPI `torch` wheel bundles CUDA; on Apple silicon training runs on the GPU via MPS.

## Data preparation

- Alter `import_crop_save.py` if your input images are in a different directory structure. It expects `<snapshots>/pin_ON/Pin_gripper_on_gonio` and `<snapshots>/pin_OFF/Before_sample_load`.
- Run `python src/Training/import_crop_save.py --snapshots /path/to/snapshots --imgdir /path/to/save/cropped/images`
- This crops 100 px from every side of each image into `<imgdir>/goniopin_auto_<ddmmyyyy>_binary/{pinon,pinoff}` and moves a random 10% into `<imgdir>/test_<ddmmyyyy>_binary` as a held-out test set.
- Check images in output folders manually.

## Training

- Run training `python src/Training/train_binary_classification_model.py --train-dir /path/to/save/cropped/images/goniopin_auto_<ddmmyyyy>_binary`
- Add `--tune` to search hyperparameters with Optuna first (`--tuner hyperband|bayesianoptimization|randomsearch|gridsearch`, `--tune-trials N`).
  Every model stays at chance for the first 3–7 epochs, and wider models only learn at learning rates of 2e-4 or below, so the search uses learning rates 1e-4–1e-3 and Hyperband cannot prune a trial before `--tune-min-epochs` (default 8).
- `--hparams '{"conv1": 80, ..., "learning_rate": 5e-4}'` trains a given architecture, e.g. the best trial from a search's `trials.csv`, without searching again.
- `--plateau-warmup` (default 10) stops ReduceLROnPlateau from acting during the first epochs, when the model is still at chance; without it, configurations at lr ≤ 5e-4 get their learning rate cut before they start learning.
- `--duty-cycle 0.5` sleeps between batches so the GPU only trains half the time. Use it on a laptop whose charger can't keep up with a long run.
- `--device` picks the device (default `auto`: CUDA, then MPS, then CPU). `--parallel` uses `DataParallel` over all CUDA GPUs; each GPU then sees `batch-size / num_gpus` samples with its own BatchNorm statistics, so raise `--batch-size` accordingly.
- Batch size matters on this small dataset. A multi-seed sweep (5 seeds × batch sizes 2–64) showed batch ≥ 8 trains reliably to ~0.99, while batch 2–4 frequently collapse to the majority-class baseline (~0.53). The default `--batch-size` is **16** and `--seed` is **0**, matching the model reported in the manuscript.
- Outputs (in `--tmpdir`, default `./outputs`): the final model `<timestamp>_binary_batch16_div5.pt`, a checkpoint per epoch in `checkpoints/`, TensorBoard logs and a CSV log in `logs/fit/<timestamp>/`, training curves, a validation confusion matrix and sample images.
- On Diamond's cluster: `sbatch src/Training/train_slurm.sh /path/to/goniopin_auto_<ddmmyyyy>_binary`

## Evaluating on the held-out test set

```bash
python src/Metrics/evaluate_test_set.py \
  --model-path ./outputs/YOUR_MODEL.pt \
  --test-dir /path/to/save/cropped/images/test_<ddmmyyyy>_binary
```

Reports coverage, accuracy on the confident subset and the confusion matrix using the same 0.85 confidence gate as the IOC.

## Converting Keras models

Models trained with the TensorFlow version (`.h5`, or `.keras`) can be converted to PyTorch checkpoints without installing TensorFlow:

```bash
python src/Training/convert_keras_to_torch.py --model old_model.keras --output old_model.pt
```

Converted models reproduce the Keras predictions (to ~1e-6; ~1e-3 for the 2025 model, which Keras ran in `mixed_float16`). Both committed Keras models are provided converted in `src/GoniOwl/`.

## Inference

- Copy model from training into src/GoniOwl/
- Alter GoniOwl_controller.py to target model of choice and set log output directory.
- From src/GoniOwl/ run `pipenv run python -m GoniOwl`. This will launch the ioc from this terminal, so it is advisable to run this via screen/tmux.
- The IOC (`__main__.py`, `GoniOwl_controller.py`) has not been ported yet: it still loads the Keras `.h5` model with TensorFlow and runs on `cothread`/`softioc`, which are not installed by `pyproject.toml`. For offline PyTorch inference use the scripts below.

## Metrics on beamline logs

These read the GoniOwl decision logs (`/dls_sw/i23/logs/GoniOwl/*.csv` by default, `--log-dir` to override). The model input size is read from the checkpoint; use `--crop-border 100` if the logged images are raw, uncropped camera frames.

- `src/Metrics/infer_all.py --model-path MODEL.pt`: fresh inference on every logged image, with annotated images, a CSV and confidence histograms.
- `src/Metrics/disagreement_inference.py --model-path MODEL.pt`: inference on the images where the logged GoniOwl decision and the human disagree.
- `src/Metrics/GoniOwlMetrics.py`: agreement statistics between histogram, GoniOwl and human decisions.

## Inference on Disagreements

To evaluate model performance on images where the live model and training model disagree:

```bash
python src/Metrics/inference_on_disagreements.py \
  --model-path ./outputs/YOUR_MODEL.pt \
  --disagreement-dir /dls/science/groups/i23/scripts/chris/GoniOwl_metrics/disagreements \
  --output-dir ./outputs/inference_results
```

**Arguments:**
- `--model-path`: Path to the trained `.pt` model file (required)
- `--disagreement-dir`: Path to folder containing misclassified/disagreement images (required)
- `--img-height` / `--img-width`: Override the image size (default: the size stored in the checkpoint, e.g. 152×218 for `--image-divider 5` on 764×1092 images)
- `--output-dir`: Directory to save results and visualizations (default: `./outputs/inference_results`)

**Output:**
- PNG visualizations of each image with predictions saved to `--output-dir`
- CSV summary file (`inference_results.csv`) with filename, predicted class, raw prediction score, and confidence for all images

## Testing

- Unit and integration tests: `pytest` (or `pytest -m "not slow"` to skip the tests that train models).
- To check that GoniOwl is running and serving PVs, try `caget GONIOWL-TEST:WHOAMI` -> `GONIOWL-TEST:WHOAMI            GoniOwl Python IOC`
- To check that inference is working, try:
`caput GONIOWL-TEST:INFER 1` -> `Old : GONIOWL-TEST:INFER             Disable` `New : GONIOWL-TEST:INFER             Enable`
then
`caget GONIOWL-TEST:GONIOSTATUS` -> `GONIOWL-TEST:GONIOSTATUS       1`

## Beamline Implementation

- On I23, Diamond Light Source, the PV `GONIOWL-TEST:INFER` is called by GDA. Logic inside the ioc runs an inference on the latest image from the specified camera feed. A sleep of 0.1s is added here to allow time for the inference to occur with allowance for filesystem lag. 
- The logic first checks if the image is too light or too dark by getting an average of all the pixels. If this is the case, a PV `GONIOWL-TEST:GONIOSTATUS` is set to 3 and used in GDA as a failsafe stop.
- If the image passes this first check. Inference is run using the model and a score between 0-1 is generated. 0 indicates 100% certainty of a pin on the goniometer and 1 indicates 100% certainty of no pin on the goniometer. At least 85% certainty is needed to give a definitive decision, so if the score is < 0.15, the PV `GONIOWL-TEST:GONIOSTATUS` is set to 1. If the score is > 0.85, the PV `GONIOWL-TEST:GONIOSTATUS` is set to 2. If the score is (0.85 <= 0.15), the PV `GONIOWL-TEST:GONIOSTATUS` is set to 3. 
- This PV is checked by GDA to give a decision on whether there is a pin present on the goniometer of not. 1 = pin on, 2 = pin off, 3 = not sure so stop.

# Overview

Synchrotron beamlines require precise mechanical coordination to avoid collisions between components such as goniometers, sample changers, and detectors. Traditional vision-based methods (e.g., histogram analysis) are sensitive to lighting and camera variability, limiting their reliability.

This project introduces a compact CNN classifier trained on real beamline images to robustly detect sample pin presence. Integrated into the beamline control system, it enables real-time inference and fail-safe decision-making, significantly improving safety and throughput.

---

# Features

- **Binary classification**: Detects `sample_on` vs `sample_off` states.
- **Robust to lighting and viewpoint changes** via data augmentation.
- **Real-time inference** on standard beamline control PCs (CPU/GPU).
- **Fail-safe logic** for low-confidence or invalid exposures.
- **Training pipeline** with augmentation, validation split, and logging.
- **Optional hyperparameter tuning** via Optuna.
- **TensorBoard, CSV logging, and checkpointing** included.

---

# Performance

Held-out test set (875 images), deployment-style offline inference with confidence gating (confident iff max(score, 1−score) > 0.85; otherwise the fail-safe *undetermined* state):
- **Coverage**: 100% (875/875 confident)
- **Accuracy (confident subset)**: 99.77% (873/875)
- **Confusion matrix** [[TN, FP], [FN, TP]], classes [pinoff, pinon]: [[458, 0], [2, 415]]

Evaluated with manually verified labels. Over a 3-month deployment, the model prevented multiple potential collisions and outperformed legacy histogram-based methods.

The PyTorch pipeline, trained from scratch with the defaults (batch 16, seed 0, 100 epochs, 30% validation split; ~11 min on an Apple M5 Max via MPS), on a fresh 875-image held-out split (458 pin off, 417 pin on) evaluated with `evaluate_test_set.py`:
- **Coverage**: 99.89% (874/875 confident; the one undetermined image is a pin-on frame scored 0.84)
- **Accuracy (confident subset)**: 100% (874/874)
- **Confusion matrix** [[TN, FP], [FN, TP]]: [[458, 0], [0, 416]]
- **Validation** (2,363 images): accuracy 99.92%

Raw training images: 10.5281/zenodo.17047675

Singularity build used for training the manuscript (TensorFlow) model: 10.5281/zenodo.17047767

# Issues
- **Keep augmentation gentle** — the pin on/off signal is subtle; overly aggressive photometric augmentation (e.g. contrast 0.6–1.4) stops the model learning entirely (training stuck at chance). The current `augmentations()` ranges are tuned to reach ~0.99 validation accuracy.
- **Resize without antialiasing** — images are resized with bilinear interpolation without antialiasing, exactly like `tf.image.resize` in the original training. Antialiased resizing (e.g. PIL's `BILINEAR`) makes the 2025 model confidently misclassify some pin-on frames as pin-off.
- **Monitor `val_auc`** — the learning-rate scheduler and early stopping monitor `val_auc`, not `val_loss`; on this small dataset `val_loss` is noisy enough to collapse training to the majority class.
- **cv2 clobbers CUDA libpath** (TensorFlow version only) — TF dropped to CPU because cv2 overrode `LD_LIBRARY_PATH`. The PyTorch wheels locate their CUDA libraries themselves, so the re-exec workaround is no longer needed.
