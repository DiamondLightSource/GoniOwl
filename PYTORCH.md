# GoniOwl in PyTorch

This page covers the PyTorch port of the GoniOwl model: how it relates to the original TensorFlow/Keras version, how to use it from Python, and what the hyperparameter search found. For installation and the command-line workflow, see the [README](README.md).

The port covers the model, training, conversion, evaluation and the offline metrics scripts. The beamline IOC (`src/GoniOwl/__main__.py`, `GoniOwl_controller.py`) and the Dockerfile are unchanged: they still load the Keras `.h5` model with TensorFlow.

## What maps to what

| Keras | PyTorch |
|---|---|
| `Sequential` model built in `train_binary_classification_model.py` | `GoniOwlNet(Hyperparameters)` in `src/GoniOwl/model.py` |
| `Rescaling(1/255)` layer | `forward` divides by 255; it takes raw 0–255 pixels |
| `Dense(1, activation="sigmoid")` | final `Linear` returns a logit; `predict_proba` applies the sigmoid |
| `BatchNormalization()` defaults | `eps=1e-3`, `momentum=0.01` (Keras' 0.99 in PyTorch's convention) |
| `glorot_uniform` initialiser | `xavier_uniform_` |
| `Adam()` | `Adam(eps=1e-7)` |
| `tf.image.resize` (bilinear) | `preprocess.resize`: `F.interpolate(..., antialias=False)` |
| `ReduceLROnPlateau`, `EarlyStopping` | `PlateauLR`, `EarlyStopping` (same semantics, on `val_auc`) |
| Keras augmentation layers | `augmentations()`: per-image on the GPU, same ranges |
| KerasTuner | Optuna (`--tuner hyperband` = random sampling + `HyperbandPruner`) |
| `.h5` / `.keras` | `.pt` checkpoint (`format: goniowl-torch-v1`) |

`count_parameters` includes the BatchNorm running statistics, so it gives the same totals as Keras' `model.summary()`: 51,313 for the default architecture.

## Using the model from Python

```python
from GoniOwl.model import load_checkpoint, predict_proba
from GoniOwl.preprocess import decide, is_image_valid, load_image

model, info = load_checkpoint("src/GoniOwl/20260626_140916_epoch30_binary_batch16.pt")
size = (info["img_height"], info["img_width"])

image = load_image("frame.jpg", size, border=100)  # border=0 if the image is already cropped
if is_image_valid(image):
    score = float(predict_proba(model, image)[0])  # P(pin on)
    status, label = decide(score)  # 1 = pin off, 2 = pin on, 3 = unsure
```

Each checkpoint stores its architecture (`hparams`), input size, class names and training metadata, so nothing has to be hard-coded. Always save with `save_checkpoint` and load with `load_checkpoint`, not raw `torch.save`.

## Preprocessing must match everywhere

All of it is in `src/GoniOwl/preprocess.py`, and every consumer uses it:

1. Crop 100 px from every side of the raw camera frame.
2. Resize with bilinear interpolation **without antialiasing**. This matches `tf.image.resize`, to within 1.5e-5. PIL's `BILINEAR` antialiases, and it made the 2025 model confidently call pin-on frames pin off.
3. Feed raw 0–255 RGB. Don't normalise it yourself.

JPEG decoding still differs slightly: TensorFlow's default fast DCT and PIL's accurate DCT disagree by about one grey level on some pixels. This doesn't change any decisions.

## Converted Keras models

`src/Training/convert_keras_to_torch.py` reads `.h5` and Keras 3 `.keras` files with `h5py`, so it doesn't need TensorFlow. Both shipped Keras models are committed in converted form in `src/GoniOwl/`:

- `20260626_140916_epoch30_binary_batch16.pt` is the manuscript model. It matches Keras to about 1e-6.
- `20250714-111015_epoch20_binary_batch4.pt` is the model deployed in 2025. Keras ran it in `mixed_float16`, so the float32 port differs by about 1e-3.

Both give the same decisions as Keras on the test fixtures (`tests/fixtures/`), and `test_committed_checkpoints_are_current` checks they still match their Keras sources.

## Training behaviour worth knowing

- **Learning is sudden.** Every model guesses at chance for the first 3–10 epochs, then jumps to about 99% within an epoch or two. How long that takes depends on the learning rate and the network's width.
  - At lr 2e-3, nothing learns.
  - The default architecture starts by epoch 3 at lr 1e-3.
  - Networks 4–5× wider don't learn at 1e-3 at all; they need 5e-4 or less, and take 5–10 epochs to start.
  - `--plateau-warmup` (default 10) keeps the learning-rate schedule from cutting the rate during that phase.
- **Whether training starts at all is partly luck.** For some architectures, one random seed learns and another stays at chance. The default architecture at lr 1e-3 starts reliably for batch sizes of 8 and above.
- **Validation accuracy swings during the early epochs.** It can jump between about 0.5 and 0.99 from one epoch to the next while BatchNorm's running statistics settle. That's why the scheduler and early stopping watch `val_auc`.
- **On tiny synthetic data, accuracy collapses in eval mode while AUC stays perfect.** This comes from dropout feeding BatchNorm in this architecture (the same layer order as the Keras model). It doesn't happen on the real data.
- `--duty-cycle` throttles training if a laptop's charger can't keep up with the GPU.

## Hyperparameter search results (October 2026)

**What was run:**
- **Search:** Optuna Hyperband, 40 trials per batch size, each trial up to 30 epochs and never pruned before epoch 8.
- **Search space:**
  - convolution widths 16–128
  - dense layer widths 32–192
  - learning rates 1e-4 to 1e-3
- **Final models:** the best trial was retrained for 100 epochs with seed 0.
- **Data:** 7,878 training images; the held-out test set is 875 images.

| model | params | unsure | confident errors | test log-loss |
|---|---|---|---|---|
| **default**: 16/48/16/96, dense 96/128, lr 1e-3, batch 16 | 51k | 1 | 0 | 0.00041 |
| best at batch 16: 80/104/88/80, dense 96/128, lr 5e-4 | 245k | 0 | 0 | 0.00036 |
| best at batch 32: 32/112/16/96, dense 160/160, lr 5e-4 | 106k | 3 | 0 | 0.0044 |
| best at batch 64: 40/104/64/80, dense 32/128, lr 5e-4 | 153k | 184 | 17 | 0.21 |
| best at batch 8: 64/112/96/56, dense 160/96, lr 5e-4 | 238k | 875 | — (never learned) | 0.68 |

**Findings:**
- **The search can't rank architectures on this data.** Dozens of configurations, from about 60k to 300k parameters, reached the same best validation accuracy: 99.92%, or 2 errors in 2,363 images.
- **What separates them is whether they train reliably.** Two of the four winners failed when retrained from a different random seed.
- **The default architecture stays.** It's as accurate as anything the search found, at a fifth of the size, and it starts learning reliably.

## Decision mapping

`preprocess.decide` maps P(pin on) to the `GONIOSTATUS` value that GDA reads:

| score | status |
|---|---|
| < 0.15 | 1: pin off |
| > 0.85 | 2: pin on |
| anything else | 3: unsure (stop) |

Images whose mean grey level is outside 80–180 are also unsure. This is the same mapping as the IOC code. The README's "Beamline Implementation" section documents 1 as pin on, which is the opposite; that still needs confirming against GDA.
