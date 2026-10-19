"""
GoniOwl binary classifier training script (PyTorch) with logging, optional
hyperparameter tuning (Optuna), TensorBoard/CSV logging, per-epoch checkpoints
and optional multi-GPU DataParallel.

Example:
  python train_binary_classification_model.py \
    --train-dir /data/train \
    --val-dir /data/val \
    --tmpdir ./outputs \
    --img-height 764 --img-width 1092 \
    --batch-size 16 --epochs 100 \
    --log-level INFO --log-file ./outputs/run.log \
    --parallel \
    --tune
"""

import argparse
import csv
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn

from GoniOwl.model import (
    GoniOwlNet,
    Hyperparameters,
    count_parameters,
    save_checkpoint,
    select_device,
    unwrap,
)
from GoniOwl.preprocess import load_image

try:
    import optuna

    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for console and optional file."""
    numeric_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    fmt = "[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.debug("Logger initialized (level=%s, file=%s)", level, log_file)
    return logger


logger = logging.getLogger(__name__)

METRIC_KEYS = ["loss", "accuracy", "precision", "recall", "auc"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ImageSet:
    """Resized float32 NHWC images held in memory, the equivalent of tf.data's cache()."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def class_split(self) -> dict:
        unique, counts = np.unique(self.labels.numpy(), return_counts=True)
        return {float(u): int(c) for u, c in zip(unique, counts)}


def list_image_folder(directory: str) -> tuple[list[str], np.ndarray, list[str]]:
    """List images under directory/<class>/ with classes in alphabetical order (pinoff=0, pinon=1)."""
    root = Path(directory)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class subdirectories found in {directory}")
    paths, labels = [], []
    for label, name in enumerate(class_names):
        files = sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        paths += [str(p) for p in files]
        labels += [label] * len(files)
    return paths, np.array(labels, dtype=np.int64), class_names


def load_images(paths: list[str], size: tuple[int, int], workers: int) -> np.ndarray:
    logger.info("Loading %d images at %dx%d...", len(paths), size[0], size[1])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        images = list(ex.map(lambda p: load_image(p, size), paths))
    return np.stack(images) if images else np.empty((0, size[0], size[1], 3), dtype=np.float32)


def save_sample_images(data: ImageSet, tmpdir: str, seed: int):
    """Save a few sample images to disk for visual inspection."""
    sample_dir = os.path.join(tmpdir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, idx in enumerate(rng.permutation(len(data))[:20]):
        label = int(data.labels[idx].item())
        plt.imshow(data.images[idx].numpy().astype("uint8"))
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.savefig(os.path.join(sample_dir, f"sample_{i}_label_{label}.png"))
        plt.close()


def build_datasets(
    train_dir: str,
    img_height: int,
    img_width: int,
    seed: int,
    tmpdir: str,
    val_dir: Optional[str] = None,
    val_split: float = 0.0,
    workers: int = 8,
) -> tuple[ImageSet, ImageSet, list[str]]:
    """Create in-memory train/val sets from class-per-folder directories."""
    logger.info("Using image size for training: %dx%d", img_height, img_width)
    size = (img_height, img_width)
    paths, labels, class_names = list_image_folder(train_dir)

    if val_dir:
        logger.info("Using explicit validation directory: %s", val_dir)
        val_paths, val_labels, val_class_names = list_image_folder(val_dir)
        if val_class_names != class_names:
            raise ValueError(f"Class folders differ: train {class_names} vs val {val_class_names}")
        train = ImageSet(load_images(paths, size, workers), labels)
        val = ImageSet(load_images(val_paths, size, workers), val_labels)
    else:
        if not (0.0 < val_split < 1.0):
            raise ValueError("When --val-dir is not provided, you must set --val-split in (0,1).")
        logger.info("Using train/validation split from: %s (val_split=%.2f)", train_dir, val_split)
        # Same scheme as Keras' validation_split: shuffle with the seed, last fraction is validation.
        order = np.random.default_rng(seed).permutation(len(paths))
        num_val = int(val_split * len(paths))
        train_idx, val_idx = order[: len(order) - num_val], order[len(order) - num_val :]
        images = load_images(paths, size, workers)
        train = ImageSet(images[train_idx], labels[train_idx])
        val = ImageSet(images[val_idx], labels[val_idx])

    logger.info("Training class split: %s", train.class_split())
    logger.info("Validation class split: %s", val.class_split())
    logger.info("Training samples: %d, Validation samples: %d", len(train), len(val))
    logger.info("Classes: %s", class_names)
    save_sample_images(train, tmpdir, seed)
    return train, val, class_names


def iterate_batches(data: ImageSet, batch_size: int, shuffle: bool):
    """Yield (images NHWC, labels) batches, reshuffling on every call."""
    n = len(data)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if shuffle and len(idx) == 1:
            continue  # BatchNorm cannot train on a single sample
        yield data.images[idx], data.labels[idx]


def to_model_input(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    return images.to(device, non_blocking=True).permute(0, 3, 1, 2).float()


def augmentations(x: torch.Tensor) -> torch.Tensor:
    """Random per-image augmentation of float NCHW images in 0-255.

    The pin on/off signal is subtle: keep these ranges gentle, stronger
    photometric augmentation (e.g. contrast 0.6-1.4) stops the model learning.
    """
    n, _, h, w = x.shape
    dev = x.device

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=dev) * (hi - lo) + lo

    # Rotation (+-3 degrees) and translation (+-5%) in one resample with reflection fill.
    # The affine is built in pixel space then mapped to normalised coords so that
    # rotating a non-square image does not shear it.
    angle = uniform(-math.radians(3.0), math.radians(3.0))
    cos, sin = torch.cos(angle), torch.sin(angle)
    tx, ty = uniform(-0.1, 0.1), uniform(-0.1, 0.1)  # 5% of the extent; normalised coords span 2
    theta = torch.stack(
        [
            torch.stack([cos, -sin * h / w, tx], dim=1),
            torch.stack([sin * w / h, cos, ty], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    x = x + uniform(-0.1 * 255.0, 0.1 * 255.0).view(n, 1, 1, 1)  # brightness
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * uniform(0.9, 1.1).view(n, 1, 1, 1) + mean  # contrast
    x = x + torch.randn_like(x) * 3.0  # sensor noise
    return x.clamp(0.0, 255.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def binary_metrics(labels: np.ndarray, probs: np.ndarray, loss: float) -> dict:
    pred = probs > 0.5
    truth = labels > 0.5
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    try:
        auc = float(roc_auc_score(truth, probs))
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "loss": loss,
        "accuracy": float(np.mean(pred == truth)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "auc": auc,
    }


@torch.no_grad()
def evaluate(model: nn.Module, data: ImageSet, batch_size: int, device: torch.device) -> tuple[dict, np.ndarray]:
    model.eval()
    probs, total_loss = [], 0.0
    for images, labels in iterate_batches(data, batch_size, shuffle=False):
        logits = model(to_model_input(images, device))
        total_loss += F.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item()
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.empty(0)
    return binary_metrics(data.labels.numpy(), probs, total_loss / max(len(data), 1)), probs


class PlateauLR:
    """Keras ReduceLROnPlateau semantics: cut the LR after `patience` epochs without
    improving by more than min_delta."""

    def __init__(self, optimizer, mode="max", factor=0.1, patience=5, min_delta=1e-4, min_lr=0.0):
        self.optimizer, self.factor, self.patience = optimizer, factor, patience
        self.min_delta, self.min_lr = min_delta, min_lr
        self.sign = 1.0 if mode == "max" else -1.0
        self.best = -math.inf
        self.wait = 0

    def step(self, value: float, epoch: int) -> None:
        if math.isnan(value):
            value = -math.inf
        if self.sign * value - self.min_delta > self.best:
            self.best = self.sign * value
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            for group in self.optimizer.param_groups:
                old_lr = group["lr"]
                if old_lr > self.min_lr:
                    group["lr"] = max(old_lr * self.factor, self.min_lr)
                    logger.info("Epoch %d: ReduceLROnPlateau reducing learning rate to %.4g.", epoch + 1, group["lr"])
            self.wait = 0


class EarlyStopping:
    """Keras EarlyStopping semantics on a metric to maximise."""

    def __init__(self, patience: int, restore_best_weights: bool = False):
        self.patience, self.restore = patience, restore_best_weights
        self.best = -math.inf
        self.best_state = None
        self.wait = 0

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Return True when training should stop."""
        self.wait += 1
        if not math.isnan(value) and value > self.best:
            self.best = value
            self.wait = 0
            if self.restore:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if self.wait >= self.patience and epoch > 0:
            logger.info("Epoch %d: early stopping (best val_auc=%.4f).", epoch + 1, self.best)
            if self.restore and self.best_state is not None:
                logger.info("Restoring model weights from the end of the best epoch.")
                model.load_state_dict(self.best_state)
            return True
        return False


def fit(
    model: nn.Module,
    train: ImageSet,
    val: ImageSet,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    plateau_patience: Optional[int] = None,
    early_stopping: Optional[EarlyStopping] = None,
    epoch_callback: Optional[Callable[[int, dict], None]] = None,
    writers: Optional[dict] = None,
    quiet: bool = False,
) -> dict:
    """Train ``model`` and return a Keras-style history dict of per-epoch metrics."""
    # Keras Adam defaults (epsilon=1e-7).
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    plateau = PlateauLR(optimizer, patience=plateau_patience) if plateau_patience else None
    history: dict[str, list] = {}
    step = 0

    for epoch in range(epochs):
        if not quiet:
            logger.info("Epoch %d started.", epoch + 1)
        model.train()
        total_loss, seen, probs, labels_seen = 0.0, 0, [], []
        for images, labels in iterate_batches(train, batch_size, shuffle=True):
            x = augmentations(to_model_input(images, device))
            y = labels.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * len(y)
            seen += len(y)
            probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
            labels_seen.append(labels.numpy())
            if writers:
                writers["train"].add_scalar("batch_loss", batch_loss, step)
            step += 1

        logs = binary_metrics(np.concatenate(labels_seen), np.concatenate(probs), total_loss / max(seen, 1))
        val_logs, _ = evaluate(model, val, batch_size, device)
        logs.update({f"val_{k}": v for k, v in val_logs.items()})
        logs["learning_rate"] = optimizer.param_groups[0]["lr"]
        for k, v in logs.items():
            history.setdefault(k, []).append(v)

        if not quiet:
            parts = [f"{k}={logs[k]:.4f}" for k in METRIC_KEYS + [f"val_{m}" for m in METRIC_KEYS]]
            logger.info("Epoch %d end — %s", epoch + 1, ", ".join(parts))
        if writers:
            for k in METRIC_KEYS:
                writers["train"].add_scalar(f"epoch_{k}", logs[k], epoch)
                writers["validation"].add_scalar(f"epoch_{k}", logs[f"val_{k}"], epoch)
            writers["train"].add_scalar("epoch_learning_rate", logs["learning_rate"], epoch)
            for name, param in unwrap(model).named_parameters():
                writers["train"].add_histogram(name, param.detach().cpu(), epoch)
        if epoch_callback:
            epoch_callback(epoch, logs)

        if plateau:
            plateau.step(logs["val_auc"], epoch)
        if early_stopping and early_stopping.step(logs["val_auc"], epoch, model):
            break

    return history


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

TUNERS = ("hyperband", "bayesianoptimization", "gridsearch", "randomsearch")


def suggest_hyperparameters(trial: "optuna.Trial") -> tuple[Hyperparameters, float]:
    hp = Hyperparameters(
        conv1=trial.suggest_int("conv1", 16, 128, step=8),
        conv1_2=trial.suggest_int("conv1_2", 24, 128, step=8),
        conv2=trial.suggest_int("conv2", 16, 128, step=8),
        conv2_2=trial.suggest_int("conv2_2", 24, 128, step=8),
        dense_units_1=trial.suggest_int("dense_units_1", 32, 192, step=32),
        dense_units_2=trial.suggest_int("dense_units_2", 32, 192, step=32),
    )
    learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-3, 2e-3])
    return hp, learning_rate


def make_study(tuner: str, seed: int, max_epochs: int) -> "optuna.Study":
    pruner = optuna.pruners.NopPruner()
    if tuner == "bayesianoptimization":
        sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=5)
    elif tuner == "gridsearch":
        space = {
            "conv1": list(range(16, 129, 8)),
            "conv1_2": list(range(24, 129, 8)),
            "conv2": list(range(16, 129, 8)),
            "conv2_2": list(range(24, 129, 8)),
            "dense_units_1": list(range(32, 193, 32)),
            "dense_units_2": list(range(32, 193, 32)),
            "learning_rate": [5e-4, 1e-3, 2e-3],
        }
        sampler = optuna.samplers.GridSampler(space, seed=seed)
    elif tuner == "randomsearch":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        # Hyperband: random configurations, successive-halving on the epoch budget.
        sampler = optuna.samplers.RandomSampler(seed=seed)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_epochs, reduction_factor=3)
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)


def tune(args, train: ImageSet, val: ImageSet, device: torch.device, directory: str) -> tuple[Hyperparameters, float]:
    os.makedirs(directory, exist_ok=True)
    logger.info("Starting hyperparameter tuning with %s...", args.tuner)
    logger.info("Tuner directory: %s", directory)
    epochs = args.tune_max_epochs if args.tuner == "hyperband" else args.tune_search_epochs
    study = make_study(args.tuner, args.seed, epochs)

    def objective(trial: "optuna.Trial") -> float:
        hp, lr = suggest_hyperparameters(trial)
        torch.manual_seed(args.seed + trial.number)
        model = GoniOwlNet(hp).to(device)
        best = 0.0

        def report(epoch: int, logs: dict) -> None:
            nonlocal best
            best = max(best, logs["val_accuracy"])
            trial.report(logs["val_accuracy"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit(
            model, train, val,
            epochs=epochs, learning_rate=lr, batch_size=args.batch_size, device=device,
            early_stopping=EarlyStopping(args.es_patience), epoch_callback=report, quiet=True,
        )
        logger.info("Trial %d: val_accuracy=%.4f %s lr=%s", trial.number, best, hp, lr)
        return best

    study.optimize(objective, n_trials=args.tune_trials)
    study.trials_dataframe().to_csv(os.path.join(directory, "trials.csv"), index=False)
    logger.info("Hyperparameter search finished.")

    best_hp, best_lr = suggest_hyperparameters(optuna.trial.FixedTrial(study.best_params))
    logger.info(
        "Best HPs — conv1=%s conv1_2=%s conv2=%s conv2_2=%s dense1=%s dense2=%s lr=%s (val_accuracy=%.4f)",
        best_hp.conv1, best_hp.conv1_2, best_hp.conv2, best_hp.conv2_2,
        best_hp.dense_units_1, best_hp.dense_units_2, best_lr, study.best_value,
    )
    return best_hp, best_lr


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, tmpdir: str, name_suffix: str = ""):
    """Plot training vs validation loss and accuracy over epochs."""
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    if "loss" in history:
        ax_loss.plot(epochs, history["loss"], label="train")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()

    if "accuracy" in history:
        ax_acc.plot(epochs, history["accuracy"], label="train")
    if "val_accuracy" in history:
        ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.legend()

    fig.tight_layout()
    fname = f"training_curves_{name_suffix}.png" if name_suffix else "training_curves.png"
    out_path = os.path.join(tmpdir, fname)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Training curves saved to: %s", out_path)


def plot_confusion_matrix(y_true: np.ndarray, probs: np.ndarray, out_path: str) -> np.ndarray:
    cm = confusion_matrix(y_true, probs > 0.5, labels=[0, 1])
    _, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


def run(args: argparse.Namespace) -> str:
    """Train a model and return the path of the saved final model."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPUs detected: %s", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    now_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    img_size = (int(args.img_height / args.image_divider), int(args.img_width / args.image_divider))

    train, val, class_names = build_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_height=img_size[0],
        img_width=img_size[1],
        seed=args.seed,
        tmpdir=tmpdir,
        val_split=args.val_split,
        workers=args.workers,
    )

    if args.tune:
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed. Install with: pip install optuna")
        directory = os.path.join(tmpdir, f"tune_b{args.batch_size}_{now_string}")
        hp, learning_rate = tune(args, train, val, device, directory)
        torch.manual_seed(args.seed)
    else:
        logger.info("Building predefined model...")
        hp, learning_rate = Hyperparameters(), args.learning_rate

    model: nn.Module = GoniOwlNet(hp).to(device)
    for line in str(model).splitlines():
        logger.info(line)
    logger.info("Total params (incl. BatchNorm statistics): %d", count_parameters(model))
    if args.parallel:
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            logger.info("Running with DataParallel (replicas=%d).", torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            logger.warning("--parallel requested but fewer than 2 CUDA GPUs are available; using one device.")

    log_dir = os.path.join(tmpdir, "logs", "fit", now_string)
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writers = None
    if HAS_TENSORBOARD:
        logger.info("TensorBoard log dir: %s", log_dir)
        writers = {
            "train": SummaryWriter(os.path.join(log_dir, "train")),
            "validation": SummaryWriter(os.path.join(log_dir, "validation")),
        }
    else:
        logger.warning("TensorBoard not installed; skipping TensorBoard logging.")
    csv_log_path = os.path.join(log_dir, f"training_{now_string}.csv")
    logger.info("CSVLogger path: %s", csv_log_path)

    csv_file = open(csv_log_path, "w", newline="")
    csv_writer = None

    def on_epoch_end(epoch: int, logs: dict) -> None:
        nonlocal csv_writer
        save_checkpoint(
            os.path.join(ckpt_dir, f"{now_string}_epoch{epoch + 1:02d}_binary_batch{args.batch_size}.pt"),
            model, img_size, class_names, epoch=epoch + 1, **logs,
        )
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch"] + sorted(logs))
            csv_writer.writeheader()
        csv_writer.writerow({"epoch": epoch, **logs})
        csv_file.flush()

    logger.info("Training started.")
    logger.info(
        "Starting training for %d epochs (batch=%d, image=%dx%d)...",
        args.epochs, args.batch_size, img_size[0], img_size[1],
    )
    try:
        history = fit(
            model, train, val,
            epochs=args.epochs,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            device=device,
            # Monitor val_auc, NOT val_loss: on this small dataset val_loss/val_accuracy
            # swing wildly from epoch to epoch (BatchNorm running-stat noise), while
            # val_auc is stable ~0.998. Monitoring val_loss made the scheduler slash the
            # LR on noise and freeze the model at the majority-class baseline.
            plateau_patience=args.plateau_patience,
            early_stopping=EarlyStopping(args.es_patience, restore_best_weights=True) if args.early_stopping else None,
            epoch_callback=on_epoch_end,
            writers=writers,
        )
    finally:
        csv_file.close()
        if writers:
            for w in writers.values():
                w.close()
    logger.info("Training complete.")
    logger.info("Training finished. Final val_accuracy=%.4f.", history["val_accuracy"][-1])
    plot_training_curves(history, tmpdir, name_suffix=now_string)

    final_model_path = os.path.join(
        tmpdir,
        f"{now_string}_binary_batch{args.batch_size}_div{args.image_divider}{'_tuned' if args.tune else ''}.pt",
    )
    logger.info("Saving model to: %s", final_model_path)
    save_checkpoint(
        final_model_path, model, img_size, class_names,
        learning_rate=learning_rate, batch_size=args.batch_size, seed=args.seed, epochs=len(history["loss"]),
    )
    logger.info("Model saved successfully.")

    logger.info("Running predictions on validation set for confusion matrix...")
    _, probs = evaluate(model, val, args.batch_size, device)
    cm = plot_confusion_matrix(val.labels.numpy(), probs, os.path.join(tmpdir, f"confusionmatrix_{now_string}.png"))
    logger.info("Validation confusion matrix [[TN, FP], [FN, TP]]: %s", cm.tolist())
    return final_model_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier with logging and optional tuning.")

    parser.add_argument("--train-dir", required=True, help="Path to training images directory.")
    parser.add_argument("--val-dir", default=None, help="Path to validation images directory (optional).")
    parser.add_argument("--val-split", type=float, default=0.3, help="Validation split (0-1) if --val-dir not provided.")
    parser.add_argument("--img-height", type=int, default=764, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=1092, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size. Default 16 matches the manuscript (batch-16/seed-0) model. A multi-seed sweep found batch>=8 trains reliably; 2-4 collapse to ~chance.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default 0 matches the manuscript (batch-16/seed-0) model.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable EarlyStopping on val_auc.")
    parser.add_argument("--es-patience", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--plateau-patience", type=int, default=5, help="Patience for ReduceLROnPlateau.")

    parser.add_argument("--tune", action="store_true", help="Use Optuna to search hyperparameters.")
    parser.add_argument("--tuner", type=str, default="hyperband", choices=TUNERS, help="Search method: hyperband (random sampling + Hyperband pruning), bayesianoptimization (Gaussian process), gridsearch, randomsearch.")
    parser.add_argument("--tune-max-epochs", type=int, default=15, help="Hyperband only: max epochs per trial.")
    parser.add_argument("--tune-trials", type=int, default=25, help="Total models to try. Small values (e.g. 3) get stuck at the majority-class baseline on this dataset.")
    parser.add_argument("--tune-search-epochs", type=int, default=5, help="Epochs per trial for bayesianoptimization/gridsearch/randomsearch; Hyperband uses --tune-max-epochs.")

    parser.add_argument("--parallel", action="store_true", help="Use DataParallel across all visible CUDA GPUs. Each GPU sees batch-size/num_gpus samples and keeps its own BatchNorm statistics, so raise --batch-size to keep >=8 per GPU.")
    parser.add_argument("--device", default="auto", help="Device to train on: auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Threads used to decode images.")
    parser.add_argument("--tmpdir", default="./outputs", help="Base output directory for logs/checkpoints/models.")
    parser.add_argument("--image-divider", type=int, default=5, help="Value to divide image width and height by before training.")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")
    parser.add_argument("--log-file", default=None, help="Optional path to a log file.")

    args = parser.parse_args(argv)

    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"--train-dir does not exist: {args.train_dir}")
    if args.val_dir and not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f"--val-dir does not exist: {args.val_dir}")
    if not args.val_dir and not (0.0 < args.val_split < 1.0):
        raise ValueError("--val-split must be set in (0,1) when --val-dir is not provided.")

    return args


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("Arguments: %s", vars(args))
    run(args)
