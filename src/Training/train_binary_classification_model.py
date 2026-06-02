"""
GoniOwl binary classifier training script with logging, optional hyperparameter tuning,
TensorBoard/CSV logging, checkpoints, and MirroredStrategy.

Example:
  python train_binary_classification_model.py \
    --train-dir /data/train \
    --val-dir /data/val \
    --tmpdir ./outputs \
    --img-height 224 --img-width 224 \
    --batch-size 32 --epochs 50 \
    --log-level INFO --log-file ./outputs/run.log \
    --parallel \
    --tune
"""

import os
import sys

def _ensure_cuda_libpath():
    if os.environ.get("_GONIOWL_CUDA_LIBPATH") == "1":
        return
    try:
        import site
        site_pkgs = site.getsitepackages()
    except Exception:
        site_pkgs = []
    lib_dirs = []
    for sp in site_pkgs:
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in sorted(os.listdir(nvidia_root)):
            libdir = os.path.join(nvidia_root, pkg, "lib")
            if os.path.isdir(libdir):
                lib_dirs.append(libdir)
    if not lib_dirs:
        os.environ["_GONIOWL_CUDA_LIBPATH"] = "1"
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + ([existing] if existing else []))
    os.environ["_GONIOWL_CUDA_LIBPATH"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_cuda_libpath()

import io
import json
import zipfile
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend as K

try:
    import keras_tuner as kt
    HAS_TUNER = True
except Exception:
    HAS_TUNER = False

from typing import Optional, Tuple, List

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for console and optional file."""
    numeric_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    fmt = "[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.debug("Logger initialized (level=%s, file=%s)", level, log_file)
    return logger


logger = logging.getLogger(__name__)

_KEYS_TO_STRIP = {"quantization_config"}


def _strip_unsupported_keys_from_obj(obj):
    """Recursively remove keys that break older Keras 3 versions."""
    if isinstance(obj, dict):
        return {k: _strip_unsupported_keys_from_obj(v) for k, v in obj.items() if k not in _KEYS_TO_STRIP}
    if isinstance(obj, list):
        return [_strip_unsupported_keys_from_obj(item) for item in obj]
    return obj


def _strip_unsupported_keys(keras_path: str) -> None:
    """Patch a .keras file in-place to remove keys unsupported by older Keras 3."""
    if not zipfile.is_zipfile(keras_path):
        return
    buf = io.BytesIO()
    with zipfile.ZipFile(keras_path, "r") as zin, zipfile.ZipFile(buf, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                config = _strip_unsupported_keys_from_obj(config)
                data = json.dumps(config, indent=2).encode("utf-8")
            zout.writestr(item, data)
    with open(keras_path, "wb") as f:
        f.write(buf.getvalue())
    logger.info("Stripped unsupported keys %s from saved model.", _KEYS_TO_STRIP)


class LoggingCallback(keras.callbacks.Callback):
    """Log concise, per-epoch metrics."""

    def on_train_begin(self, logs=None):
        logger.info("Training started.")

    def on_epoch_begin(self, epoch, logs=None):
        logger.info("Epoch %d started.", epoch + 1)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        keys = [
            "loss", "accuracy", "precision", "recall", "auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_auc"
        ]
        parts = [f"{k}={logs[k]:.4f}" for k in keys if k in logs and logs[k] is not None]
        logger.info("Epoch %d end — %s", epoch + 1, ", ".join(parts) if parts else "(no logs)")

    def on_train_end(self, logs=None):
        logger.info("Training complete.")

rotation_layer = layers.RandomRotation(factor=3.0 / 360.0, fill_mode="reflect", interpolation="bilinear")
translation_layer = layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="reflect")

def augmentations(image, label):
    image = tf.cast(image, tf.float32)
    image = rotation_layer(image)
    image = translation_layer(image)
    image = tf.image.random_brightness(image, max_delta=0.1 * 255.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = image + (tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=3.0))
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def save_sample_images(dataset: tf.data.Dataset, tmpdir: str):
    """Save a batch of sample images to disk for visual inspection."""
    sample_dir = os.path.join(tmpdir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    for images, labels in dataset.take(1):
        for i in range(min(20, len(images))):
            label = int(labels[i].numpy().item())
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.savefig(os.path.join(sample_dir, f"sample_{i}_label_{label}.png"))
            plt.close()


def plot_training_curves(history, tmpdir: str):
    """Plot training vs validation loss and accuracy over epochs."""
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    if "loss" in hist:
        ax_loss.plot(epochs, hist["loss"], label="train")
    if "val_loss" in hist:
        ax_loss.plot(epochs, hist["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()

    if "accuracy" in hist:
        ax_acc.plot(epochs, hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        ax_acc.plot(epochs, hist["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.legend()

    fig.tight_layout()
    out_path = os.path.join(tmpdir, "training_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Training curves saved to: %s", out_path)


def predefined_model(img_height: int, img_width: int, learning_rate: float = 1e-3) -> keras.Model:
    """Predetermined model architecture."""
    logger.info("Building predefined model...")
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(20, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(44, (3, 3), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(24, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(48, (3, 3), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(176, activation="silu", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation="silu", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(1, activation="sigmoid", dtype="float32"))

    model.compile(
        keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    logger.info("Model compiled.")
    model.summary(print_fn=lambda l: logger.info(l))
    return model


def model_builder(hp: "kt.HyperParameters", img_height: int, img_width: int) -> keras.Model:
    """For tuning model with KerasTuner."""
    conv1 = hp.Int("conv1", min_value=16, max_value=256, step=8, default=20)
    conv1_2 = hp.Int("conv1_2", min_value=24, max_value=256, step=8, default=44)
    conv2 = hp.Int("conv2", min_value=16, max_value=256, step=8, default=24)
    conv2_2 = hp.Int("conv2_2", min_value=24, max_value=256, step=8, default=48)
    dense1 = hp.Int("dense_units_1", min_value=32, max_value=256, step=32, default=176)
    dense2 = hp.Int("dense_units_2", min_value=32, max_value=256, step=32, default=64)
    learning_rate = hp.Choice("learning_rate", values=[5e-4, 1e-3, 2e-3], default=1e-3)

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(conv1, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(conv1_2, (3, 3), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(conv2, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(conv2_2, (3, 3), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(dense1, activation="silu", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(dense2, activation="silu", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(1, activation="sigmoid", dtype="float32"))

    model.compile(
        keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(
    train_dir: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    seed: int,
    tmpdir: str,
    val_dir: Optional[str] = None,
    val_split: float = 0.0,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Create train/val datasets from directories."""
    logger.info("Using image size for training: %dx%d", img_height, img_width)
    if val_dir:
        logger.info("Using explicit validation directory: %s", val_dir)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="binary",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        all_labels = np.concatenate([y.numpy() for _, y in train_ds])
        unique, counts = np.unique(all_labels, return_counts=True)
        logger.info("Training class split: %s", dict(zip(unique, counts)))
        save_sample_images(train_ds, tmpdir)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode="binary",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        all_val_labels = np.concatenate([y.numpy() for _, y in val_ds])
        val_unique, val_counts = np.unique(all_val_labels, return_counts=True)
        logger.info("Validation class split: %s", dict(zip(val_unique, val_counts)))
        class_names = val_ds.class_names
        logger.info("Training samples: %d, Validation samples: %d", len(train_ds)*batch_size, len(val_ds)*batch_size)
        logger.info("Classes: %s", class_names)
    else:
        if not (0.0 < val_split < 1.0):
            raise ValueError("When --val-dir is not provided, you must set --val-split in (0,1).")
        logger.info("Using train/validation split from: %s (val_split=%.2f)", train_dir, val_split)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="binary",
            validation_split=val_split,
            subset="training",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        all_labels = np.concatenate([y.numpy() for _, y in train_ds])
        unique, counts = np.unique(all_labels, return_counts=True)
        logger.info("Training class split: %s", dict(zip(unique, counts)))
        save_sample_images(train_ds, tmpdir)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="binary",
            validation_split=val_split,
            subset="validation",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        all_val_labels = np.concatenate([y.numpy() for _, y in val_ds])
        val_unique, val_counts = np.unique(all_val_labels, return_counts=True)
        logger.info("Validation class split: %s", dict(zip(val_unique, val_counts)))
        class_names = val_ds.class_names

    logger.info("Classes: %s", class_names)

    # Pipeline order matters: cache the *raw* (un-augmented) images, then shuffle
    # and augment on every epoch. Caching the augmented stream (the previous
    # behaviour) froze both the example order and the random augmentation, so the
    # network saw the identical batches every epoch — a likely cause of the
    # "validation better than training" anomaly on this small dataset.
    train_ds = (
        train_ds.cache()
        .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        .map(augmentations, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def run(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPUs detected: %s", [gpu.name if hasattr(gpu, "name") else str(gpu) for gpu in gpus])
    else:
        logger.info("No GPU detected; running on CPU.")

    now_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_height=int(args.img_height / args.image_divider),
        img_width=int(args.img_width / args.image_divider),
        batch_size=args.batch_size,
        seed=args.seed,
        tmpdir=tmpdir,
        val_split=args.val_split,
    )

    if args.tune:
        if not HAS_TUNER:
            raise RuntimeError("KerasTuner is not installed. Install with: pip install keras-tuner")
        directory = os.path.join(tmpdir, f"tune_b{args.batch_size}_{now_string}")
        project_name = f"tuning_img{int(args.img_height / args.image_divider)}x{int(args.img_width / args.image_divider)}"
        tunertype = args.tuner # hyperband is default

        logger.info("Starting hyperparameter tuning with %s...", tunertype)
        logger.info("Tuner directory: %s", directory)
        logger.info("Tuner project:   %s", project_name)
        if tunertype == "bayesianoptimization":
            tuner = kt.BayesianOptimization(
                lambda hp: model_builder(hp, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider)),
                objective="val_accuracy",
                max_trials=args.tune_iterations,
                num_initial_points=5,
                alpha=0.0001, #exploration factor, higher = more exploration
                beta=2.576, # standard value for 99% confidence
                directory=directory,
                project_name=project_name,
                seed=args.seed,
            )
        elif tunertype == "gridsearch":
            tuner = kt.GridSearch(
                lambda hp: model_builder(hp, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider)),
                objective="val_accuracy",
                directory=directory,
                project_name=project_name,
                seed=args.seed,
            )
        elif tunertype == "randomsearch":
            tuner = kt.RandomSearch(
                lambda hp: model_builder(hp, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider)),
                objective="val_accuracy",
                max_trials=args.tune_iterations,
                directory=directory,
                project_name=project_name,
                seed=args.seed,
            )
        else:
            tuner = kt.Hyperband(
                lambda hp: model_builder(hp, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider)),
                objective="val_accuracy",
                max_epochs=args.tune_max_epochs,
                hyperband_iterations=args.tune_iterations,
                factor=3,
                directory=directory,
                project_name=project_name,
                seed=args.seed,
            )

        tuner.search(
            train_ds,
            epochs=args.tune_search_epochs,
            validation_data=val_ds,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience)],
        )
        K.clear_session()
        logger.info("Hyperparameter search finished.")

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(
            "Best HPs — conv1=%s conv1_2=%s conv2=%s conv2_2=%s dense1=%s dense2=%s lr=%s",
            best_hps.get("conv1"),
            best_hps.get("conv1_2"),
            best_hps.get("conv2"),
            best_hps.get("conv2_2"),
            best_hps.get("dense_units_1"),
            best_hps.get("dense_units_2"),
            best_hps.get("learning_rate"),
        )
        model = model_builder(best_hps, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider))
        model.summary(print_fn=lambda l: logger.info(l))
    else:
        model = predefined_model(int(args.img_height / args.image_divider), int(args.img_width / args.image_divider), args.learning_rate)

    log_dir = os.path.join(tmpdir, "logs", "fit", now_string)
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("TensorBoard log dir: %s", log_dir)
    csv_log_path = os.path.join(log_dir, f"training_{now_string}.csv")
    logger.info("CSVLogger path: %s", csv_log_path)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                ckpt_dir, f"{now_string}_epoch{{epoch:02d}}_binary_batch{args.batch_size}.keras"
            ),
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=args.plateau_patience, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="batch"),
        keras.callbacks.CSVLogger(csv_log_path),
        LoggingCallback(),
    ]

    if args.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.es_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    logger.info(
        "Starting training for %d epochs (batch=%d, image=%dx%d)...",
        args.epochs, args.batch_size, int(args.img_height / args.image_divider), int(args.img_width / args.image_divider)
    )
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        initial_epoch=0,
    )
    logger.info("Training finished. Final val_accuracy=%.4f.",
                history.history.get("val_accuracy", [float("nan")])[-1])
    plot_training_curves(history, tmpdir)

    final_model_path = os.path.join(
        tmpdir,
        f"{now_string}_binary_batch{args.batch_size}_div{args.image_divider}"
        f"{'_tuned' if args.tune else ''}.keras"
    )
    logger.info("Saving model to: %s", final_model_path)
    model.save(final_model_path)
    _strip_unsupported_keys(final_model_path)
    logger.info("Model saved successfully.")

    logger.info("Running predictions on validation set for confusion matrix...")
    # Collect labels and predictions in a single pass so they stay aligned even
    # when val_ds is shuffled (it reshuffles on each iteration).
    y_true, y_pred = [], []
    for images, labels in val_ds:
        y_pred.append(model.predict(images, verbose=0).flatten())
        y_true.append(labels.numpy().flatten())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred > 0.5)
    _, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{args.tmpdir}/confusionmatrix.png")
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier with logging and optional tuning.")

    parser.add_argument("--train-dir", required=True, help="Path to training images directory.")
    parser.add_argument("--val-dir", default=None, help="Path to validation images directory (optional).")
    parser.add_argument("--val-split", type=float, default=0.3, help="Validation split (0-1) if --val-dir not provided.")
    parser.add_argument("--img-height", type=int, default=764, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=1092, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size. Values of 16-64 work well; very small (e.g. 4) hurt training on this small dataset, and >=128 starts to degrade.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable EarlyStopping on val_loss.")
    parser.add_argument("--es-patience", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--plateau-patience", type=int, default=5, help="Patience for ReduceLROnPlateau.")

    parser.add_argument("--tune", action="store_true", help="Use KerasTuner to search hyperparameters.")
    parser.add_argument("--tuner", type=str, default="hyperband", help="Which kerastuner method to use. Choose from: hyperband, bayesianoptimization, gridsearch, randomsearch")
    parser.add_argument("--tune-max-epochs", type=int, default=15, help="Hyperband max epochs per bracket.")
    parser.add_argument("--tune-iterations", type=int, default=3, help="Hyperband iterations.")
    parser.add_argument("--tune-search-epochs", type=int, default=5, help="Epochs used in tuner.search().")

    parser.add_argument("--parallel", action="store_true", help="Use MirroredStrategy if GPUs are available.")
    parser.add_argument("--tmpdir", default="./outputs", help="Base output directory for logs/checkpoints/models.")
    parser.add_argument("--image-divider", type=int, default=5, help="Value to divide image width and height by before training.")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")
    parser.add_argument("--log-file", default=None, help="Optional path to a log file.")

    args = parser.parse_args()

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

    if args.parallel:
        logger.info("Initializing MirroredStrategy...")
        strategy = tf.distribute.MirroredStrategy()
        logger.info("Running with MirroredStrategy (replicas=%d).", strategy.num_replicas_in_sync)
        with strategy.scope():
            run(args)
    else:
        logger.info("Running in single device mode.")
        run(args)