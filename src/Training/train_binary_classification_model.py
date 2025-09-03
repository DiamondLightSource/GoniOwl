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
import argparse
import logging
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend as K

try:
    import keras_tuner as kt
    HAS_TUNER = True
except Exception:
    HAS_TUNER = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
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


def predefined_model(img_height: int, img_width: int, learning_rate: float = 1e-3) -> keras.Model:
    """Predetermined model architecture."""
    logger.info("Building predefined model...")
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(20, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(44, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(24, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(48, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(176, activation="silu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation="silu"))
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
    # Defaults mirror your predefined architecture
    conv1 = hp.Int("conv1", min_value=16, max_value=64, step=8, default=20)
    conv1_2 = hp.Int("conv1_2", min_value=24, max_value=96, step=8, default=44)
    conv2 = hp.Int("conv2", min_value=16, max_value=64, step=8, default=24)
    conv2_2 = hp.Int("conv2_2", min_value=24, max_value=96, step=8, default=48)
    dense1 = hp.Int("dense_units_1", min_value=64, max_value=256, step=32, default=176)
    dense2 = hp.Int("dense_units_2", min_value=32, max_value=128, step=32, default=64)
    learning_rate = hp.Choice("learning_rate", values=[5e-4, 1e-3, 2e-3], default=1e-3)

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.Conv2D(conv1, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(conv1_2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(conv2, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.Conv2D(conv2_2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("silu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(dense1, activation="silu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(dense2, activation="silu"))
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
    val_dir: str | None = None,
    val_split: float = 0.0,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Create train/val datasets from directories."""
    if val_dir:
        logger.info("Using explicit validation directory: %s", val_dir)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="binary",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
        )
        class_names = train_ds.class_names
    else:
        if not (0.0 < val_split < 1.0):
            raise ValueError("When --val-dir is not provided, you must set --val-split in (0,1).")
        logger.info("Using train/validation split from: %s (val_split=%.2f)", train_dir, val_split)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            validation_split=val_split,
            subset="training",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            validation_split=val_split,
            subset="validation",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        class_names = train_ds.class_names

    logger.info("Classes: %s", class_names)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
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
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
    )

    if args.tune:
        if not HAS_TUNER:
            raise RuntimeError("KerasTuner is not installed. Install with: pip install keras-tuner")
        directory = os.path.join(tmpdir, f"tune_b{args.batch_size}_{now_string}")
        project_name = f"tuning_img{args.img_height}x{args.img_width}"

        logger.info("Starting hyperparameter tuning with Hyperband...")
        logger.info("Tuner directory: %s", directory)
        logger.info("Tuner project:   %s", project_name)

        tuner = kt.Hyperband(
            lambda hp: model_builder(hp, args.img_height, args.img_width),
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
        model = model_builder(best_hps, args.img_height, args.img_width)
        model.summary(print_fn=lambda l: logger.info(l))
    else:
        model = predefined_model(args.img_height, args.img_width, args.learning_rate)

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
                ckpt_dir, f"{now_string}_epoch{{epoch:02d}}_binary_batch{args.batch_size}.h5"
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
        args.epochs, args.batch_size, args.img_height, args.img_width
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

    final_model_path = os.path.join(
        tmpdir,
        f"{now_string}_binary_batch{args.batch_size}_div{args.image_divider}"
        f"{'_tuned' if args.tune else ''}.h5"
    )
    logger.info("Saving model to: %s", final_model_path)
    model.save(final_model_path)
    logger.info("Model saved successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier with logging and optional tuning.")

    parser.add_argument("--train-dir", required=True, help="Path to training images directory.")
    parser.add_argument("--val-dir", default=None, help="Path to validation images directory (optional).")
    parser.add_argument("--val-split", type=float, default=0.0, help="Validation split (0-1) if --val-dir not provided.")
    parser.add_argument("--img-height", type=int, default=224, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=224, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable EarlyStopping on val_loss.")
    parser.add_argument("--es-patience", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--plateau-patience", type=int, default=5, help="Patience for ReduceLROnPlateau.")

    parser.add_argument("--tune", action="store_true", help="Use KerasTuner Hyperband to search hyperparameters.")
    parser.add_argument("--tune-max-epochs", type=int, default=15, help="Hyperband max epochs per bracket.")
    parser.add_argument("--tune-iterations", type=int, default=3, help="Hyperband iterations.")
    parser.add_argument("--tune-search-epochs", type=int, default=5, help="Epochs used in tuner.search().")

    parser.add_argument("--parallel", action="store_true", help="Use MirroredStrategy if GPUs are available.")
    parser.add_argument("--tmpdir", default="./outputs", help="Base output directory for logs/checkpoints/models.")
    parser.add_argument("--image-divider", type=int, default=1, help="Parameter to match previous artifacts.")

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