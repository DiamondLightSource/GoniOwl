import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.models import Sequential
import datetime
import matplotlib.pyplot as plt
import math
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras.backend as K
import keras_tuner as kt

# cv2.setNumThreads(0)
mixed_precision.set_global_policy("mixed_float16")
parallel = False
now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tmpdir = "/dls/tmp/vwg85559"
data_dir_name = os.path.join(tmpdir, "goniopin_auto_24062025_binary")
batch_size = 4
resume_epoch = 47
cont = False  # continue from saved epoch
checkpoint_path = f"{tmpdir}/checkpoints/20250625-161225_epoch47_binary_batch16.h5"
image_divider = 5

os.makedirs(os.path.join(tmpdir, "checkpoints"), exist_ok=True)


def get_first_image_size(data_dir):
    pinon_dir = os.path.join(data_dir_name, "pinon")
    files = [
        f for f in os.listdir(pinon_dir) if os.path.isfile(os.path.join(pinon_dir, f))
    ]
    if not files:
        raise FileNotFoundError("No files found in pinon directory")
    first_file = os.path.join(pinon_dir, files[0])
    img_bytes = tf.io.read_file(first_file)
    img = tf.image.decode_image(img_bytes)
    height = img.shape[0]
    width = img.shape[1]
    height = int(height / image_divider)
    width = int(width / image_divider)
    return height, width


def augmentations(image, label):
    radians = tf.random.uniform([], -3, 3) * (np.pi / 180.0)
    image = tfa.image.rotate(image, radians, fill_mode="reflect")

    input_shape = tf.shape(image)
    width_shift = tf.cast(input_shape[1], tf.float32) * tf.random.uniform(
        [], -0.05, 0.05
    )
    height_shift = tf.cast(input_shape[0], tf.float32) * tf.random.uniform(
        [], -0.05, 0.05
    )
    image = tfa.image.translate(image, [width_shift, height_shift], fill_mode="reflect")

    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = image + (tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=5.0))
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def run():
    print("Using TensorFlow v%s" % tf.__version__)

    cwd = os.getcwd()
    data_dir = data_dir_name
    img_height, img_width = get_first_image_size(data_dir_name)

    seed = random.randint(11111111, 99999999)
    all_labels = []
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="binary",
    )
    print(train_ds.class_names)
    for images, labels in train_ds:
        all_labels.extend(labels.numpy())
    unique, counts = np.unique(all_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    num_samples = sum([len(batch[0]) for batch in train_ds])
    train_ds = train_ds.map(augmentations, num_parallel_calls=tf.data.AUTOTUNE)

    os.makedirs(f"{tmpdir}/sample_images", exist_ok=True)
    for images, labels in train_ds.take(1):
        for i in range(min(20, len(images))):
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
            plt.savefig(
                f"{tmpdir}/sample_images/sample_{i}_label_{int(labels[i].numpy())}.png"
            )
            plt.close()
    all_labels = []
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="binary",
    )
    print(val_ds.class_names)

    for images, labels in val_ds:
        all_labels.extend(labels.numpy())
    unique, counts = np.unique(all_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    os.makedirs(f"{tmpdir}/examples", exist_ok=True)
    num_to_save = 20
    saved = 0

    for images, labels in train_ds:
        for i in range(images.shape[0]):
            if saved >= num_to_save:
                break
            img = images[i].numpy()
            if img.max() <= 1.1:
                img = img * 255
            img = img.astype("uint8")
            plt.imshow(img)
            plt.title(f"Label: {int(labels[i].numpy())}")
            plt.axis("off")
            plt.savefig(
                f"{tmpdir}/examples/img{saved:03d}_label_{int(labels[i].numpy())}.png"
            )
            plt.close()
            saved += 1
        if saved >= num_to_save:
            break
    print(f"Saved {saved} augmented images to ./examples/")

    train_ds_final = train_ds.repeat()
    train_ds_final = train_ds_final.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def model_builder(hp):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
        model.add(layers.Rescaling(1.0 / 255))

        hp_units_1 = hp.Int("conv1", min_value=4, max_value=256, step=4)
        model.add(layers.Conv2D(hp_units_1, 3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        hp_units_2 = hp.Int("conv1_2", min_value=4, max_value=128, step=4)
        model.add(layers.Conv2D(hp_units_2, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        hp_units_3 = hp.Int("conv2", min_value=4, max_value=128, step=4)
        model.add(layers.Conv2D(hp_units_3, (3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        hp_units_4 = hp.Int("conv2_2", min_value=4, max_value=64, step=4)
        model.add(layers.Conv2D(hp_units_4, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("silu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.GlobalAveragePooling2D())

        hp_dense_1 = hp.Int("dense_units_1", min_value=16, max_value=256, step=16)
        model.add(layers.Dense(hp_dense_1, activation="silu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        hp_dense_2 = hp.Int("dense_units_2", min_value=16, max_value=256, step=16)
        model.add(layers.Dense(hp_dense_2, activation="silu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(1, activation="sigmoid", dtype="float32"))

        hp_learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3])

        model.compile(
            keras.optimizers.Adam(hp_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall", "AUC"],
        )

        return model

    def predefined_model():
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
            keras.optimizers.Adam(0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall", "AUC"],
        )

        return model

    tune = False
    if tune:
        directory = f"{tmpdir}/tune_b{batch_size}_imgdiv_{image_divider}_{now_string}"
        project_name = (
            f"{tmpdir}/tuning_b{batch_size}_imgdiv_{image_divider}_{now_string}"
        )
        print(f"KT dir: {directory}")
        print(f"KT proj name: {project_name}")
        tuner = kt.Hyperband(
            model_builder,
            objective="val_accuracy",
            max_epochs=15,
            hyperband_iterations=3,
            factor=3,
            directory=directory,
            project_name=project_name,
        )
        tuner.search(
            train_ds,
            epochs=5,
            validation_data=val_ds,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
            ],
        )
        K.clear_session()

        best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]

        print(
            f"""
        The hyperparameters search is complete. The best hyperparameters are:
        Conv units 1: {best_hps.get("conv1")},
        Conv units 1_2: {best_hps.get("conv1_2")},
        Conv units 2: {best_hps.get("conv2")},
        Conv units 2_2: {best_hps.get("conv2_2")},
        Dense units: {best_hps.get("dense_units_1")},
        Dense units 2: {best_hps.get("dense_units_2")},
        Learning rate: {best_hps.get("learning_rate")}.
        """
        )

        model = model_builder(best_hps)
    else:
        model = predefined_model()

    model.summary()
    log_dir = f"{tmpdir}/logs/fit/" + now_string
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="batch"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{tmpdir}/checkpoints/{now_string}_epoch{{epoch:02d}}_binary_batch{batch_size}.h5",
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, verbose=1
        ),
        tensorboard_callback,
    ]

    model.fit(
        train_ds,
        callbacks=callbacks,
        epochs=100,
        validation_data=val_ds,
        initial_epoch=0,
    )

    model.save(
        f"{tmpdir}/{now_string}_binary_batch{str(batch_size)}_div{str(image_divider)}_tuned.h5"
    )


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    if not parallel:
        run()
    else:
        with strategy.scope():
            run()
