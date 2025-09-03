# GoniOwl - Sample Pin Detection for Beamline Collision Prevention

![IMG_0781(1)](https://github.com/user-attachments/assets/8b08de35-6e63-461e-a933-2129fb25e74c)

This repository contains the training and deployment code for a convolutional neural network (CNN) that detects the presence or absence of sample pins on a goniometer, in this case beamline I23 at Diamond Light Source. The model is designed to prevent equipment collisions during automated sample handling by providing real-time classification from beamline camera feeds.

---
# Usage

## Training

- Create python venv `python -m venv .venv`
- Activate python venv `source .venv/bin/activate`
- Install required modules `pip install tqdm opencv-python`
- Alter the script import_crop_save.py depending on the directory structure of your input images.
- Run `python src/Training/import_crop_save.py --snapshots /path/to/snapshots --imgdir /path/to/save/cropped/images`
- Check images in output folders manually
- Generate or download Singularity image
- Set tfimage environment variable `tfimage=/path/to/tensorflow_2.8.2-gpu-jupyter.sif`
- Run training `singularity exec --nv --home $PWD $tfimage python src/Training/train_binary_classification_model.py --train_dir /path/to/save/cropped/images`

## Inference

- Copy model from training into src/GoniOwl/
- Alter GoniOwl_controller.py to target model of choice and set log output directory.
- From src/GoniOwl/ run `pipenv run python -m GoniOwl`. This will launch the ioc from this terminal, so it is advisable to run this via screen/tmux.

## Testing

- To check that GoniOwl is running and serving PVs, try `caget GONIOWL-TEST:WHOAMI` -> `GONIOWL-TEST:WHOAMI            GoniOwl Python IOC`
- To check that inference is working, try:
`caput GONIOWL-TEST:INFER 1` -> ```Old : GONIOWL-TEST:INFER             Disable
New : GONIOWL-TEST:INFER             Enable
```
then
`caget GONIOWL-TEST:GONIOSTATUS` -> `GONIOWL-TEST:GONIOSTATUS       0`

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
- **Optional hyperparameter tuning** via KerasTuner.
- **TensorBoard, CSV logging, and checkpointing** included.

---

# Performance

- **Accuracy**: 99.73%
- **Precision**: 99.56%
- **Recall**: 99.85%
- **F1 Score**: 99.71%

Evaluated on a held-out test set with manually verified labels. Over a 3-month deployment, the model prevented multiple potential collisions and outperformed legacy histogram-based methods.

Raw training images: 10.5281/zenodo.17047675

Singularity build used for training: 10.5281/zenodo.17047767
