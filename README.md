# GoniOwl - Sample Pin Detection for Beamline Collision Prevention

![IMG_0781(1)](https://github.com/user-attachments/assets/8b08de35-6e63-461e-a933-2129fb25e74c)

This repository contains the training and deployment code for a convolutional neural network (CNN) that detects the presence or absence of sample pins on a goniometer, in this case beamline I23 at Diamond Light Source. The model is designed to prevent equipment collisions during automated sample handling by providing real-time classification from beamline camera feeds.

---

## Overview

Synchrotron beamlines require precise mechanical coordination to avoid collisions between components such as goniometers, sample changers, and detectors. Traditional vision-based methods (e.g., histogram analysis) are sensitive to lighting and camera variability, limiting their reliability.

This project introduces a compact CNN classifier trained on real beamline images to robustly detect sample pin presence. Integrated into the beamline control system, it enables real-time inference and fail-safe decision-making, significantly improving safety and throughput.

---

## Features

- **Binary classification**: Detects `sample_on` vs `sample_off` states.
- **Robust to lighting and viewpoint changes** via data augmentation.
- **Real-time inference** on standard beamline control PCs (CPU/GPU).
- **Fail-safe logic** for low-confidence or invalid exposures.
- **Training pipeline** with augmentation, validation split, and logging.
- **Optional hyperparameter tuning** via KerasTuner.
- **TensorBoard, CSV logging, and checkpointing** included.

---

## Performance

- **Accuracy**: 99.73%
- **Precision**: 99.56%
- **Recall**: 99.85%
- **F1 Score**: 99.71%

Evaluated on a held-out test set with manually verified labels. Over a 3-month deployment, the model prevented multiple potential collisions and outperformed legacy histogram-based methods.

Raw training images: 10.5281/zenodo.17047675
Singularity build used for training: 10.5281/zenodo.17047767
