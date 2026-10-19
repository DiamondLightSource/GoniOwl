#!/usr/bin/env python3
"""
Convert a trained Keras GoniOwl model into a PyTorch checkpoint.

Reads the weights straight out of the HDF5 data with h5py, so TensorFlow does
not need to be installed. Handles both the legacy Keras 2 HDF5 format (``.h5``,
including HDF5 files saved with a ``.keras`` extension) and Keras 3 ``.keras``
zip archives.

Example:
  python convert_keras_to_torch.py \
    --model ../GoniOwl/20260626_140916_epoch30_binary_batch16.keras \
    --output ../GoniOwl/20260626_140916_epoch30_binary_batch16.pt
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from GoniOwl.model import GoniOwlNet, Hyperparameters, save_checkpoint

PARAM_LAYERS = ("Conv2D", "BatchNormalization", "Dense")
SILU_NAMES = ("silu", "swish")


def _read_legacy_h5(path: Path) -> tuple[dict, dict[str, list[np.ndarray]], str]:
    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        weights = {}
        root = f["model_weights"]
        for name in root:
            group = root[name]
            names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs.get("weight_names", [])]
            weights[name] = [np.asarray(group[n]) for n in names]
        version = f.attrs.get("keras_version", "unknown")
    return config, weights, version.decode() if isinstance(version, bytes) else str(version)


def _read_keras_v3(path: Path) -> tuple[dict, dict[str, list[np.ndarray]], str]:
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
        version = json.loads(z.read("metadata.json")).get("keras_version", "unknown")
        weights = {}
        with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as f:
            for name in f["layers"]:
                vars_group = f["layers"][name].get("vars")
                if vars_group is None:
                    continue
                weights[name] = [np.asarray(vars_group[str(i)]) for i in range(len(vars_group))]
    return config, weights, version


def read_keras_model(path: Path) -> tuple[list[dict], dict[str, list[np.ndarray]], str]:
    """Return (layer configs in order, weights by layer name, keras version)."""
    if zipfile.is_zipfile(path):
        config, weights, version = _read_keras_v3(path)
    elif h5py.is_hdf5(path):
        config, weights, version = _read_legacy_h5(path)
    else:
        raise ValueError(f"Unrecognised Keras model file: {path}")
    if config.get("class_name") != "Sequential":
        raise ValueError(f"Expected a Sequential model, got {config.get('class_name')}")
    return config["config"]["layers"], weights, version


def infer_architecture(layers: list[dict]) -> tuple[Hyperparameters, tuple[int, int]]:
    """Work out the GoniOwlNet hyperparameters and input size from a Keras config."""
    convs = [l["config"] for l in layers if l["class_name"] == "Conv2D"]
    denses = [l["config"] for l in layers if l["class_name"] == "Dense"]
    if len(convs) != 4 or len(denses) != 3:
        raise ValueError(f"Expected 4 Conv2D and 3 Dense layers, found {len(convs)} and {len(denses)}")
    for l in layers:
        cfg = l["config"]
        if l["class_name"] == "Activation" and cfg["activation"] not in SILU_NAMES:
            raise ValueError(f"Unsupported activation {cfg['activation']}")
    for d in denses[:2]:
        if d["activation"] not in SILU_NAMES:
            raise ValueError(f"Unsupported dense activation {d['activation']}")
    if denses[2]["activation"] != "sigmoid" or denses[2]["units"] != 1:
        raise ValueError("Final layer must be Dense(1, activation='sigmoid')")

    biases = {c["use_bias"] for c in convs + denses[:2]}
    if len(biases) != 1:
        raise ValueError("Mixed use_bias settings are not supported")

    hp = Hyperparameters(
        conv1=convs[0]["filters"],
        conv1_2=convs[1]["filters"],
        conv2=convs[2]["filters"],
        conv2_2=convs[3]["filters"],
        dense_units_1=denses[0]["units"],
        dense_units_2=denses[1]["units"],
        use_bias=biases.pop(),
    )

    shape = None
    for l in layers:
        if l["class_name"] == "InputLayer":
            cfg = l["config"]
            shape = cfg.get("batch_shape") or cfg.get("batch_input_shape")
    if shape is None:
        raise ValueError("Could not find the input shape in the Keras config")
    return hp, (int(shape[1]), int(shape[2]))


def load_weights_into(model: GoniOwlNet, layers: list[dict], weights: dict[str, list[np.ndarray]]) -> None:
    keras_layers = [l for l in layers if l["class_name"] in PARAM_LAYERS]
    torch_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d))]
    if len(keras_layers) != len(torch_layers):
        raise ValueError(f"Layer count mismatch: Keras {len(keras_layers)} vs PyTorch {len(torch_layers)}")

    def assign(param: torch.Tensor, value: np.ndarray) -> None:
        value = torch.from_numpy(np.ascontiguousarray(value)).to(param.dtype)
        if value.shape != param.shape:
            raise ValueError(f"Shape mismatch: {tuple(value.shape)} vs {tuple(param.shape)}")
        param.copy_(value)

    with torch.no_grad():
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            name = k_layer["config"]["name"]
            w = weights[name]
            if isinstance(t_layer, nn.Conv2d):
                assign(t_layer.weight, w[0].transpose(3, 2, 0, 1))  # (kh, kw, in, out) -> (out, in, kh, kw)
                if t_layer.bias is not None:
                    assign(t_layer.bias, w[1])
            elif isinstance(t_layer, nn.Linear):
                assign(t_layer.weight, w[0].T)  # (in, out) -> (out, in)
                if t_layer.bias is not None:
                    assign(t_layer.bias, w[1])
            else:
                gamma, beta, mean, var = w
                assign(t_layer.weight, gamma)
                assign(t_layer.bias, beta)
                assign(t_layer.running_mean, mean)
                assign(t_layer.running_var, var)


def convert(model_path: Path, output_path: Path) -> GoniOwlNet:
    layers, weights, version = read_keras_model(model_path)
    hp, img_size = infer_architecture(layers)
    model = GoniOwlNet(hp)
    load_weights_into(model, layers, weights)
    model.eval()
    save_checkpoint(
        output_path,
        model,
        img_size,
        source=str(model_path.name),
        converted_from_keras=version,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Keras GoniOwl model to a PyTorch checkpoint.")
    parser.add_argument("--model", "-m", required=True, help="Path to the Keras .h5 or .keras model.")
    parser.add_argument("--output", "-o", default=None, help="Output .pt path (default: <input>.pt).")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1
    output_path = Path(args.output) if args.output else model_path.with_suffix(".pt")

    model = convert(model_path, output_path)
    print(f"Converted {model_path} -> {output_path}")
    print(f"Architecture: {model.hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
