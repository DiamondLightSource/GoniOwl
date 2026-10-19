"""GoniOwl CNN in PyTorch, plus checkpoint save/load helpers.

The layer stack and initialisation mirror the original Keras model so that
converted Keras weights give identical predictions and new training runs
behave the same way:

- Inputs are float NCHW images in 0-255; the model divides by 255 itself.
- BatchNorm uses Keras' defaults (eps=1e-3, momentum 0.99 == PyTorch 0.01).
- Conv/Linear weights use Glorot-uniform init, biases start at zero.
- ``forward`` returns logits; use ``predict_proba`` for P(pin on).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from .preprocess import CLASS_NAMES

CHECKPOINT_FORMAT = "goniowl-torch-v1"

BN_EPS = 1e-3
BN_MOMENTUM = 0.01


@dataclass(frozen=True)
class Hyperparameters:
    """Architecture hyperparameters.

    Defaults are the Hyperband result for the batch-16 / seed-0 manuscript
    model (~51,313 Keras params). ``use_bias`` is only True for models
    converted from the 2025 Keras architecture.
    """

    conv1: int = 16
    conv1_2: int = 48
    conv2: int = 16
    conv2_2: int = 96
    dense_units_1: int = 96
    dense_units_2: int = 128
    use_bias: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Hyperparameters":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _conv_block(in_ch: int, out_ch: int, padding: int, bias: bool) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, 3, padding=padding, bias=bias),
        nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.SiLU(),
    ]


def _dense_block(in_f: int, out_f: int, bias: bool) -> list[nn.Module]:
    # Keras order: Dense(activation=silu) -> BatchNorm -> Dropout
    return [
        nn.Linear(in_f, out_f, bias=bias),
        nn.SiLU(),
        nn.BatchNorm1d(out_f, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.Dropout(0.4),
    ]


class GoniOwlNet(nn.Module):
    def __init__(self, hp: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.hp = hp
        b = hp.use_bias
        self.features = nn.Sequential(
            *_conv_block(3, hp.conv1, padding=1, bias=b),
            *_conv_block(hp.conv1, hp.conv1_2, padding=0, bias=b),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            *_conv_block(hp.conv1_2, hp.conv2, padding=1, bias=b),
            *_conv_block(hp.conv2, hp.conv2_2, padding=0, bias=b),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            *_dense_block(hp.conv2_2, hp.dense_units_1, bias=b),
            *_dense_block(hp.dense_units_1, hp.dense_units_2, bias=b),
            nn.Linear(hp.dense_units_2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x / 255.0)
        x = x.mean(dim=(2, 3))  # GlobalAveragePooling2D
        return self.classifier(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Parameter count as Keras reports it (includes BatchNorm running stats)."""
    params = sum(p.numel() for p in model.parameters())
    stats = sum(b.numel() for name, b in model.named_buffers() if not name.endswith("num_batches_tracked"))
    return params + stats


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap(model: nn.Module) -> "GoniOwlNet":
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    img_size: tuple[int, int],
    class_names: Optional[list[str]] = None,
    **metadata: Any,
) -> None:
    """Save weights plus everything needed to rebuild and feed the model."""
    net = unwrap(model)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "hparams": asdict(net.hp),
            "img_height": int(img_size[0]),
            "img_width": int(img_size[1]),
            "class_names": list(class_names or CLASS_NAMES),
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> tuple[GoniOwlNet, dict]:
    """Load a checkpoint written by ``save_checkpoint``; returns (model in eval mode, info)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a GoniOwl PyTorch checkpoint")
    model = GoniOwlNet(Hyperparameters.from_dict(ckpt["hparams"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    info = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, info


@torch.no_grad()
def predict_proba(model: nn.Module, images: Union[np.ndarray, torch.Tensor], batch_size: int = 64) -> np.ndarray:
    """P(pin on) for uint8/float images shaped (N, H, W, 3) or (H, W, 3)."""
    images = torch.as_tensor(np.asarray(images) if not isinstance(images, torch.Tensor) else images)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device).permute(0, 3, 1, 2).float()
        out.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(out).numpy() if out else np.empty(0, dtype=np.float32)
