from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .mlp import TabularMLP, ImageMLP
from .cnn import SimpleCNN
from .attention import TabularTransformer, VisionTransformerTiny


def create_model(
    architecture: str,
    input_dim: Any,
    num_classes: int,
    dataset_type: str,
    model_cfg: Dict[str, Any],
) -> nn.Module:
    """
    Create a model given an architecture name and dataset type.

    architecture: "mlp" | "cnn" | "attention"
    dataset_type: "adult" | "cifar10" | "pcam"
    """
    arch = architecture.lower()
    ds = dataset_type.lower()

    if arch == "mlp":
        hidden_sizes = model_cfg.get("hidden_sizes", [256, 128])
        dropout = float(model_cfg.get("dropout", 0.5))
        if isinstance(input_dim, int):
            return TabularMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=num_classes, dropout=dropout)
        else:
            # For image datasets, flatten and use an MLP classifier
            return ImageMLP(input_shape=input_dim, hidden_sizes=hidden_sizes, num_classes=num_classes, dropout=dropout)

    if arch == "cnn":
        return SimpleCNN(
            input_shape=input_dim,
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )

    if arch == "attention":
        if ds == "adult":
            # Transformer-style encoder on tabular data
            d_model = int(model_cfg.get("d_model", 128))
            n_heads = int(model_cfg.get("num_heads", 4))
            num_layers = int(model_cfg.get("num_layers", 2))
            dropout = float(model_cfg.get("dropout", 0.1))
            return TabularTransformer(
                input_dim=input_dim,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            # Vision Transformer style model for images
            patch_size = int(model_cfg.get("patch_size", 8))
            d_model = int(model_cfg.get("d_model", 128))
            n_heads = int(model_cfg.get("num_heads", 4))
            num_layers = int(model_cfg.get("num_layers", 4))
            mlp_dim = int(model_cfg.get("mlp_dim", 256))
            dropout = float(model_cfg.get("dropout", 0.1))
            return VisionTransformerTiny(
                image_shape=input_dim,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=d_model,
                depth=num_layers,
                heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )

    raise ValueError(f"Unknown architecture: {architecture}")

