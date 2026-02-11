from __future__ import annotations

from typing import Iterable, List

from torch import nn


class TabularMLP(nn.Module):
    """
    Simple MLP for tabular data with two or more hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ImageMLP(nn.Module):
    """
    MLP applied to flattened image features.
    This is intentionally suboptimal for image data to highlight inductive bias differences.
    """

    def __init__(
        self,
        input_shape,
        hidden_sizes: Iterable[int],
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        input_dim = c * h * w

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hdim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = hdim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

