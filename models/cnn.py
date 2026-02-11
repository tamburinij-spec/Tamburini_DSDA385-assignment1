from __future__ import annotations

from torch import nn


class SimpleCNN(nn.Module):
    """
    Small CNN with at least two conv layers and a classifier head.
    Works for CIFAR-10 and PCam sized images.
    """

    def __init__(
        self,
        input_shape,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # We support two cases:
        # - Image data: input_shape is (C, H, W) -> use 2D conv/pooling
        # - Tabular data: input_shape is an int feature_dim -> use 1D conv/pooling
        if isinstance(input_shape, int):
            self.is_tabular = True

            in_channels = 1
            self.features_1d = nn.Sequential(
                nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
            # Reduce sequence length to a fixed size
            self.pool_1d = nn.AdaptiveAvgPool1d(4)
            feat_dim = base_channels * 4 * 4
        else:
            self.is_tabular = False

            in_channels, _, _ = input_shape

            self.features_2d = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

            # Adaptive pooling to handle both 32x32 and 96x96
            self.pool_2d = nn.AdaptiveAvgPool2d((4, 4))
            feat_dim = base_channels * 4 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self.is_tabular:
            # Tabular: x is (N, F) -> (N, 1, F)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.features_1d(x)
            x = self.pool_1d(x)
        else:
            # Images: expect (N, C, H, W)
            x = self.features_2d(x)
            x = self.pool_2d(x)

        x = self.classifier(x)
        return x

