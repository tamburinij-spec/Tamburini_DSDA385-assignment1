from __future__ import annotations

import math

import torch
from torch import nn


class TabularTransformer(nn.Module):
    """
    Simple Transformer-style encoder for tabular data.
    We treat each feature as a token with learned positional embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.token_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Embedding(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, features)
        bsz, feat_dim = x.shape
        x = x.view(bsz, feat_dim, 1)
        tokens = self.token_proj(x)

        positions = (
            nn.functional.pad(
                x.new_zeros((feat_dim,), dtype=x.dtype),
                (0, 0),
                "constant",
                0,
            )
        )
        pos_ids = (
            nn.functional.pad(
                x.new_zeros((feat_dim,), dtype=x.dtype),
                (0, 0),
                "constant",
                0,
            )
        )
        # simpler: positions are just 0..feat_dim-1
        pos_ids = (
            x.new_tensor(range(feat_dim), dtype=torch.long)  # type: ignore[name-defined]
        )
        pos_emb = self.pos_embedding(pos_ids)[None, :, :]
        tokens = tokens + pos_emb

        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.cls_head(pooled)


class PatchEmbedding(nn.Module):
    def __init__(self, img_channels: int, patch_size: int, emb_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            img_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)  # (B, emb_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, emb_dim)
        return x


class VisionTransformerTiny(nn.Module):
    """
    Very small ViT-style model appropriate for CIFAR-10 and PCam.
    """

    def __init__(
        self,
        image_shape,
        patch_size: int,
        num_classes: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c, h, w = image_shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image dims must be divisible by patch_size"

        self.patch_embed = PatchEmbedding(c, patch_size, dim)
        num_patches = (h // patch_size) * (w // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # type: ignore[name-defined]
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim)  # type: ignore[name-defined]
        )
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # type: ignore[arg-type]

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # type: ignore[name-defined]
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        cls = x[:, 0]
        return self.mlp_head(cls)

