# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class LatentProjector(nn.Module):
    """潜在空间投影器。

    将输入特征从 head_dim 投影到 latent_dim，用于压缩 K/V 表示。
    """

    def __init__(self, head_dim: int, latent_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(head_dim, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim) -> (..., latent_dim)
        if x.size(-1) != self.head_dim:
            raise ValueError(f"expected last dim {self.head_dim}, got {x.size(-1)}")
        return self.proj(x)

