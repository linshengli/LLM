# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLAConfig:
    """MLA 配置。

    latent_dim:
      潜在维度（用于压缩 K/V 表示）。应小于 head_dim 才有明显收益。
    """

    enabled: bool = False
    latent_dim: int = 32

    def validate(self, head_dim: int) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.latent_dim > head_dim:
            raise ValueError("latent_dim must be <= head_dim")

