# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    """MoE 配置（基础版）。"""

    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2

    def validate(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")

