# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MTPConfig:
    """MTP 配置。

    predict_k:
      每步要预测的未来 token 数量。
    """

    enabled: bool = False
    predict_k: int = 4

    def validate(self) -> None:
        if self.predict_k <= 0:
            raise ValueError("predict_k must be > 0")

