# -*- coding: utf-8 -*-
"""统一优化配置管理器。

目标:
- 将 MLA/MTP/MoE 的启用开关与关键参数收敛到一个 profile 中
- 为后续在模型前向/生成路径中的注入提供稳定接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .mla.config import MLAConfig
from .moe.config import MoEConfig
from .mtp.config import MTPConfig


@dataclass(frozen=True)
class OptimizationProfile:
    """封装一次运行中所有优化开关与参数。"""

    mla: Optional[MLAConfig] = None
    mtp: Optional[MTPConfig] = None
    moe: Optional[MoEConfig] = None

    @staticmethod
    def default() -> "OptimizationProfile":
        # 默认全部关闭，确保向后兼容。
        return OptimizationProfile(mla=MLAConfig(False), mtp=MTPConfig(False), moe=MoEConfig(False))
