# -*- coding: utf-8 -*-
"""统一 API 层。

该文件提供“如何将优化配置注入到模型/推理路径”的最小抽象。
后续可扩展到:
- 生成侧的 MTP
- 更精细的监控与指标输出
"""

from __future__ import annotations

from typing import Optional

from src.optimizers.config_manager import OptimizationProfile


def resolve_profile(profile: Optional[OptimizationProfile]) -> OptimizationProfile:
    """将 None 归一化为默认 profile（全关闭）。"""
    return profile if profile is not None else OptimizationProfile.default()

