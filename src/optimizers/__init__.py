# -*- coding: utf-8 -*-
"""优化模块入口。

该目录用于承载 DeepSeek 相关的可插拔优化实现（MLA/MTP/MoE）以及统一配置/监控。
当前实现以“可配置 + 可测试 + 可渐进集成”为优先级。
"""

from .config_manager import OptimizationProfile

__all__ = ["OptimizationProfile"]

