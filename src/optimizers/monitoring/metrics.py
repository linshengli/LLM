# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    """通用性能指标载体（可按模块扩展）。"""

    peak_memory_bytes: Optional[int] = None
    elapsed_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

