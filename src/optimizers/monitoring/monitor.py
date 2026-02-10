# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from .metrics import PerformanceMetrics


class PerformanceMonitor:
    """简易性能监控器。

    设计目标:
    - 在无 GPU 环境下也可使用（仅计时）
    - 在 CUDA 可用时记录 max_memory_allocated()
    """

    def __init__(self) -> None:
        self.last: Optional[PerformanceMetrics] = None

    @contextmanager
    def measure(self, use_cuda_memory: bool = True) -> Iterator[PerformanceMetrics]:
        if use_cuda_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        metrics = PerformanceMetrics()
        try:
            yield metrics
        finally:
            end = time.perf_counter()
            metrics.elapsed_ms = (end - start) * 1000.0
            if use_cuda_memory and torch.cuda.is_available():
                metrics.peak_memory_bytes = int(torch.cuda.max_memory_allocated())
            self.last = metrics

