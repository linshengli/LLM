# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.monitoring import PerformanceMonitor


def test_performance_monitor_records_time():
    mon = PerformanceMonitor()
    with mon.measure(use_cuda_memory=False) as m:
        _ = 1 + 1
    assert mon.last is m
    assert m.elapsed_ms is not None
    assert m.elapsed_ms >= 0.0

