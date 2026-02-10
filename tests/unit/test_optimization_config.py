# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.config_manager import OptimizationProfile


def test_default_profile_all_disabled():
    p = OptimizationProfile.default()
    assert p.mla is not None and p.mla.enabled is False
    assert p.mtp is not None and p.mtp.enabled is False
    assert p.moe is not None and p.moe.enabled is False

