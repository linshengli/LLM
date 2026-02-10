# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.model import StudentModel
from src.optimizers.config_manager import OptimizationProfile
from src.optimizers.mla import MLAConfig
from src.optimizers.moe import MoEConfig


def test_student_model_accepts_optimization_profile():
    cfg = ModelConfig(hidden_size=64, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000)
    profile = OptimizationProfile(
        mla=MLAConfig(enabled=True, latent_dim=cfg.head_dim // 2),
        mtp=None,
        moe=MoEConfig(enabled=True, num_experts=4, top_k=2),
    )
    model = StudentModel(cfg, optimization=profile)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, cfg.vocab_size)

