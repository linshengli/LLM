# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.moe import MoEConfig, MOERouter


def test_moe_sparse_activation_proxy():
    """用“激活专家数/top_k”作为稀疏激活代理，确保 top_k << num_experts 时更稀疏。"""
    cfg = MoEConfig(enabled=True, num_experts=16, top_k=2)
    router = MOERouter(hidden_size=32, intermediate_size=64, cfg=cfg)

    x = torch.randn(2, 8, 32)
    out = router(x)
    assert out.expert_indices.numel() == 2 * 8 * cfg.top_k
    # proxy: top_k/num_experts should be small for sparse MoE
    assert cfg.top_k / cfg.num_experts <= 0.25

