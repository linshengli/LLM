# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.moe import MoEConfig, MOERouter


def test_moe_router_shapes_and_ranges():
    cfg = MoEConfig(enabled=True, num_experts=4, top_k=2)
    router = MOERouter(hidden_size=16, intermediate_size=32, cfg=cfg)

    x = torch.randn(2, 5, 16)
    out = router(x)
    assert out.y.shape == x.shape
    assert out.expert_indices.shape == (2, 5, 2)
    assert out.gate_probs.shape == (2, 5, 2)
    assert out.expert_counts.shape == (cfg.num_experts,)
    assert out.expert_indices.min().item() >= 0
    assert out.expert_indices.max().item() < cfg.num_experts
