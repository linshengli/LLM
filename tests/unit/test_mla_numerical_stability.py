# -*- coding: utf-8 -*-
import pytest
torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.optimizers.mla import MLAConfig, MLAAttention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mla_no_nans(dtype):
    cfg = ModelConfig(hidden_size=128, num_heads=8, num_kv_heads=2, max_seq_len=64, vocab_size=1000)
    attn = MLAAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        cfg=MLAConfig(enabled=True, latent_dim=cfg.head_dim // 2),
    )

    x = torch.randn(2, 32, cfg.hidden_size, dtype=dtype)
    position_ids = torch.arange(32).unsqueeze(0).repeat(2, 1)
    y = attn(x, position_ids)

    assert torch.isfinite(y.float()).all()


def test_mla_backward_is_finite():
    cfg = ModelConfig(hidden_size=128, num_heads=8, num_kv_heads=2, max_seq_len=64, vocab_size=1000)
    attn = MLAAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        cfg=MLAConfig(enabled=True, latent_dim=cfg.head_dim // 2),
    )

    x = torch.randn(2, 16, cfg.hidden_size, requires_grad=True)
    position_ids = torch.arange(16).unsqueeze(0).repeat(2, 1)
    y = attn(x, position_ids).sum()
    y.backward()
    assert torch.isfinite(x.grad).all()
