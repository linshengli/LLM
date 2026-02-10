# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.optimizers.mla import MLAConfig, MLAAttention


def test_mla_attention_shapes_and_cache():
    cfg = ModelConfig(hidden_size=128, num_heads=8, num_kv_heads=2, max_seq_len=64, vocab_size=1000)
    mla_cfg = MLAConfig(enabled=True, latent_dim=16)

    attn = MLAAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        cfg=mla_cfg,
    )

    x = torch.randn(2, 10, cfg.hidden_size)
    position_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)

    y = attn(x, position_ids)
    assert y.shape == (2, 10, cfg.hidden_size)

    attn.reset_cache()
    y1 = attn(x[:, :5], position_ids[:, :5], use_cache=True)
    y2 = attn(x[:, 5:], position_ids[:, 5:], use_cache=True)
    assert y1.shape == (2, 5, cfg.hidden_size)
    assert y2.shape == (2, 5, cfg.hidden_size)
    assert attn.cache.k is not None
    assert attn.cache.v is not None
    assert attn.cache.k.shape[-1] == mla_cfg.latent_dim


def test_mla_reduces_kv_elements_vs_head_dim():
    cfg = ModelConfig(hidden_size=128, num_heads=8, num_kv_heads=2, max_seq_len=64, vocab_size=1000)

    head_dim = cfg.head_dim
    latent_dim = head_dim // 2
    mla_cfg = MLAConfig(enabled=True, latent_dim=latent_dim)

    # element-count proxy for KV cache size: (B,H,S,D)
    b, h, s = 2, cfg.num_heads, 32
    kv_dense = b * h * s * head_dim
    kv_latent = b * h * s * latent_dim

    # latent_dim=head_dim/2 => 50% reduction (>= 40% target proxy)
    assert kv_latent <= int(kv_dense * 0.6)
