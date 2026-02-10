# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.mtp import MTPConfig, MTPredictor


def test_mtp_greedy_tokens_match_argmax():
    cfg = MTPConfig(enabled=True, predict_k=2)
    predictor = MTPredictor(hidden_size=8, vocab_size=10, cfg=cfg)

    # 手工构造 hidden，使得 head 输出可预测: 用全零输入 + 直接改权重，使 token 0 永远最大
    with torch.no_grad():
        for h in predictor.head.heads:
            h.weight.zero_()
            h.weight[0, 0] = 1.0

    hidden = torch.zeros(1, 3, 8)
    out = predictor(hidden)
    assert (out.tokens == 0).all()

