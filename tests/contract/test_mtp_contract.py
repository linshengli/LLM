# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.mtp import MTPConfig, MTPredictor


def test_mtp_predictor_output_shapes():
    cfg = MTPConfig(enabled=True, predict_k=4)
    predictor = MTPredictor(hidden_size=32, vocab_size=100, cfg=cfg)

    hidden = torch.randn(2, 8, 32)
    out = predictor(hidden)
    assert out.logits.shape == (2, 8, 4, 100)
    assert out.tokens.shape == (2, 8, 4)


