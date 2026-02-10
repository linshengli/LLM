# -*- coding: utf-8 -*-
import time

import pytest

torch = pytest.importorskip("torch")

from src.optimizers.mtp import MTPConfig, MTPredictor


class _CountingModel:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        # pretend this is a transformer; just return "hidden states"
        return x


def test_mtp_reduces_step_calls_vs_baseline():
    hidden_size = 32
    steps = 20
    k = 4

    model = _CountingModel(hidden_size)
    predictor = MTPredictor(hidden_size=hidden_size, vocab_size=100, cfg=MTPConfig(enabled=True, predict_k=k))

    x = torch.randn(2, 1, hidden_size)

    # baseline: 1 forward per token
    model.calls = 0
    for _ in range(steps):
        _ = model.forward(x)
    baseline_calls = model.calls

    # mtp: 1 forward per k tokens group
    model.calls = 0
    for _ in range((steps + k - 1) // k):
        hidden = model.forward(x)
        _ = predictor(hidden)
    mtp_calls = model.calls

    assert mtp_calls < baseline_calls

