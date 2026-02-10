# -*- coding: utf-8 -*-
import pytest

torch = pytest.importorskip("torch")

from src.optimizers.moe.balance import load_balance_loss


def test_balance_loss_is_small_for_uniform():
    idx = torch.tensor([[0, 1, 2, 3] * 10])  # fairly uniform
    idx = idx.unsqueeze(-1)  # (..., top_k=1)
    loss = load_balance_loss(idx, num_experts=4)
    assert loss.item() < 0.01


def test_balance_loss_is_large_for_skewed():
    idx = torch.zeros((100,), dtype=torch.long).unsqueeze(-1)
    loss = load_balance_loss(idx, num_experts=4)
    assert loss.item() > 0.05

