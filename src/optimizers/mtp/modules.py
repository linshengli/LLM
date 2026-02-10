# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MTPHead(nn.Module):
    """一个最小可用的 MTP head。

    输入 hidden_states: (B, S, H)
    输出 logits: (B, S, K, V)
    """

    def __init__(self, hidden_size: int, vocab_size: int, predict_k: int):
        super().__init__()
        if predict_k <= 0:
            raise ValueError("predict_k must be > 0")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.predict_k = predict_k

        self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(predict_k)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"expected hidden_states dim=3, got {hidden_states.dim()}")
        logits = [h(hidden_states) for h in self.heads]  # K * (B,S,V)
        return torch.stack(logits, dim=2)  # (B,S,K,V)


def greedy_tokens_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """从 logits 中取 argmax token id。

    支持:
    - (B, V) -> (B,)
    - (B, S, V) -> (B, S)
    - (B, S, K, V) -> (B, S, K)
    """
    if logits.dim() == 2:
        return torch.argmax(logits, dim=-1)
    if logits.dim() == 3:
        return torch.argmax(logits, dim=-1)
    if logits.dim() == 4:
        return torch.argmax(logits, dim=-1)
    raise ValueError(f"unexpected logits dim: {logits.dim()}")


def weighted_future_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """一个简单的 MTP 训练损失（未来 K token 的加权交叉熵）。

    logits: (B,S,K,V)
    targets: (B,S,K)
    weights: (K,) or None
    """
    import torch.nn.functional as F

    if logits.dim() != 4:
        raise ValueError("logits must be (B,S,K,V)")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must be (B,S,K)")

    b, s, k, v = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * s * k, v),
        targets.reshape(b * s * k),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(b, s, k)

    if weights is None:
        return loss.mean()
    if weights.shape != (k,):
        raise ValueError("weights must be (K,)")
    w = weights.view(1, 1, k).to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp(min=1.0)

