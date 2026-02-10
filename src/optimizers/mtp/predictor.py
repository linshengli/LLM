# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import MTPConfig
from .modules import MTPHead, greedy_tokens_from_logits, weighted_future_loss


@dataclass
class MTPOutput:
    tokens: torch.Tensor  # (B,S,K)
    logits: torch.Tensor  # (B,S,K,V)


class MTPredictor(nn.Module):
    """MTP 预测器最小实现。

    当前版本用于提供:
    - 结构化输出（logits + greedy tokens）
    - 训练损失接口（weighted_future_loss）
    """

    def __init__(self, hidden_size: int, vocab_size: int, cfg: MTPConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.head = MTPHead(hidden_size=hidden_size, vocab_size=vocab_size, predict_k=cfg.predict_k)

    def forward(self, hidden_states: torch.Tensor) -> MTPOutput:
        logits = self.head(hidden_states)  # (B,S,K,V)
        tokens = greedy_tokens_from_logits(logits)  # (B,S,K)
        return MTPOutput(tokens=tokens, logits=logits)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        return weighted_future_loss(logits, targets, weights=weights, ignore_index=ignore_index)

