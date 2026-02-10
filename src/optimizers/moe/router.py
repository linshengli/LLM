# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .balance import load_balance_loss
from .config import MoEConfig
from .experts import Experts


@dataclass
class MoEOutput:
    y: torch.Tensor
    expert_indices: torch.Tensor  # (B,S,top_k)
    gate_probs: torch.Tensor  # (B,S,top_k)
    balance_loss: torch.Tensor
    expert_counts: torch.Tensor  # (E,)


class MOERouter(nn.Module):
    """基础 MoE 路由器（token-level top-k gating）。"""

    def __init__(self, hidden_size: int, intermediate_size: int, cfg: MoEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k

        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = Experts(
            num_experts=self.num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size
        )

    def forward(self, x: torch.Tensor) -> MoEOutput:
        # x: (B,S,H)
        if x.dim() != 3:
            raise ValueError(f"expected x dim=3, got {x.dim()}")
        b, s, h = x.shape
        logits = self.gate(x)  # (B,S,E)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # (B,S,K)

        # 计算每个 token 的 MoE 输出: sum_k p_k * expert_k(x)
        y = torch.zeros_like(x)
        for k in range(self.top_k):
            idx_k = topk_idx[..., k]  # (B,S)
            p_k = topk_probs[..., k].unsqueeze(-1)  # (B,S,1)
            # 按 expert 分桶执行，避免逐 token 调用模块（这里是基础实现，优先正确性）。
            for e in range(self.num_experts):
                mask = idx_k.eq(e)  # (B,S)
                if not mask.any():
                    continue
                x_e = x[mask]  # (N,H)
                y_e = self.experts.forward_expert(e, x_e)  # (N,H)
                y[mask] += y_e * p_k[mask]

        bal_loss = load_balance_loss(topk_idx, num_experts=self.num_experts)
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts)
        return MoEOutput(
            y=y,
            expert_indices=topk_idx,
            gate_probs=topk_probs,
            balance_loss=bal_loss,
            expert_counts=counts,
        )
