# -*- coding: utf-8 -*-
from __future__ import annotations

import torch


def load_balance_loss(expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """一个简化的负载均衡损失。

    expert_indices: (..., top_k) 取值范围 [0, num_experts)
    返回:
      越均衡 loss 越小；极端倾斜时更大。
    """
    if expert_indices.numel() == 0:
        return torch.tensor(0.0, device=expert_indices.device)

    flat = expert_indices.reshape(-1)
    counts = torch.bincount(flat, minlength=num_experts).float()
    probs = counts / counts.sum().clamp(min=1.0)

    # 与均匀分布的 L2 距离
    uniform = torch.full_like(probs, 1.0 / num_experts)
    return torch.mean((probs - uniform) ** 2)

