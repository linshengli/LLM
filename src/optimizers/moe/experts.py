# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class myExpertFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, fn: str = "silu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        if fn == "silu":
            self.att_fn = F.silu
        elif fn == "gelu":
            self.att_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {fn}")
        
    def forward(self, x: torch.Tensor):
        return self.fc2(self.att_fn(self.fn1(x)))

class myExpertsBlock(nn.Module):
    def __init__(self, hidden_size: int, num_expert: int, top_k: int, fn: str = "silu"):
        super().__init__()
        self.num_expert = num_expert
        self.router = nn.Linear(hidden_size, num_expert)
        self.experts = nn.ModuleList([myExpertFFN(hidden_size, hidden_size, fn) for _ in range(num_expert)])
    
    def forward_expert(self, expert_id: int, x: torch.Tensor):
        return self.experts[expert_id](x)
    
    def forward(self, x: torch.Tensor):
        logits = self.router(x) # B S H

        probs = F.softmax(logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)

        y = torch.zeros_like(x) # B S k * expert_H

        # for each expert, run with x_k, each expert have its own x_k to eval.
        # TODO
        for idx in range(self.top_k):
            y[..., idx] = self.forward_expert(top_k_indices[..., idx], x)
        
        return y
        



class ExpertFFN(nn.Module):
    """一个简单的 FFN 专家。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class Experts(nn.Module):
    """专家集合。"""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size=hidden_size, intermediate_size=intermediate_size) for _ in range(num_experts)]
        )

    def forward_expert(self, expert_id: int, x: torch.Tensor) -> torch.Tensor:
        return self.experts[expert_id](x)

