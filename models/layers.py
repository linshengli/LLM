from abc import abstractmethod, ABC

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    num_heads: int
    num_layers: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: torch.float = 1e-9):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x = torch.pow(x, 2)
        mx = torch.mean(x)
        x =  x * torch.rsqrt(mx + self.eps)
        return self.weights(x)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, up_size):
        super().__init__()
        # SwiGLU: silu(gate(x)) * up(x), then project back down.
        self.up = nn.Linear(hidden_size, up_size, bias=False)
        self.gate = nn.Linear(hidden_size, up_size, bias=False)
        self.down = nn.Linear(up_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(f"position_ids must be 2D (batch, seq_len), got {tuple(position_ids.shape)}")
        bsz, seq_len = position_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        freqs = torch.einsum("bs,d->bsd", position_ids.to(self.inv_freq.dtype), self.inv_freq)  # (b, s, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (b, s, dim)
        cos = emb.cos().unsqueeze(1)  # (b, 1, s, dim)
        sin = emb.sin().unsqueeze(1)  # (b, 1, s, dim)
        return cos, sin

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (b, h, s, d), cos/sin: (b, 1, s, d)
    return (x * cos) + (_rotate_half(x) * sin)
