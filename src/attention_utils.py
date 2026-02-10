# -*- coding: utf-8 -*-
"""注意力相关通用工具。

将可复用的 RoPE 与 KV 头复制逻辑放在独立模块中，避免优化模块与模型文件产生循环依赖。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """旋转位置编码（Rotary Position Embedding, RoPE）。"""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 1000000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        angles = position_ids.unsqueeze(-1).float() * self.freqs.unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头扩展到与 Q 头数量一致（GQA/MQA 需要）。"""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

