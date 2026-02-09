"""
Minimal decoder-only Transformer model (Qwen-like scaffold) in pure PyTorch.

Goal of this file: a first working version that can instantiate and run a forward
pass to produce logits with shape (batch, seq_len, vocab_size).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenConfig:
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    ffn_mult: int = 4


def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean causal mask of shape (1, 1, seq_len, seq_len).
    True means "allowed to attend".
    """
    m = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
    return m.view(1, 1, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q/k/v: (batch, seq_len, d_model)
        mask: boolean where True means keep, broadcastable to (batch, heads, q_len, k_len)
        """
        bsz, q_len, _ = q.shape
        _, k_len, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, q, d)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, k, d)
        v = v.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, k, d)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, q, k)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        attn = weights @ v  # (b, h, q, d)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.o_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_mult: int) -> None:
        super().__init__()
        inner = ffn_mult * hidden_size
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class DecodeLayer(nn.Module):
    def __init__(self, config: QwenConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config.hidden_size, config.num_heads)
        self.ffn = FeedForward(config.hidden_size, config.ffn_mult)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, mask=attn_mask)
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class QwenModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int, num_layers: int, num_heads: int = 32):
        super().__init__()
        self.config = QwenConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([DecodeLayer(self.config, i) for i in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len), dtype=torch.long
        returns logits: (batch, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.tok_emb(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.pos_emb(position_ids)

        attn_mask = _make_causal_mask(seq_len, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

