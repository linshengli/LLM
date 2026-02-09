"""
Minimal decoder-only Transformer in pure PyTorch.

Includes two embedding variants:
1) Traditional embedding: token embedding + learned absolute position embedding
2) RoPE variant: token embedding only + rotary position applied to Q/K
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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
    pos_emb_type: Literal["learned", "sinusoidal", "rope"] = "learned"
    rope_theta: float = 10000.0


def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Boolean causal mask of shape (1, 1, seq_len, seq_len).
    True means "allowed to attend".
    """
    m = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
    return m.view(1, 1, seq_len, seq_len)


def _position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)


class TraditionalEmbedding(nn.Module):
    """token embedding + learned absolute position embedding"""

    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        pos = _position_ids(bsz, seq_len, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        return x, pos


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-trainable) sinusoidal positional encoding from
    "Attention Is All You Need" (2017).
    """

    def __init__(self, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_size, dtype=torch.float32)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )  # (H/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # (L, H)

    def forward(self, position_ids: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """
        position_ids: (batch, seq_len)
        returns: (batch, seq_len, hidden_size)
        """
        return self.pe[position_ids].to(dtype=dtype)


class SinusoidalEmbedding(nn.Module):
    """token embedding + fixed sinusoidal absolute position encoding"""

    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        pos = _position_ids(bsz, seq_len, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_enc(pos, dtype=self.tok_emb.weight.dtype)
        return x, pos


class RotaryEmbedding(nn.Module):
    """
    RoPE positional embedding. Produces cos/sin tensors used to rotate Q/K.
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch, seq_len)
        returns cos, sin: (batch, 1, seq_len, dim)
        """
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
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: boolean, broadcastable to (batch, heads, q_len, k_len). True = keep.
        position_ids: (batch, seq_len) when using RoPE.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, s, d)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            if position_ids is None:
                raise ValueError("position_ids is required when using RoPE")
            cos, sin = rope(position_ids)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, s, s)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        attn = weights @ v  # (b, h, s, d)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
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

        self.rope = None
        if config.pos_emb_type == "rope":
            self.rope = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.attn_norm(x)
        x = x + self.attn(h, mask=attn_mask, position_ids=position_ids, rope=self.rope)
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class QwenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int = 32,
        pos_emb_type: Literal["learned", "sinusoidal", "rope"] = "learned",
    ) -> None:
        super().__init__()
        self.config = QwenConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            pos_emb_type=pos_emb_type,
        )

        if pos_emb_type == "learned":
            self.embedding = TraditionalEmbedding(vocab_size, hidden_size, max_seq_len)
        elif pos_emb_type == "sinusoidal":
            self.embedding = SinusoidalEmbedding(vocab_size, hidden_size, max_seq_len)
        elif pos_emb_type == "rope":
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        else:
            raise ValueError(f"unknown pos_emb_type: {pos_emb_type}")

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

        pos = _position_ids(bsz, seq_len, device=input_ids.device)

        if self.config.pos_emb_type in ("learned", "sinusoidal"):
            x, _ = self.embedding(input_ids)
        else:
            x = self.embedding(input_ids)

        attn_mask = _make_causal_mask(seq_len, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, position_ids=pos, attn_mask=attn_mask)

        x = self.norm(x)
        return self.lm_head(x)
