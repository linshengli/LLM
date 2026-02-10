# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention_utils import RotaryEmbedding, repeat_kv

from .cache import KVCache
from .config import MLAConfig
from .projection import LatentProjector


logger = logging.getLogger(__name__)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention 的简化实现。

    说明:
    - 该实现以“潜在维度压缩 K/V”为核心，使 K/V cache 的元素数随 latent_dim 线性降低。
    - 这里对 Q/K/V 都在 latent 空间计算注意力（Q 同样投影到 latent_dim），以保持维度一致。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        cfg: MLAConfig,
    ):
        super().__init__()
        cfg.validate(head_dim=head_dim)

        self.cfg = cfg
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.latent_dim = cfg.latent_dim
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        self.q_latent = LatentProjector(head_dim=head_dim, latent_dim=self.latent_dim)
        self.k_latent = LatentProjector(head_dim=head_dim, latent_dim=self.latent_dim)
        self.v_latent = LatentProjector(head_dim=head_dim, latent_dim=self.latent_dim)

        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        self.cache = KVCache()

    def reset_cache(self) -> None:
        self.cache = KVCache()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x dim=3, got {x.dim()}")
        if position_ids.dim() != 2:
            raise ValueError(f"expected position_ids dim=2, got {position_ids.dim()}")
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # latent projection
        q_l = self.q_latent(q)
        k_l = self.k_latent(k)
        v_l = self.v_latent(v)

        # repeat kv heads to match q heads
        k_l = repeat_kv(k_l, self.n_rep)
        v_l = repeat_kv(v_l, self.n_rep)

        if use_cache:
            # cache is stored in latent space (B,H,S,L)
            self.cache.append(k_l, v_l)
            k_l_all = self.cache.k
            v_l_all = self.cache.v
            logger.debug(
                "MLA cache appended: k=%s v=%s", tuple(k_l_all.shape), tuple(v_l_all.shape)
            )
        else:
            k_l_all = k_l
            v_l_all = v_l

        scale = math.sqrt(self.latent_dim)
        attn = torch.matmul(q_l, k_l_all.transpose(-2, -1)) / scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_l_all)  # (B, H, S, L)

        # Project back to head_dim by a zero-pad trick to keep output shape compatible.
        # This keeps integration simple; later we can add a learned up-projection if needed.
        if self.latent_dim < self.head_dim:
            pad = self.head_dim - self.latent_dim
            out = F.pad(out, (0, pad))

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)
