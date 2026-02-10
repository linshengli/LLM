# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KVCache:
    """MLA 的简化 KV cache（latent 空间）。"""

    k: Optional[torch.Tensor] = None  # (B, H, S, L)
    v: Optional[torch.Tensor] = None  # (B, H, S, L)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k is None:
            self.k = k_new
            self.v = v_new
            return
        self.k = torch.cat([self.k, k_new], dim=-2)
        self.v = torch.cat([self.v, v_new], dim=-2)

