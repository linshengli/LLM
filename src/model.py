# -*- coding: utf-8 -*-
"""
从零实现的 Decoder-Only Transformer 模型架构。

包含以下核心组件（对齐 Qwen2.5 架构特性）：
- RMSNorm: Root Mean Square 归一化
- RotaryEmbedding: 旋转位置编码（RoPE）
- GQAAttention: 分组查询注意力（Grouped-Query Attention）
- SwiGLUFFN: SwiGLU 前馈网络
- TransformerBlock: 单个 Transformer 解码器层
- StudentModel: 完整的学生语言模型
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.optimizers.api import resolve_profile
from src.optimizers.config_manager import OptimizationProfile


class RMSNorm(nn.Module):
    """RMS 归一化层（Root Mean Square Layer Normalization）。

    与 LayerNorm 不同，RMSNorm 不进行均值中心化，仅做方差归一化，
    计算效率更高且效果相当。Qwen2.5/LLaMA 系列均使用此归一化方式。

    公式: output = x * rsqrt(mean(x²) + eps) * gamma
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        参数:
            hidden_size: 归一化维度
            eps: 数值稳定性 epsilon
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x — shape (batch, seq_len, hidden_size)
        输出: shape (batch, seq_len, hidden_size)
        """
        # 计算输入的均方根: sqrt(mean(x²))
        # 使用 float32 计算避免精度问题
        input_dtype = x.dtype
        x = x.float()
        # variance = mean(x²)，沿最后一维计算
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # x * rsqrt(variance + eps) 完成归一化
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（Rotary Position Embedding, RoPE）。

    RoPE 通过对 Q/K 向量施加旋转变换来编码位置信息。
    核心思想：将每对相邻维度视为二维平面上的向量，根据位置旋转不同角度。

    频率公式: freq_i = 1 / (theta^(2i/dim))，i = 0, 1, ..., dim/2-1
    旋转变换: [x_2i, x_{2i+1}] → [x_2i*cos - x_{2i+1}*sin, x_2i*sin + x_{2i+1}*cos]
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 1000000.0):
        """
        参数:
            dim: 每个注意力头的维度 (head_dim)
            max_seq_len: 最大序列长度
            theta: RoPE 旋转基数（越大高频衰减越慢）
        """
        super().__init__()
        # 预计算频率向量: freq_i = 1 / (theta^(2i/dim))
        # 维度索引 [0, 2, 4, ..., dim-2] / dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用旋转位置编码。

        输入: x — shape (batch, num_heads, seq_len, head_dim)
              position_ids — shape (batch, seq_len)
        输出: 应用旋转位置编码后的张量，shape 不变
        """
        # position_ids: (batch, seq_len) → (batch, seq_len, 1)
        # freqs: (dim/2,) → (1, 1, dim/2)
        # angles: (batch, seq_len, dim/2) — 每个位置在每个频率上的旋转角度
        angles = position_ids.unsqueeze(-1).float() * self.freqs.unsqueeze(0).unsqueeze(0)

        # cos/sin: (batch, seq_len, dim/2) → (batch, 1, seq_len, dim/2) 用于广播到 num_heads
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)

        # 将 x 的最后一维拆分为两半进行旋转
        # x_even: 偶数索引维度, x_odd: 奇数索引维度
        x_even = x[..., 0::2]  # (batch, heads, seq, dim/2)
        x_odd = x[..., 1::2]   # (batch, heads, seq, dim/2)

        # 旋转变换: [even, odd] → [even*cos - odd*sin, even*sin + odd*cos]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 交错合并偶数和奇数维度
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)  # (batch, heads, seq, dim)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头扩展到与 Q 头数量一致（GQA 的关键操作）。

    当 num_kv_heads < num_heads 时，每个 KV 头被复制 n_rep 次，
    使得 KV 头的总数等于 Q 头的总数。

    输入: x — shape (batch, num_kv_heads, seq_len, head_dim)
    输出: shape (batch, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    # 在 kv_heads 维度后插入新维度并扩展
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GQAAttention(nn.Module):
    """分组查询注意力（Grouped-Query Attention）。

    GQA 是 MHA（多头注意力）和 MQA（多查询注意力）的折中方案：
    - MHA: num_kv_heads == num_heads（每个 Q 头有独立的 KV 头）
    - MQA: num_kv_heads == 1（所有 Q 头共享一组 KV 头）
    - GQA: 1 < num_kv_heads < num_heads（每组 Q 头共享一组 KV 头）

    本实现中 num_heads=8, num_kv_heads=2，即每 4 个 Q 头共享一组 KV。
    """

    def __init__(self, config: ModelConfig):
        """
        参数:
            config: 包含 hidden_size, num_heads, num_kv_heads 等配置
        """
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads  # 每个 KV 头被复制的次数

        # Q 投影: hidden_size → num_heads * head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        # K 投影: hidden_size → num_kv_heads * head_dim
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # V 投影: hidden_size → num_kv_heads * head_dim
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # O 投影: num_heads * head_dim → hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入: x — shape (batch, seq_len, hidden_size)
              position_ids — shape (batch, seq_len)
              attention_mask — 因果遮罩, shape (batch, 1, seq_len, seq_len) 或 None
        输出: shape (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape

        # 线性投影: (batch, seq, hidden) → (batch, seq, num_*_heads * head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头形式: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 对 Q 和 K 应用旋转位置编码（V 不需要）
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # GQA: 将 KV 头扩展到与 Q 头数量一致
        k = repeat_kv(k, self.n_rep)  # (batch, num_heads, seq, head_dim)
        v = repeat_kv(v, self.n_rep)

        # 缩放点积注意力: attn = softmax(Q·K^T / sqrt(d_k)) · V
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 应用因果遮罩（上三角为 -inf，防止关注未来 token）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # 合并多头: (batch, heads, seq, head_dim) → (batch, seq, heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # 输出投影
        return self.o_proj(attn_output)


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络。

    SwiGLU 是 GLU（门控线性单元）的 SiLU 变体，由 Google PaLM 引入，
    Qwen2.5/LLaMA 系列均采用此 FFN 结构。

    公式: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    其中 SiLU(x) = x * sigmoid(x)

    相比标准 FFN (ReLU(W1·x)·W2)，SwiGLU 使用门控机制，
    gate 分支控制信息流的通断，up 分支提供信息内容。
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        参数:
            hidden_size: 输入/输出维度
            intermediate_size: 中间层维度
        """
        super().__init__()
        # 门控投影：控制信息通断
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 上投影：提供信息内容
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 下投影：映射回隐藏维度
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x — shape (batch, seq_len, hidden_size)
        输出: shape (batch, seq_len, hidden_size)
        """
        # SiLU(gate(x)) * up(x) → down 投影回 hidden_size
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器层（Pre-norm 架构）。

    使用 Pre-norm 结构（先归一化再计算），与 Post-norm（先计算再归一化）相比，
    Pre-norm 训练更稳定，是 LLaMA/Qwen 等现代 LLM 的标准做法。

    结构: x → norm → attention → + residual → norm → ffn → + residual
    """

    def __init__(self, config: ModelConfig, opt: Optional[OptimizationProfile] = None):
        """包含 attention_norm → attention → ffn_norm → ffn 的 pre-norm 结构。"""
        super().__init__()
        opt = resolve_profile(opt)
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if opt.mla is not None and getattr(opt.mla, "enabled", False):
            from src.optimizers.mla import MLAAttention

            self.attention = MLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                cfg=opt.mla,
            )
        else:
            self.attention = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if opt.moe is not None and getattr(opt.moe, "enabled", False):
            from src.optimizers.moe import MOERouter

            self.ffn = MOERouter(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                cfg=opt.moe,
            )
        else:
            self.ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入/输出: shape (batch, seq_len, hidden_size)
        使用残差连接: x = x + attention(norm(x)); x = x + ffn(norm(x))
        """
        # 注意力子层 + 残差
        x = x + self.attention(self.attention_norm(x), position_ids, attention_mask)
        # FFN 子层 + 残差
        ffn_out = self.ffn(self.ffn_norm(x))
        # MoE 返回结构化输出
        if hasattr(ffn_out, "y"):
            ffn_out = ffn_out.y
        x = x + ffn_out
        return x


class StudentModel(nn.Module):
    """从零实现的学生语言模型（Decoder-Only Transformer）。

    架构: token_embedding → TransformerBlock × N → RMSNorm → lm_head
    与 Qwen2.5 对齐的特性：GQA、RoPE、SwiGLU FFN、RMSNorm、权重共享
    """

    def __init__(self, config: ModelConfig, optimization: Optional[OptimizationProfile] = None):
        """
        初始化 embedding → transformer_blocks × N → final_norm → lm_head。
        lm_head 与 embedding 权重共享（减少参数量）。
        """
        super().__init__()
        self.config = config

        # Token 嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # N 层 Transformer 解码器
        self.layers = nn.ModuleList(
            [TransformerBlock(config, opt=optimization) for _ in range(config.num_layers)]
        )
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        # 语言模型输出头（与 embedding 权重共享）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享: lm_head 的权重与 embedding 的权重是同一个张量
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入: input_ids — shape (batch, seq_len), dtype=torch.long
              attention_mask — 可选因果遮罩
        输出: logits — shape (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape

        # 自动生成位置 ID: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        # 自动生成因果遮罩（上三角为 -inf）
        if attention_mask is None:
            # 创建因果遮罩: 上三角矩阵为 True，对角线为 False
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            # True → -inf, False → 0.0
            attention_mask = causal_mask.float().masked_fill(causal_mask, float("-inf"))
            # 扩展维度以匹配注意力权重: (1, 1, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Token 嵌入
        hidden_states = self.embedding(input_ids)

        # 逐层 Transformer 前向传播
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 映射到词表空间
        logits = self.lm_head(hidden_states)

        return logits

    def count_parameters(self) -> int:
        """返回模型总参数量（考虑权重共享，不重复计算）。"""
        return sum(p.numel() for p in self.parameters())
