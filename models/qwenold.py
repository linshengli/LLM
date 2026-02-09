"""
Implement my own Qwen model from scratch using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# class RoPE(nn.Embedding):
#     def __init__(self, num_vo):

class sincosEmbedding(nn.Module):
    def __init__(self, len: int):
        super().__init__()
        self.len = len
    
    def embedding(pos, d_model):
        """
            calculate sin/cos embedding
        """
        sin_p = [math.sin(pos / (10000 * ( (i // 2) / d_model))) for i in range(0, len, 2)] 
        cos_p = [math.cos(pos / (10000 * ( (i // 2) / d_model))) for i in range(1, len, 2)] 


class QwenConfig:
    def __init__(self, hidden_size, num_heads) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads

qwen_config = QwenConfig(hidden_size = 1024, num_heads = 32)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        assert self.d_model == self.head_dim * self.num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        # self.q_norm = nn.LayerNorm(d_model)

        self.k_proj = nn.Linear(d_model, d_model)
        # self.k_norm = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = q.shape
        q = self.q_proj(q) # shape: batch, num_heads, seq_len * head_dim
        k = self.k_proj(k) # shape: batch, num_heads, seq_len * head_dim
        v = self.v_proj(v) # shape: batch, num_heads, seq_len * head_dim

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q = self.q_norm(q)
        # k = self.k_norm(k)

        # scores = QK^/sqrt(d_k)
        # q: shape: batch_size, num_heads, seq_len, head_dim
        # k: shape: batch_size, num_heads, seq_len, head_dim
        # scores: shape: batch_size, num_heads, seq_len, seq_len
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) 
        # mask here
        if mask is not None:
            scores = scores.masked_fill(mask == 0)
        weights = F.softmax(scores, dim = -1)

        # attn: shape: batch_size, num_heads, seq_len, head_dim
        attn = weights @ v
        attn = attn.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.o_proj(attn)
        return output

class DecodeLayer(nn.Module):
    def __init__(self, qwen_config: QwenConfig, layer_index: int):
        super().__init__()
        hidden_size = qwen_config.hidden_size
        num_heads = qwen_config.num_heads
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        x = self.pre_norm(x)
        output = self.attention(x, x, x)
        output = self.post_norm(output)
        output = self.ffn(output)
        return output

class QwenModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len, num_layers):
        super().__init__()
        self.transformer_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            DecodeLayer(qwen_config, idx) for idx in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_ids):
        # Embedding layer
        x = self.transformer_embedding(input_ids)
        pos = self.position_embedding(input_ids)
        x = x + pos

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # FFN layer
        x = self.norm(x)
        x = self.lm_head(x)
        # Output layer
        