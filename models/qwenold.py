"""
Implement my own Qwen model from scratch using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def _causal_mask(seq_len, device):
    mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).tril()
    return mask.view(1, 1, seq_len, seq_len)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (- math.log(10000.0)) / d_model)
        pe[:,0::2] = torch.sin(pos * div)#sin 
        pe[:,1::2] = torch.cos(pos * div)#cos
        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, position_ids):
        """
            calculate sin/cos embedding
        """
        return self.pe[position_ids]


class QwenConfig:
    def __init__(self, hidden_size, num_heads, num_layers) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        assert self.d_model == self.head_dim * self.num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        # self.q_norm = nn.LayerNorm(d_model)

        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        # self.k_norm = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x) # shape: batch, num_heads, seq_len * head_dim
        k = self.k_proj(x) # shape: batch, num_heads, seq_len * head_dim
        v = self.v_proj(x) # shape: batch, num_heads, seq_len * head_dim

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
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim = -1)

        # attn: shape: batch_size, num_heads, seq_len, head_dim
        attn = weights @ v
        attn = attn.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.o_proj(attn)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_mul):
        super().__init__()
        inner = hidden_size * ffn_mul
        self.up = nn.Linear(hidden_size, inner)
        self.down = nn.Linear(inner, hidden_size)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class DecodeLayer(nn.Module):
    def __init__(self, qwen_config: QwenConfig, layer_index: int):
        super().__init__()
        hidden_size = qwen_config.hidden_size
        num_heads = qwen_config.num_heads
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, 4)

    def forward(self, x, mask: torch.Tensor = None):
        h = self.pre_norm(x)
        x = x + self.attention(h, mask)
        h = self.post_norm(x)
        x = x + self.ffn(h)
        return x

class QwenModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_len,
        num_layers,
        num_heads,
    ):
        super().__init__()
        self.transformer_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = SinusoidalEmbedding(max_seq_len, hidden_size)

        qwen_config = QwenConfig(hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
        self.layers = nn.ModuleList([
            DecodeLayer(qwen_config, idx) for idx in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_ids):
        # Embedding layer
        bsz, seq_len = input_ids.shape
        mask = _causal_mask(seq_len, device=input_ids.device)
        x = self.transformer_embedding(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        pos = self.position_embedding(position_ids)
        x = x + pos

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask = mask)

        x = self.norm(x)
        return self.lm_head(x)
        
