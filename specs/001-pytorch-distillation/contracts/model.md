# 模块接口契约: model.py

**文件**: `src/model.py`
**职责**: 从零实现 Decoder-Only Transformer 模型的所有组件

## 公开接口

### RMSNorm

```python
class RMSNorm(nn.Module):
    """RMS 归一化层（Root Mean Square Layer Normalization）"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        参数:
            hidden_size: 归一化维度
            eps: 数值稳定性 epsilon
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x — shape (batch, seq_len, hidden_size)
        输出: shape (batch, seq_len, hidden_size)
        """
```

### RotaryEmbedding

```python
class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 1000000.0):
        """
        参数:
            dim: 每个注意力头的维度 (head_dim)
            max_seq_len: 最大序列长度
            theta: RoPE 旋转基数
        """

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        输入: x — shape (batch, num_heads, seq_len, head_dim)
              position_ids — shape (batch, seq_len)
        输出: 应用旋转位置编码后的张量，shape 不变
        """
```

### GQAAttention

```python
class GQAAttention(nn.Module):
    """Grouped-Query Attention（分组查询注意力）"""

    def __init__(self, config: ModelConfig):
        """
        参数:
            config: 包含 hidden_size, num_heads, num_kv_heads 等配置
        """

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
```

### SwiGLUFFN

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        参数:
            hidden_size: 输入/输出维度
            intermediate_size: 中间层维度
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x — shape (batch, seq_len, hidden_size)
        输出: shape (batch, seq_len, hidden_size)
        计算: SiLU(gate_proj(x)) * up_proj(x) → down_proj
        """
```

### TransformerBlock

```python
class TransformerBlock(nn.Module):
    """单个 Transformer 解码器层"""

    def __init__(self, config: ModelConfig):
        """包含 attention_norm → attention → ffn_norm → ffn 的 pre-norm 结构"""

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
```

### StudentModel

```python
class StudentModel(nn.Module):
    """从零实现的学生语言模型"""

    def __init__(self, config: ModelConfig):
        """
        初始化 embedding → transformer_blocks × N → final_norm → lm_head
        lm_head 与 embedding 权重共享
        """

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

    def count_parameters(self) -> int:
        """返回模型总参数量"""
```

## 依赖

- `config.py`: ModelConfig

## 不做

- 不实现 KV Cache（Phase 1 不涉及高效推理优化）
- 不实现 Flash Attention（保持代码可读性，标准实现即可）
- 不实现多 GPU 并行（单 Colab T4）
