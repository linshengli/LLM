# -*- coding: utf-8 -*-
"""
配置数据类模块。

定义模型架构和训练过程中使用的所有超参数配置。
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型架构超参数配置。

    定义 Decoder-Only Transformer 学生模型的所有架构参数，
    与 Qwen2.5 架构特性对齐（GQA、RoPE、SwiGLU）。
    """

    hidden_size: int = 512          # 隐藏层维度
    num_layers: int = 12            # Transformer 层数
    num_heads: int = 8              # 注意力头数（Q 头数）
    num_kv_heads: int = 2           # KV 头数（GQA 分组）
    intermediate_size: int = 2048   # SwiGLU FFN 中间维度
    vocab_size: int = 151665        # 词表大小（与 Qwen2.5-0.5B Tokenizer 一致）
    max_seq_len: int = 512          # 最大序列长度
    rope_theta: float = 1000000.0   # RoPE 旋转基数
    norm_eps: float = 1e-6          # RMSNorm epsilon
    dropout: float = 0.0            # Dropout 率（蒸馏时通常为 0）

    def __post_init__(self):
        """验证配置参数的合法性。"""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 "
                f"num_heads ({self.num_heads}) 整除"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) 必须能被 "
                f"num_kv_heads ({self.num_kv_heads}) 整除"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) 必须大于 0")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) 必须大于 0")

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""
        return self.hidden_size // self.num_heads


@dataclass
class TrainingConfig:
    """训练超参数配置。

    定义知识蒸馏训练过程中的所有可调参数。
    """

    batch_size: int = 8             # 批大小
    learning_rate: float = 3e-4     # 初始学习率
    weight_decay: float = 0.01      # 权重衰减
    warmup_steps: int = 500         # 学习率线性预热步数
    num_epochs: int = 3             # 训练轮数
    gradient_clip: float = 1.0      # 梯度裁剪阈值
    alpha: float = 0.5              # 蒸馏损失权重（vs 标签损失）
    temperature: float = 2.0        # 蒸馏温度
    checkpoint_dir: str = "checkpoints/"  # 检查点保存路径
    log_interval: int = 50          # 日志打印步数间隔
    eval_interval: int = 500        # 验证评估步数间隔
    save_interval: int = 1000       # 检查点保存步数间隔
    # 蒸馏 KL 的显存优化：对 vocab 维度分块计算，避免一次性构造 (B,S,V) 级别的 probs/log_probs 临时张量
    use_chunked_kl: bool = True
    kl_chunk_size: int = 4096
