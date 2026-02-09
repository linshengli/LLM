# 模块接口契约: data.py

**文件**: `src/data.py`
**职责**: 数据集加载、Tokenizer 集成和 DataLoader 构建

## 公开接口

### load_tokenizer

```python
def load_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B") -> PreTrainedTokenizer:
    """
    加载与教师模型兼容的 Tokenizer。

    参数:
        model_id: HuggingFace 模型 ID
    返回:
        配置好 pad_token 的 Tokenizer 实例
    """
```

### WikiDataset

```python
class WikiDataset(torch.utils.data.Dataset):
    """Wikipedia 中文子集数据集，预处理为固定长度 token 序列"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        参数:
            tokenizer: Tokenizer 实例
            seq_len: 每条样本的 token 序列长度
            split: "train" 或 "validation"
            max_samples: 最大样本数（用于控制数据量）

        处理流程:
            1. 从 HuggingFace 加载 wikipedia/20231101.zh
            2. 提取文本字段，使用 tokenizer 编码
            3. 将所有 token 拼接为一维长序列
            4. 按 seq_len 切分为等长样本
        """

    def __len__(self) -> int:
        """返回样本总数"""

    def __getitem__(self, idx: int) -> dict:
        """
        返回:
            {
                "input_ids": torch.Tensor — shape (seq_len,), dtype=long
                "labels": torch.Tensor — shape (seq_len,), dtype=long
            }
            labels 为 input_ids 右移一位，首位填充 -100（忽略）
        """
```

### create_dataloaders

```python
def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    config: TrainingConfig,
    model_config: ModelConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    参数:
        tokenizer: Tokenizer 实例
        config: 训练配置（含 batch_size）
        model_config: 模型配置（含 max_seq_len）
    返回:
        (train_loader, val_loader)
    """
```

## 依赖

- `config.py`: TrainingConfig, ModelConfig
- 外部: transformers.AutoTokenizer, datasets

## 不做

- 不实现自定义 Tokenizer（使用 HuggingFace 预训练 Tokenizer）
- 不实现数据增强（小规模数据直接使用原文）
- 不实现分布式数据采样（单 GPU）
