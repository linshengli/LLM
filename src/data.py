# -*- coding: utf-8 -*-
"""
数据集加载、Tokenizer 集成和 DataLoader 构建模块。

使用 Qwen2.5-0.5B 的 AutoTokenizer 对 Wikipedia 中文子集进行编码，
采用"拼接再分块"（concatenate-then-chunk）策略生成固定长度的训练样本。
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset

from src.config import ModelConfig, TrainingConfig


def load_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B") -> PreTrainedTokenizer:
    """加载与教师模型兼容的 Tokenizer。

    Qwen2.5 Tokenizer 默认没有 pad_token，需要设置 pad_token = eos_token
    以避免 DataLoader 在 padding 时报错。

    参数:
        model_id: HuggingFace 模型 ID
    返回:
        配置好 pad_token 的 Tokenizer 实例
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen2.5 Tokenizer 默认无 pad_token，设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class WikiDataset(Dataset):
    """Wikipedia 中文子集数据集，预处理为固定长度 token 序列。

    处理流程（concatenate-then-chunk 策略）：
    1. 从 HuggingFace 加载 wikipedia/20231101.zh 数据集
    2. 提取文本字段，使用 tokenizer 编码为 token ID
    3. 将所有文章的 token ID 拼接为一维长序列
    4. 按 seq_len 切分为等长样本（丢弃末尾不足一个样本的部分）
    5. 构造 labels：input_ids 右移一位，首位填充 -100

    此策略避免了短文本的 padding 浪费，最大化数据利用效率。
    """

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
            max_samples: 原始文章的最大条数（用于控制数据量）
        """
        super().__init__()
        self.seq_len = seq_len

        # 加载 Wikipedia 中文数据集（wikimedia 命名空间）
        dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")

        # 限制原始文章数量（控制数据量在 ~50MB）
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # 按 90/10 比例拆分训练集/验证集
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        raw_data = split_dataset["train"] if split == "train" else split_dataset["test"]

        # 批量编码所有文本并拼接为一维长序列
        all_tokens = []
        for example in raw_data:
            tokens = tokenizer.encode(example["text"], add_special_tokens=False)
            all_tokens.extend(tokens)

        # 按 seq_len 切分为等长样本（丢弃末尾不完整部分）
        total_tokens = len(all_tokens)
        num_samples = total_tokens // seq_len
        all_tokens = all_tokens[: num_samples * seq_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(num_samples, seq_len)

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """返回单条训练样本。

        返回:
            {
                "input_ids": shape (seq_len,), dtype=long — 输入 token 序列
                "labels": shape (seq_len,), dtype=long — 目标标签（右移一位）
            }
            labels[0] = -100（忽略位置），labels[1:] = input_ids[:-1]
            即 labels[i] 是 input_ids[i-1]，模型需根据当前 token 预测下一个 token
        """
        input_ids = self.data[idx]
        # 构造 labels: 右移一位，首位填充 -100
        labels = torch.cat([
            torch.tensor([-100], dtype=torch.long),
            input_ids[:-1],
        ])
        return {"input_ids": input_ids, "labels": labels}


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    config: TrainingConfig,
    model_config: ModelConfig,
    max_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。

    参数:
        tokenizer: Tokenizer 实例
        config: 训练配置（含 batch_size）
        model_config: 模型配置（含 max_seq_len）
        max_samples: 原始文章最大条数（传递给 WikiDataset）
    返回:
        (train_loader, val_loader)
    """
    train_dataset = WikiDataset(
        tokenizer=tokenizer,
        seq_len=model_config.max_seq_len,
        split="train",
        max_samples=max_samples,
    )
    val_dataset = WikiDataset(
        tokenizer=tokenizer,
        seq_len=model_config.max_seq_len,
        split="validation",
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
