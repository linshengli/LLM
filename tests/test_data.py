"""数据管道测试 — 验证 Tokenizer 加载、数据集构建和 DataLoader 输出。"""

import torch
import pytest
from src.config import ModelConfig, TrainingConfig
from src.data import load_tokenizer, WikiDataset, create_dataloaders


SEQ_LEN = 64  # 测试用较短序列长度


class TestLoadTokenizer:
    """Tokenizer 加载测试。"""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return load_tokenizer()

    def test_tokenizer_not_none(self, tokenizer):
        """load_tokenizer 应返回有效 Tokenizer。"""
        assert tokenizer is not None

    def test_vocab_size(self, tokenizer):
        """vocab_size 应与 Qwen2.5-0.5B 一致。"""
        assert tokenizer.vocab_size == 151643

    def test_pad_token_set(self, tokenizer):
        """pad_token 应已设置（Qwen2.5 默认无 pad_token）。"""
        assert tokenizer.pad_token is not None

    def test_encode_decode_roundtrip(self, tokenizer):
        """编码后解码应能还原原始文本。"""
        text = "你好，世界！Hello, World!"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        assert text in decoded


class TestWikiDataset:
    """WikiDataset 数据集测试（使用少量数据快速验证）。"""

    @pytest.fixture(scope="class")
    def dataset(self):
        tokenizer = load_tokenizer()
        return WikiDataset(
            tokenizer=tokenizer,
            seq_len=SEQ_LEN,
            split="train",
            max_samples=100,  # 仅取 100 条加速测试
        )

    def test_length(self, dataset):
        """数据集长度应大于 0。"""
        assert len(dataset) > 0

    def test_getitem_keys(self, dataset):
        """__getitem__ 应返回包含 input_ids 和 labels 的字典。"""
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item

    def test_input_ids_shape(self, dataset):
        """input_ids 的形状应为 (seq_len,)。"""
        item = dataset[0]
        assert item["input_ids"].shape == (SEQ_LEN,)

    def test_input_ids_dtype(self, dataset):
        """input_ids 应为 torch.long 类型。"""
        item = dataset[0]
        assert item["input_ids"].dtype == torch.long

    def test_labels_shape(self, dataset):
        """labels 的形状应为 (seq_len,)。"""
        item = dataset[0]
        assert item["labels"].shape == (SEQ_LEN,)

    def test_labels_first_is_ignore(self, dataset):
        """labels 的第一个位置应为 -100（忽略标记）。"""
        item = dataset[0]
        assert item["labels"][0].item() == -100

    def test_labels_shift_relation(self, dataset):
        """labels[1:] 应等于 input_ids[:-1]（右移关系）。"""
        item = dataset[0]
        # labels[i] = input_ids[i-1]，即 labels 相对 input_ids 右移一位
        assert torch.equal(item["labels"][1:], item["input_ids"][:-1])


class TestCreateDataloaders:
    """DataLoader 构建测试。"""

    def test_returns_two_dataloaders(self):
        """create_dataloaders 应返回 (train_loader, val_loader)。"""
        tokenizer = load_tokenizer()
        model_config = ModelConfig(max_seq_len=SEQ_LEN)
        training_config = TrainingConfig(batch_size=4)
        train_loader, val_loader = create_dataloaders(
            tokenizer=tokenizer,
            config=training_config,
            model_config=model_config,
            max_samples=100,
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_shape(self):
        """DataLoader 输出 batch 的 input_ids 形状应为 (batch_size, seq_len)。"""
        batch_size = 4
        tokenizer = load_tokenizer()
        model_config = ModelConfig(max_seq_len=SEQ_LEN)
        training_config = TrainingConfig(batch_size=batch_size)
        train_loader, _ = create_dataloaders(
            tokenizer=tokenizer,
            config=training_config,
            model_config=model_config,
            max_samples=100,
        )
        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["input_ids"].shape[1] == SEQ_LEN
