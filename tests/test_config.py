"""ModelConfig 和 TrainingConfig 配置验证测试。"""

import pytest
from src.config import ModelConfig, TrainingConfig


class TestModelConfig:
    """ModelConfig 默认值与验证规则测试。"""

    def test_default_values(self):
        """测试 ModelConfig 默认值正确性。"""
        config = ModelConfig()
        assert config.hidden_size == 512
        assert config.num_layers == 12
        assert config.num_heads == 8
        assert config.num_kv_heads == 2
        assert config.intermediate_size == 2048
        assert config.vocab_size == 151665
        assert config.max_seq_len == 512
        assert config.rope_theta == 1000000.0
        assert config.norm_eps == 1e-6
        assert config.dropout == 0.0

    def test_head_dim(self):
        """测试 head_dim 属性计算正确。"""
        config = ModelConfig()
        assert config.head_dim == 64  # 512 / 8

    def test_hidden_size_not_divisible_by_num_heads(self):
        """hidden_size 不能被 num_heads 整除时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="hidden_size"):
            ModelConfig(hidden_size=512, num_heads=7)

    def test_num_heads_not_divisible_by_num_kv_heads(self):
        """num_heads 不能被 num_kv_heads 整除时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="num_heads"):
            ModelConfig(num_heads=8, num_kv_heads=3)

    def test_invalid_vocab_size(self):
        """vocab_size <= 0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="vocab_size"):
            ModelConfig(vocab_size=0)

    def test_invalid_max_seq_len(self):
        """max_seq_len <= 0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="max_seq_len"):
            ModelConfig(max_seq_len=0)

    def test_custom_values(self):
        """测试自定义参数值。"""
        config = ModelConfig(hidden_size=256, num_layers=6, num_heads=4, num_kv_heads=2)
        assert config.hidden_size == 256
        assert config.num_layers == 6
        assert config.head_dim == 64


class TestTrainingConfig:
    """TrainingConfig 默认值测试。"""

    def test_default_values(self):
        """测试 TrainingConfig 默认值正确性。"""
        config = TrainingConfig()
        assert config.batch_size == 8
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 500
        assert config.num_epochs == 3
        assert config.gradient_clip == 1.0
        assert config.alpha == 0.5
        assert config.temperature == 2.0
        assert config.checkpoint_dir == "checkpoints/"
        assert config.log_interval == 50
        assert config.eval_interval == 500
        assert config.save_interval == 1000
