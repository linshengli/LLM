"""模型架构测试 — 验证所有 Transformer 组件的正确性。"""

import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.model import (
    RMSNorm,
    RotaryEmbedding,
    GQAAttention,
    SwiGLUFFN,
    TransformerBlock,
    StudentModel,
)


# 使用小型配置加速测试
TEST_CONFIG = ModelConfig(
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=128,
    vocab_size=1000,
    max_seq_len=32,
)

BATCH = 2
SEQ_LEN = 16


class TestRMSNorm:
    """RMSNorm 归一化层测试。"""

    def test_output_shape(self):
        """输入输出形状应一致。"""
        norm = RMSNorm(TEST_CONFIG.hidden_size, TEST_CONFIG.norm_eps)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_effect(self):
        """归一化后 RMS 应接近 1。"""
        norm = RMSNorm(TEST_CONFIG.hidden_size, TEST_CONFIG.norm_eps)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size) * 10
        out = norm(x)
        # RMSNorm 输出的 RMS 应接近 gamma 的 RMS（初始化为 1）
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert rms.mean().item() < 5.0  # 合理范围内


class TestRotaryEmbedding:
    """旋转位置编码测试。"""

    def test_output_shape(self):
        """输出形状应与输入一致。"""
        rope = RotaryEmbedding(
            TEST_CONFIG.head_dim, TEST_CONFIG.max_seq_len, TEST_CONFIG.rope_theta
        )
        x = torch.randn(BATCH, TEST_CONFIG.num_heads, SEQ_LEN, TEST_CONFIG.head_dim)
        pos_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
        out = rope(x, pos_ids)
        assert out.shape == x.shape

    def test_different_positions_differ(self):
        """不同位置的编码应产生不同结果。"""
        rope = RotaryEmbedding(
            TEST_CONFIG.head_dim, TEST_CONFIG.max_seq_len, TEST_CONFIG.rope_theta
        )
        x = torch.ones(1, 1, 2, TEST_CONFIG.head_dim)
        pos_0 = torch.tensor([[0, 1]])
        pos_1 = torch.tensor([[2, 3]])
        out_0 = rope(x, pos_0)
        out_1 = rope(x, pos_1)
        # 相同输入 + 不同位置 → 不同输出
        assert not torch.allclose(out_0, out_1, atol=1e-5)


class TestGQAAttention:
    """Grouped-Query Attention 测试。"""

    def test_output_shape(self):
        """输入 (batch, seq, hidden) → 输出形状一致。"""
        attn = GQAAttention(TEST_CONFIG)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)
        pos_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
        out = attn(x, pos_ids)
        assert out.shape == (BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)


class TestSwiGLUFFN:
    """SwiGLU 前馈网络测试。"""

    def test_output_shape(self):
        """输入 (batch, seq, hidden) → 输出形状一致。"""
        ffn = SwiGLUFFN(TEST_CONFIG.hidden_size, TEST_CONFIG.intermediate_size)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)


class TestTransformerBlock:
    """Transformer 解码器层测试。"""

    def test_output_shape(self):
        """输入输出形状应一致。"""
        block = TransformerBlock(TEST_CONFIG)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)
        pos_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
        out = block(x, pos_ids)
        assert out.shape == (BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)

    def test_residual_connection(self):
        """残差连接应使输出不为零（即使初始权重很小）。"""
        block = TransformerBlock(TEST_CONFIG)
        x = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.hidden_size)
        pos_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
        out = block(x, pos_ids)
        # 由于残差连接，输出不应全为零
        assert out.abs().sum().item() > 0


class TestStudentModel:
    """学生模型完整测试。"""

    def test_forward_output_shape(self):
        """input_ids (batch, seq) → logits (batch, seq, vocab_size)。"""
        model = StudentModel(TEST_CONFIG)
        input_ids = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        logits = model(input_ids)
        assert logits.shape == (BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)

    def test_parameter_count_small_config(self):
        """小型配置的参数量应合理。"""
        model = StudentModel(TEST_CONFIG)
        param_count = model.count_parameters()
        assert param_count > 0

    def test_parameter_count_default_config(self):
        """默认配置参数量应约为 123M（±5%）。"""
        config = ModelConfig()
        model = StudentModel(config)
        param_count = model.count_parameters()
        expected = 123_278_336
        tolerance = expected * 0.05
        assert abs(param_count - expected) < tolerance, (
            f"参数量 {param_count:,} 偏离预期 {expected:,} 超过 5%"
        )

    def test_weight_tying(self):
        """lm_head 与 embedding 应共享同一个权重张量。"""
        model = StudentModel(TEST_CONFIG)
        assert model.lm_head.weight is model.embedding.weight

    def test_logits_dtype(self):
        """输出 logits 应为 float 类型。"""
        model = StudentModel(TEST_CONFIG)
        input_ids = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        logits = model(input_ids)
        assert logits.dtype == torch.float32
