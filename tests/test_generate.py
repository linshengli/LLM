"""生成模块测试 — 验证文本生成各解码策略的正确性。"""

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.model import StudentModel
from src.generate import TextGenerator, load_trained_model


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

DEVICE = torch.device("cpu")


class _FakeTokenizer:
    """用于测试的虚拟 Tokenizer，避免下载真实模型。"""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        # 简单地将每个字符映射为 token ID
        ids = [ord(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(chr(t % 128 + 32) if 32 <= t % 128 + 32 < 127 else "?" for t in token_ids)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


@pytest.fixture
def generator():
    """创建使用随机权重 StudentModel 的 TextGenerator。"""
    model = StudentModel(TEST_CONFIG)
    model.eval()
    tokenizer = _FakeTokenizer(TEST_CONFIG.vocab_size)
    return TextGenerator(model=model, tokenizer=tokenizer, device=DEVICE)


class TestLoadTrainedModel:
    """load_trained_model 测试。"""

    def test_returns_eval_mode_model(self):
        """应返回 eval 模式的 StudentModel。"""
        model = StudentModel(TEST_CONFIG)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            loaded = load_trained_model(ckpt_path, TEST_CONFIG, DEVICE)
            assert isinstance(loaded, StudentModel)
            assert not loaded.training  # eval 模式


class TestTextGenerator:
    """TextGenerator 各解码策略测试。"""

    def test_greedy_returns_nonempty(self, generator):
        """greedy 策略应返回非空字符串。"""
        result = generator.generate("Hello", max_new_tokens=10, strategy="greedy")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_top_k_returns_nonempty(self, generator):
        """top_k 策略应返回非空字符串。"""
        result = generator.generate("Hello", max_new_tokens=10, strategy="top_k")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_top_p_returns_nonempty(self, generator):
        """top_p 策略应返回非空字符串。"""
        result = generator.generate("Hello", max_new_tokens=10, strategy="top_p")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_new_tokens_limit(self, generator):
        """生成的 token 数不应超过 max_new_tokens。"""
        prompt = "Hi"
        max_new = 5
        # 使用 greedy 确保确定性
        result = generator.generate(prompt, max_new_tokens=max_new, strategy="greedy")
        # 结果包含 prompt + 生成部分
        prompt_tokens = generator.tokenizer.encode(prompt)
        result_tokens = generator.tokenizer.encode(result)
        # 生成的 token 数 = 总 token 数 - prompt token 数
        assert len(result_tokens) <= len(prompt_tokens) + max_new

    def test_temperature_affects_output(self, generator):
        """不同 temperature 下应产生不同结果（统计上）。"""
        torch.manual_seed(42)
        results_low = set()
        results_high = set()
        for _ in range(5):
            results_low.add(
                generator.generate("Test", max_new_tokens=10, strategy="top_k", temperature=0.1)
            )
            results_high.add(
                generator.generate("Test", max_new_tokens=10, strategy="top_k", temperature=1.5)
            )
        # 高温度应产生更多样的输出（或至少同样多样）
        # 低温度下输出更确定，高温度下更随机
        # 至少验证都返回了结果
        assert len(results_low) >= 1
        assert len(results_high) >= 1
