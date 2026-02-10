"""训练模块测试 — 验证蒸馏损失、教师模型加载、检查点和训练循环。"""

import os
import tempfile

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig, TrainingConfig
from src.model import StudentModel
from src.trainer import distillation_loss, DistillationTrainer, load_teacher_model


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

BATCH = 4
SEQ_LEN = 16
DEVICE = torch.device("cpu")


def _make_dummy_dataloaders(batch_size=BATCH, num_batches=5):
    """创建用于测试的虚拟 DataLoader。"""
    input_ids = torch.randint(0, TEST_CONFIG.vocab_size, (batch_size * num_batches, SEQ_LEN))
    labels = torch.cat([
        torch.full((batch_size * num_batches, 1), -100, dtype=torch.long),
        input_ids[:, :-1],
    ], dim=1)
    dataset = TensorDataset(input_ids, labels)

    class DictDataLoader:
        """将 TensorDataset 输出包装为字典格式。"""
        def __init__(self, dataloader):
            self._loader = dataloader
        def __iter__(self):
            for ids, labs in self._loader:
                yield {"input_ids": ids, "labels": labs}
        def __len__(self):
            return len(self._loader)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return DictDataLoader(loader), DictDataLoader(loader)


class TestDistillationLoss:
    """蒸馏损失函数测试。"""

    def test_returns_tuple(self):
        """distillation_loss 应返回 (total_loss, metrics_dict)。"""
        student_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        teacher_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        labels = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        total_loss, metrics = distillation_loss(student_logits, teacher_logits, labels)
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_total_loss_is_scalar(self):
        """total_loss 应为标量张量且 > 0。"""
        student_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        teacher_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        labels = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        total_loss, _ = distillation_loss(student_logits, teacher_logits, labels)
        assert total_loss.dim() == 0
        assert total_loss.item() > 0

    def test_metrics_keys(self):
        """metrics_dict 应包含 kl_loss, ce_loss, total_loss。"""
        student_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        teacher_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        labels = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        _, metrics = distillation_loss(student_logits, teacher_logits, labels)
        assert "kl_loss" in metrics
        assert "ce_loss" in metrics
        assert "total_loss" in metrics

    def test_alpha_zero_equals_ce(self):
        """alpha=0 时 total_loss 应近似等于 ce_loss。"""
        student_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        teacher_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        labels = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        total_loss, metrics = distillation_loss(
            student_logits, teacher_logits, labels, alpha=0.0
        )
        assert torch.isclose(total_loss, torch.tensor(metrics["ce_loss"]), atol=1e-5)

    def test_alpha_one_equals_kl(self):
        """alpha=1 时 total_loss 应近似等于 T²·kl_loss。"""
        student_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        teacher_logits = torch.randn(BATCH, SEQ_LEN, TEST_CONFIG.vocab_size)
        labels = torch.randint(0, TEST_CONFIG.vocab_size, (BATCH, SEQ_LEN))
        T = 2.0
        total_loss, metrics = distillation_loss(
            student_logits, teacher_logits, labels, alpha=1.0, temperature=T
        )
        expected = T * T * metrics["kl_loss"]
        assert torch.isclose(total_loss, torch.tensor(expected), atol=1e-4)


class TestLoadTeacherModel:
    """教师模型加载测试（跳过实际下载，仅测试接口）。"""

    @pytest.mark.skipif(
        not os.environ.get("RUN_HEAVY_TESTS"),
        reason="需要下载 Qwen2.5-0.5B 模型，设置 RUN_HEAVY_TESTS=1 启用",
    )
    def test_teacher_model_frozen(self):
        """教师模型的所有参数应冻结（requires_grad=False）。"""
        model = load_teacher_model(device=DEVICE)
        for param in model.parameters():
            assert not param.requires_grad


class TestDistillationTrainer:
    """蒸馏训练器测试（使用虚拟数据和虚拟教师模型）。"""

    def _create_trainer(self, tmpdir):
        """创建用于测试的 DistillationTrainer。"""
        student = StudentModel(TEST_CONFIG)
        # 用另一个 StudentModel 模拟教师模型（不需要真实 Qwen2.5）
        teacher = StudentModel(TEST_CONFIG)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        train_loader, val_loader = _make_dummy_dataloaders()
        training_config = TrainingConfig(
            batch_size=BATCH,
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=2,
            checkpoint_dir=tmpdir,
            log_interval=2,
            eval_interval=5,
            save_interval=5,
        )
        return DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=DEVICE,
        )

    def test_checkpoint_roundtrip(self):
        """save_checkpoint 和 load_checkpoint 应能往返恢复状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._create_trainer(tmpdir)
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

            # 做几步训练改变模型状态
            trainer.step = 3
            trainer.save_checkpoint(ckpt_path)

            # 创建新 trainer 并加载检查点
            trainer2 = self._create_trainer(tmpdir)
            trainer2.load_checkpoint(ckpt_path)
            assert trainer2.step == 3

    def test_loss_decreases(self):
        """训练 10 步后 loss 应下降。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            student = StudentModel(TEST_CONFIG)
            teacher = StudentModel(TEST_CONFIG)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            train_loader, val_loader = _make_dummy_dataloaders(num_batches=10)
            training_config = TrainingConfig(
                batch_size=BATCH,
                learning_rate=1e-3,
                num_epochs=2,
                warmup_steps=2,
                checkpoint_dir=tmpdir,
                log_interval=100,  # 减少输出
                eval_interval=100,
                save_interval=100,
            )
            trainer = DistillationTrainer(
                student_model=student,
                teacher_model=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=DEVICE,
            )
            history = trainer.train()
            losses = history["train_loss"]
            # 最后的 loss 应比第一个 loss 低
            assert losses[-1] < losses[0], (
                f"Loss 未下降: 首次 {losses[0]:.4f} → 最终 {losses[-1]:.4f}"
            )

    def test_evaluate_returns_metrics(self):
        """evaluate 应返回包含 val_loss 和 val_ppl 的字典。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._create_trainer(tmpdir)
            metrics = trainer.evaluate()
            assert "val_loss" in metrics
            assert "val_ppl" in metrics
            assert metrics["val_loss"] > 0
            assert metrics["val_ppl"] > 0
