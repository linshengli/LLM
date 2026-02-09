# 模块接口契约: trainer.py

**文件**: `src/trainer.py`
**职责**: 知识蒸馏训练循环、损失计算、检查点管理和指标记录

## 公开接口

### distillation_loss

```python
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """
    计算蒸馏损失 = α * T² * KL(student/T || teacher/T) + (1-α) * CE(student, labels)

    参数:
        student_logits: 学生模型输出, shape (batch, seq_len, vocab_size)
        teacher_logits: 教师模型输出, shape (batch, seq_len, vocab_size)
        labels: 目标标签, shape (batch, seq_len)
        alpha: 蒸馏损失权重
        temperature: 蒸馏温度
    返回:
        (total_loss, {"kl_loss": ..., "ce_loss": ..., "total_loss": ...})
    """
```

### DistillationTrainer

```python
class DistillationTrainer:
    """知识蒸馏训练管理器"""

    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ):
        """
        初始化训练器，创建优化器和学习率调度器。
        教师模型设置为 eval 模式且冻结参数。
        """

    def train(self) -> dict:
        """
        执行完整训练流程。

        返回:
            训练历史 {"train_loss": [...], "val_loss": [...], "val_ppl": [...]}

        行为:
            - 每 log_interval 步打印训练指标
            - 每 eval_interval 步在验证集上评估
            - 每 save_interval 步保存检查点
            - 训练结束时保存最终检查点
            - 检测 NaN/Inf 梯度并记录警告
        """

    def evaluate(self) -> dict:
        """
        在验证集上评估模型。

        返回:
            {"val_loss": float, "val_ppl": float}
        """

    def save_checkpoint(self, path: str) -> None:
        """
        保存训练检查点。

        保存内容: student_model.state_dict(), optimizer.state_dict(),
                  scheduler.state_dict(), epoch, step, best_val_loss, config
        """

    def load_checkpoint(self, path: str) -> None:
        """
        从检查点恢复训练状态。

        恢复: 模型权重、优化器状态、调度器状态、训练进度
        """
```

### load_teacher_model

```python
def load_teacher_model(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """
    加载教师模型（FP16, eval mode, 冻结参数）。

    参数:
        model_id: HuggingFace 模型 ID
        device: 目标设备
    返回:
        冻结的教师模型实例
    """
```

## 依赖

- `config.py`: TrainingConfig
- `model.py`: StudentModel
- 外部: transformers.AutoModelForCausalLM

## 不做

- 不实现分布式训练（DataParallel / DistributedDataParallel）
- 不实现混合精度训练（FP32 即可满足显存需求）
- 不实现 TensorBoard/WandB 集成（使用 print 日志，Notebook 内 matplotlib 可视化）
- 不实现梯度累积（batch_size=8 已足够）
