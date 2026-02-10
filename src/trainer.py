# -*- coding: utf-8 -*-
"""
知识蒸馏训练模块。

实现标准的 logit-level 知识蒸馏训练循环：
- 教师模型（Qwen2.5-0.5B）提供软标签（soft labels）
- 学生模型同时学习教师知识（KL 散度）和真实标签（交叉熵）
- 支持检查点保存/加载，可从中断处恢复训练
"""

import os
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.config import TrainingConfig
from src.model import StudentModel

logger = logging.getLogger(__name__)


def _kl_div_teacher_student_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """按 vocab 分块计算 KL(teacher || student)，降低峰值显存。

    等价于:
      KL = mean_tokens sum_v p_t(v) * (log p_t(v) - log p_s(v))
    其中 p_t = softmax(teacher_logits/T), p_s = softmax(student_logits/T)。

    关键点: 不显式构造 (B,S,V) 的 teacher_probs / student_log_probs。
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits shape {tuple(student_logits.shape)} != "
            f"teacher_logits shape {tuple(teacher_logits.shape)}"
        )
    if student_logits.dim() != 3:
        raise ValueError(f"expected logits dim=3, got {student_logits.dim()}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    bsz, seqlen, vocab = student_logits.shape
    device = student_logits.device

    token_mask = None
    if labels is not None:
        if labels.shape != (bsz, seqlen):
            raise ValueError(f"labels shape {tuple(labels.shape)} != {(bsz, seqlen)}")
        token_mask = labels.ne(ignore_index)  # (B,S)

    # 先算 max(logits) (unscaled). 利用: max(x/T) == max(x)/T (T>0) 避免分配 x/T 张量。
    s_max = student_logits.max(dim=-1).values  # (B,S)
    t_max = teacher_logits.max(dim=-1).values  # (B,S)

    inv_t = 1.0 / float(temperature)

    # log denom: log(sum(exp((x - max)/T)))，用 float32 分块累加 sumexp，避免整块 float32 拷贝。
    s_sumexp = torch.zeros((bsz, seqlen), device=device, dtype=torch.float32)
    t_sumexp = torch.zeros((bsz, seqlen), device=device, dtype=torch.float32)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        s_chunk = student_logits[..., start:end]
        t_chunk = teacher_logits[..., start:end]
        s_sumexp += torch.exp(((s_chunk - s_max.unsqueeze(-1)) * inv_t).float()).sum(dim=-1)
        t_sumexp += torch.exp(((t_chunk - t_max.unsqueeze(-1)) * inv_t).float()).sum(dim=-1)
    s_log_denom = torch.log(s_sumexp)  # (B,S)
    t_log_denom = torch.log(t_sumexp)  # (B,S)

    kl_per_token = torch.zeros((bsz, seqlen), device=device, dtype=torch.float32)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        s_chunk = student_logits[..., start:end]
        t_chunk = teacher_logits[..., start:end]

        log_p_s = ((s_chunk - s_max.unsqueeze(-1)) * inv_t).float() - s_log_denom.unsqueeze(-1)
        log_p_t = ((t_chunk - t_max.unsqueeze(-1)) * inv_t).float() - t_log_denom.unsqueeze(-1)
        p_t = torch.exp(log_p_t)
        kl_per_token += (p_t * (log_p_t - log_p_s)).sum(dim=-1)

    if token_mask is not None:
        kl_per_token = kl_per_token.masked_fill(~token_mask, 0.0)
        denom = token_mask.sum().clamp(min=1).to(torch.float32)
    else:
        denom = torch.tensor(bsz * seqlen, device=device, dtype=torch.float32)

    # reduction="batchmean" on (B*S,V) => mean over tokens.
    return kl_per_token.sum() / denom


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    use_chunked_kl: bool = True,
    kl_chunk_size: int = 4096,
) -> tuple[torch.Tensor, dict]:
    """计算蒸馏损失。

    公式: L = α * T² * KL(softmax(s/T) || softmax(t/T)) + (1-α) * CE(s, labels)

    KL 散度衡量学生与教师输出分布的差异（软标签知识）。
    交叉熵确保学生也学习真实标签（硬标签监督）。
    T² 缩放因子补偿温度对梯度的缩放效应（Hinton et al. 2015）。

    参数:
        student_logits: 学生模型输出, shape (batch, seq_len, vocab_size)
        teacher_logits: 教师模型输出, shape (batch, seq_len, vocab_size)
        labels: 目标标签, shape (batch, seq_len)
        alpha: 蒸馏损失权重（0=纯 CE, 1=纯 KL）
        temperature: 蒸馏温度（越高分布越平滑，暴露更多教师知识）
    返回:
        (total_loss, {"kl_loss": float, "ce_loss": float, "total_loss": float})
    """
    vocab_size = student_logits.size(-1)

    # KL 散度损失：比较温度缩放后的软化分布
    # 默认走分块版本，避免 log_softmax/softmax 产生 (B,S,V) 的大临时张量导致 OOM。
    if use_chunked_kl:
        kl_loss = _kl_div_teacher_student_chunked(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=temperature,
            labels=labels,
            ignore_index=-100,
            chunk_size=kl_chunk_size,
        )
    else:
        # log_softmax(student/T) 与 softmax(teacher/T)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        # KL(P || Q) = sum(P * (log(P) - log(Q)))，这里 P=teacher, Q=student
        kl_loss = F.kl_div(
            student_log_probs.view(-1, vocab_size),
            teacher_probs.view(-1, vocab_size),
            reduction="batchmean",
        )

    # 交叉熵损失：学生预测 vs 真实标签
    # ignore_index=-100 忽略 labels 中的填充位置
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )

    # 加权组合: α·T²·KL + (1-α)·CE
    total_loss = alpha * (temperature ** 2) * kl_loss + (1 - alpha) * ce_loss

    metrics = {
        "kl_loss": kl_loss.item(),
        "ce_loss": ce_loss.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


def load_teacher_model(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """加载教师模型（FP16, eval mode, 冻结参数）。

    参数:
        model_id: HuggingFace 模型 ID
        device: 目标设备
    返回:
        冻结的教师模型实例
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    # 冻结所有参数，教师模型不参与梯度更新
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    return model


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.033,  # lr_min / lr ≈ 1e-5 / 3e-4
) -> LambdaLR:
    """创建带线性预热的余弦退火学习率调度器。"""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # 线性预热
            return current_step / max(1, warmup_steps)
        # 余弦退火
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class DistillationTrainer:
    """知识蒸馏训练管理器。

    管理完整的训练生命周期：优化器初始化、训练循环、验证评估、检查点保存/加载。
    """

    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: torch.device,
    ):
        """
        初始化训练器，创建优化器和学习率调度器。
        教师模型设置为 eval 模式且冻结参数。
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 确保教师模型冻结
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # 蒸馏要求 teacher/student logits 的 vocab 维度一致，否则 KL/CE 计算会直接报错。
        teacher_vocab = getattr(getattr(self.teacher, "config", None), "vocab_size", None)
        student_vocab = getattr(getattr(self.student, "config", None), "vocab_size", None)
        if teacher_vocab is not None and student_vocab is not None and teacher_vocab != student_vocab:
            raise ValueError(
                f"teacher vocab_size ({teacher_vocab}) != student vocab_size ({student_vocab}). "
                "请用 teacher.config.vocab_size 构造 ModelConfig(vocab_size=...) 后重建 StudentModel。"
            )

        # AdamW 优化器
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 余弦退火学习率调度器
        total_steps = config.num_epochs * len(train_loader)
        self.scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer, config.warmup_steps, total_steps
        )

        # 训练状态
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")

    def train(self) -> dict:
        """执行完整训练流程。

        返回:
            训练历史 {"train_loss": [...], "val_loss": [...], "val_ppl": [...]}
        """
        history = {"train_loss": [], "val_loss": [], "val_ppl": []}

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.student.train()

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 教师前向传播（不计算梯度）
                # 关键：对 HF CausalLM 禁用 use_cache（past_key_values），否则会额外分配大量 KV cache 显存。
                # inference_mode 比 no_grad 更省内存/更快（不记录版本计数等），适合纯推理的 teacher。
                with torch.inference_mode():
                    try:
                        teacher_outputs = self.teacher(
                            input_ids,
                            use_cache=False,
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=True,
                        )
                    except TypeError:
                        # 兼容非 HF 模型或不支持上述参数的 teacher
                        teacher_outputs = self.teacher(input_ids)
                    # 兼容 HuggingFace 模型输出（有 .logits 属性）和普通张量
                    teacher_logits = (
                        teacher_outputs.logits
                        if hasattr(teacher_outputs, "logits")
                        else teacher_outputs
                    )

                # 学生前向传播
                student_logits = self.student(input_ids)

                # 计算蒸馏损失
                loss, metrics = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    alpha=self.config.alpha,
                    temperature=self.config.temperature,
                    use_chunked_kl=getattr(self.config, "use_chunked_kl", True),
                    kl_chunk_size=getattr(self.config, "kl_chunk_size", 4096),
                )
                # teacher_logits/outputs 不参与反传，尽早释放引用以降低峰值（显存由缓存分配器复用，不一定立刻回到系统）。
                del teacher_outputs, teacher_logits

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # NaN/Inf 梯度检测
                has_nan = False
                for name, param in self.student.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"步骤 {self.step}: {name} 梯度中检测到 NaN/Inf")
                            has_nan = True

                if not has_nan:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                self.step += 1
                history["train_loss"].append(metrics["total_loss"])

                # 日志输出
                if self.step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1} | Step {self.step} | "
                        f"Loss: {metrics['total_loss']:.4f} | "
                        f"KL: {metrics['kl_loss']:.4f} | "
                        f"CE: {metrics['ce_loss']:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                # 验证评估
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    history["val_loss"].append(val_metrics["val_loss"])
                    history["val_ppl"].append(val_metrics["val_ppl"])
                    print(
                        f"  验证 | Loss: {val_metrics['val_loss']:.4f} | "
                        f"PPL: {val_metrics['val_ppl']:.2f}"
                    )
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                        self.save_checkpoint(best_path)
                        print(f"  新最佳模型已保存: val_loss={self.best_val_loss:.4f}")

                # 定期保存检查点
                if self.step % self.config.save_interval == 0:
                    ckpt_path = os.path.join(
                        self.config.checkpoint_dir, f"checkpoint_step{self.step}.pt"
                    )
                    self.save_checkpoint(ckpt_path)

        # 训练结束保存最终检查点
        final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
        self.save_checkpoint(final_path)
        print(f"训练完成。最终检查点已保存至 {final_path}")

        return history

    def evaluate(self) -> dict:
        """在验证集上评估模型。

        返回:
            {"val_loss": float, "val_ppl": float}
        """
        self.student.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                try:
                    teacher_outputs = self.teacher(
                        input_ids,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                except TypeError:
                    teacher_outputs = self.teacher(input_ids)
                teacher_logits = (
                    teacher_outputs.logits
                    if hasattr(teacher_outputs, "logits")
                    else teacher_outputs
                )

                student_logits = self.student(input_ids)
                loss, _ = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    alpha=self.config.alpha,
                    temperature=self.config.temperature,
                    use_chunked_kl=getattr(self.config, "use_chunked_kl", True),
                    kl_chunk_size=getattr(self.config, "kl_chunk_size", 4096),
                )
                total_loss += loss.item()
                num_batches += 1
                del teacher_outputs, teacher_logits, student_logits

        self.student.train()

        avg_loss = total_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 100))  # 防止 exp 溢出
        return {"val_loss": avg_loss, "val_ppl": ppl}

    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点。

        保存内容: 模型权重、优化器状态、调度器状态、训练进度
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """从检查点恢复训练状态。

        恢复: 模型权重、优化器状态、调度器状态、训练进度
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
