# -*- coding: utf-8 -*-
"""
文本生成推理模块。

加载训练后的学生模型，支持三种自回归解码策略：
- greedy: 贪心搜索（每步选概率最高的 token）
- top_k: Top-K 采样（从概率最高的 K 个 token 中采样）
- top_p: Nucleus 采样（从累积概率达到 p 的最小 token 集合中采样）
"""

import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import StudentModel


def load_trained_model(
    checkpoint_path: str,
    config: ModelConfig,
    device: torch.device,
) -> StudentModel:
    """从检查点加载训练后的学生模型。

    参数:
        checkpoint_path: 检查点文件路径
        config: 模型配置
        device: 目标设备
    返回:
        加载权重并设置为 eval 模式的 StudentModel
    """
    model = StudentModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


class TextGenerator:
    """文本生成器，支持多种解码策略。"""

    def __init__(
        self,
        model: StudentModel,
        tokenizer,
        device: torch.device,
    ):
        """
        参数:
            model: 训练后的学生模型（eval mode）
            tokenizer: Tokenizer 实例
            device: 推理设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """根据提示词生成文本。

        参数:
            prompt: 输入提示词文本
            max_new_tokens: 最大生成 token 数
            strategy: "greedy" | "top_k" | "top_p"
            temperature: 采样温度（仅 top_k/top_p 时生效，<1 更确定，>1 更随机）
            top_k: top-k 采样的 k 值
            top_p: nucleus 采样的概率阈值
        返回:
            生成的完整文本（含提示词）
        """
        # 编码 prompt 为 token ID
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(self.device)

        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            if input_ids.shape[1] >= self.model.config.max_seq_len:
                input_ids = input_ids[:, -self.model.config.max_seq_len:]

            # 前向传播获取 logits
            logits = self.model(input_ids)
            # 取最后一个位置的 logits（预测下一个 token）
            next_token_logits = logits[:, -1, :]

            # 根据策略选择下一个 token
            if strategy == "greedy":
                next_token = self._greedy_decode(next_token_logits)
            elif strategy == "top_k":
                next_token = self._top_k_decode(next_token_logits, temperature, top_k)
            elif strategy == "top_p":
                next_token = self._top_p_decode(next_token_logits, temperature, top_p)
            else:
                raise ValueError(f"未知解码策略: {strategy}")

            # 拼接到序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # 遇到 eos_token 时停止
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # 解码为文本
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def _greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """贪心解码：选择概率最高的 token。"""
        return torch.argmax(logits, dim=-1)

    def _top_k_decode(
        self, logits: torch.Tensor, temperature: float, k: int
    ) -> torch.Tensor:
        """Top-K 采样：保留概率最高的 K 个 token，其余设为 -inf 后采样。

        通过限制候选 token 数量来平衡多样性和质量。
        """
        # 温度缩放
        logits = logits / max(temperature, 1e-8)
        # 找到前 K 个最大值
        top_k_logits, top_k_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        # 将非 Top-K 位置设为 -inf
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, top_k_indices, top_k_logits)
        # 按概率采样
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _top_p_decode(
        self, logits: torch.Tensor, temperature: float, p: float
    ) -> torch.Tensor:
        """Nucleus (Top-P) 采样：从累积概率达到 p 的最小 token 集合中采样。

        动态调整候选 token 数量——当模型非常确定时选择少数 token，
        当模型不确定时考虑更多 token。
        """
        # 温度缩放
        logits = logits / max(temperature, 1e-8)
        # 按概率降序排列
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到累积概率超过阈值 p 的位置，将其对应 logit 设为 -inf
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        # 恢复原始顺序
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(1, sorted_indices, sorted_logits)
        # 按概率采样
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
