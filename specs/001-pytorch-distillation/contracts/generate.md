# 模块接口契约: generate.py

**文件**: `src/generate.py`
**职责**: 加载训练后的学生模型进行文本生成推理

## 公开接口

### TextGenerator

```python
class TextGenerator:
    """文本生成器，支持多种解码策略"""

    def __init__(
        self,
        model: StudentModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        """
        参数:
            model: 训练后的学生模型（eval mode）
            tokenizer: Tokenizer 实例
            device: 推理设备
        """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        根据提示词生成文本。

        参数:
            prompt: 输入提示词文本
            max_new_tokens: 最大生成 token 数
            strategy: "greedy" | "top_k" | "top_p"
            temperature: 采样温度（仅 top_k/top_p 时生效）
            top_k: top-k 采样的 k 值
            top_p: nucleus 采样的概率阈值
        返回:
            生成的完整文本（含提示词）
        """
```

### load_trained_model

```python
def load_trained_model(
    checkpoint_path: str,
    config: ModelConfig,
    device: torch.device,
) -> StudentModel:
    """
    从检查点加载训练后的学生模型。

    参数:
        checkpoint_path: 检查点文件路径
        config: 模型配置
        device: 目标设备
    返回:
        加载权重并设置为 eval 模式的 StudentModel
    """
```

## 依赖

- `config.py`: ModelConfig
- `model.py`: StudentModel
- `data.py`: load_tokenizer

## 不做

- 不实现 KV Cache 优化（Phase 1 保持简单实现）
- 不实现 beam search（贪心 + 采样已足够验证效果）
- 不实现批量生成（单条推理即可）
- 不实现流式输出（非交互场景）
