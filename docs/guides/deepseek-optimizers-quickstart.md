# DeepSeek 优化支持快速开始

## 目标

通过统一的 `OptimizationProfile` 启用:

- MLA: 降低长序列注意力内存占用（通过 latent KV 表示）
- MTP: 多 token 预测头（用于减少步数的生成策略预留接口）
- MoE: 基础专家混合（稀疏激活 + 路由统计）

## 快速运行

在已安装依赖的环境中:

```bash
pip install -r requirements.txt
pytest
```

## 启用优化（示例）

```python
import torch

from src.config import ModelConfig
from src.model import StudentModel
from src.optimizers.config_manager import OptimizationProfile
from src.optimizers.mla import MLAConfig
from src.optimizers.moe import MoEConfig

cfg = ModelConfig(hidden_size=64, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000)
profile = OptimizationProfile(
    mla=MLAConfig(enabled=True, latent_dim=cfg.head_dim // 2),
    mtp=None,
    moe=MoEConfig(enabled=True, num_experts=4, top_k=2),
)

model = StudentModel(cfg, optimization=profile).eval()
input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(input_ids)
print(logits.shape)
```

