# 优化模块 API（MLA / MTP / MoE）

本仓库提供一套可插拔的优化实现与统一配置接口，目标是:

- 默认保持向后兼容（优化默认关闭）
- 允许通过 `OptimizationProfile` 按需启用 MLA/MTP/MoE
- 为性能基准与回归测试提供可复现的接口

## 统一配置

入口: `src/optimizers/config_manager.py`

核心类型: `OptimizationProfile`

- `mla`: `src/optimizers/mla/config.py::MLAConfig`
- `mtp`: `src/optimizers/mtp/config.py::MTPConfig`
- `moe`: `src/optimizers/moe/config.py::MoEConfig`

## 模型集成

`src/model.py::StudentModel` 支持可选参数:

```python
from src.config import ModelConfig
from src.model import StudentModel
from src.optimizers.config_manager import OptimizationProfile
from src.optimizers.mla import MLAConfig
from src.optimizers.moe import MoEConfig

cfg = ModelConfig()
profile = OptimizationProfile(
    mla=MLAConfig(enabled=True, latent_dim=cfg.head_dim // 2),
    mtp=None,
    moe=MoEConfig(enabled=True, num_experts=8, top_k=2),
)
model = StudentModel(cfg, optimization=profile)
```

说明:

- MLA 将注意力的 K/V 表示压缩到 latent 维度（当前实现为基础版）。
- MoE 在 FFN 位置提供 token-level top-k routing（当前实现为基础版）。
- MTP 当前提供预测头与损失接口，生成侧策略后续可扩展。

