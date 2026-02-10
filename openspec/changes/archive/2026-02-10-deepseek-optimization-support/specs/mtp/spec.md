# MTP (Multi-token Prediction) 规格

## ADDED Requirements

### Requirement: MTP 可通过配置启用并设置预测深度
系统 MUST 支持通过统一配置启用/禁用 MTP，并配置每步预测的 token 数（预测深度）及相关权重/策略参数。

#### Scenario: 启用 MTP 并设置预测 4 tokens
- **WHEN** 用户在配置中启用 MTP 并设置预测深度为 4
- **THEN** 系统在生成过程中按每步预测 4 个未来 token 的策略执行

### Requirement: MTP MUST 提升批处理推理吞吐
在批处理推理场景下，系统 MUST 通过 MTP 减少自回归迭代步数，从而提升整体吞吐或降低总耗时。

#### Scenario: batch_size=8 推理时间下降达标
- **WHEN** 在 batch_size=8 的推理任务中启用 MTP（预测深度=4）并与基础自回归生成对比
- **THEN** 总推理时间相对基础实现降低至少 25%，或等价地吞吐量提升至少 30%

### Requirement: MTP MUST 保持接口兼容并可回退
启用 MTP 后，系统 MUST 保持对外生成接口兼容，并允许通过配置快速回退到基础自回归路径。

#### Scenario: 关闭 MTP 回退到基础生成
- **WHEN** 用户在配置中关闭 MTP
- **THEN** 系统使用基础自回归生成路径完成推理且不依赖 MTP 专用模块

