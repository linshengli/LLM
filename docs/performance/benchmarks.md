# 性能基准说明

本仓库的优化能力包含指标型目标（例如 MLA 内存节省、MTP 吞吐提升）。

当前阶段提供:

- `tests/integration/test_mla_memory_efficiency.py`:
  - CUDA 可用时对比 `torch.cuda.max_memory_allocated()` 的峰值显存
  - CPU 环境下跳过显存断言（仅作为开发环境兼容）
- `tests/integration/test_mtp_speed_benchmark.py`:
  - 使用“步数调用减少”的代理基准（后续可替换为真实生成吞吐）
- `tests/integration/test_moe_parameter_efficiency.py`:
  - 使用稀疏激活比例（top_k/num_experts）作为代理指标

后续建议:

- 固定 GPU 型号、驱动、PyTorch/Transformers 版本
- 固定 seq_len、batch_size、dtype、seed
- 将基准结果以报告形式输出到 CI 工件

