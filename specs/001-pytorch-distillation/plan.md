# Implementation Plan: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Branch**: `001-pytorch-distillation` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-pytorch-distillation/spec.md`
**Scope**: Phase 1 only（模型架构实现）

## Summary

从零使用 PyTorch 实现一个 ~120M 参数的 Decoder-Only Transformer 模型（对齐 Qwen2.5 架构特性），使用 Qwen2.5-0.5B 作为教师模型进行在线知识蒸馏，训练数据为 Wikipedia 中文子集，运行环境为 Google Colab T4 GPU。项目以 .py 模块 + 主 Notebook 形式交付。

## Technical Context

**Language/Version**: Python 3.10+（Colab 默认环境）
**Primary Dependencies**: PyTorch 2.x, transformers（加载教师模型/Tokenizer）, datasets（加载 Wikipedia 中文）
**Storage**: 文件系统（检查点 .pt 文件，Google Drive 持久化）
**Testing**: pytest（本地单元测试），Notebook 内嵌断言（Colab 环境验证）
**Target Platform**: Google Colab（T4 GPU, 15GB VRAM, ~12GB RAM）
**Project Type**: single（ML 研究项目）
**Performance Goals**: 训练吞吐量需保证 6 小时内完成蒸馏；推理延迟无硬性要求
**Constraints**: T4 16GB VRAM 硬限制；教师模型(~1GB FP16) + 学生模型(~0.5GB) + 激活值 + 优化器状态须同时驻留显存
**Scale/Scope**: 学生模型 ~120M 参数，训练数据 ~50MB，单 GPU 单节点

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

根据SpecKit宪法检查：
- **代码整洁原则**: ✅ 通过 — 项目按功能模块拆分为独立 .py 文件，每个文件职责单一
- **架构清晰原则**: ✅ 通过 — 模型、数据、训练、生成四大模块边界清晰，依赖单向
- **注释完整原则**: ✅ 通过 — 所有模块接口需配备中文 docstring，关键算法（RoPE、GQA、SwiGLU）需行内注释
- **文档中文化原则**: ✅ 通过 — 所有设计文档使用中文撰写
- **可维护性原则**: ✅ 通过 — ModelConfig 数据类集中管理超参数，训练超参数可配置，模块间松耦合

## Project Structure

### Documentation (this feature)

```text
specs/001-pytorch-distillation/
├── spec.md              # 功能规格说明
├── plan.md              # 本文件（实现计划）
├── research.md          # Phase 0 研究产出
├── data-model.md        # Phase 1 数据模型
├── quickstart.md        # Phase 1 开发环境指南
├── contracts/           # Phase 1 模块接口契约
│   ├── model.md         # 模型模块接口
│   ├── data.md          # 数据模块接口
│   ├── trainer.md       # 训练模块接口
│   └── generate.md      # 生成模块接口
└── tasks.md             # Phase 2 任务列表（/speckit.tasks 生成）
```

### Source Code (repository root)

```text
src/
├── config.py            # ModelConfig, TrainingConfig 数据类
├── model.py             # RMSNorm, RoPE, GQA Attention, SwiGLU FFN, TransformerBlock, StudentModel
├── data.py              # WikiDataset, tokenizer 加载, DataLoader 构建
├── trainer.py           # DistillationTrainer, 蒸馏损失, 检查点管理, 指标记录
└── generate.py          # TextGenerator, 贪心/top-k/top-p 解码策略

tests/
├── test_config.py       # 配置验证测试
├── test_model.py        # 模型架构测试（前向传播、参数量、组件验证）
├── test_data.py         # 数据管道测试（编码/解码、DataLoader 输出形状）
├── test_trainer.py      # 训练循环测试（loss 下降、检查点保存/加载）
└── test_generate.py     # 生成测试（输出格式、解码策略差异）

notebooks/
└── main.ipynb           # Colab 主 Notebook（安装依赖、编排调用、可视化）
```

**Structure Decision**: 采用 Single Project 结构。`src/` 下按功能模块划分四个 .py 文件加一个配置文件，`tests/` 镜像对应，`notebooks/` 包含 Colab 入口 Notebook。模块间通过 import 显式声明依赖。

## Phase 0: Research Summary

详见 [research.md](./research.md)。核心决策：

1. **学生模型架构配置**: hidden_size=512, num_layers=12, num_heads=8, num_kv_heads=2, intermediate_size=2048, vocab_size=151936 → ~123M 参数
2. **蒸馏损失函数**: α * KL(student_logits/T, teacher_logits/T) + (1-α) * CE(student_logits, labels)，建议 α=0.5, T=2.0
3. **显存预算**: 教师模型 FP16 ~1GB + 学生模型 FP32 ~0.5GB + 优化器(AdamW) ~1GB + 激活值(batch=8, seq=512) ~4-6GB → 总计 ~8GB，留有充足余量
4. **训练超参数**: batch_size=8, lr=3e-4, warmup_steps=500, epochs根据数据量动态调整

## Phase 1: Design & Contracts

### 数据模型

详见 [data-model.md](./data-model.md)。

### 模块接口契约

详见 [contracts/](./contracts/) 目录：
- [model.md](./contracts/model.md) — 模型架构模块接口
- [data.md](./contracts/data.md) — 数据加载模块接口
- [trainer.md](./contracts/trainer.md) — 蒸馏训练模块接口
- [generate.md](./contracts/generate.md) — 文本生成模块接口

### 快速启动

详见 [quickstart.md](./quickstart.md)。

## Complexity Tracking

无宪法违规，无需追踪。
