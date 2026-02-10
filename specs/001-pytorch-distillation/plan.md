# Implementation Plan: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Branch**: `001-pytorch-distillation` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-pytorch-distillation/spec.md`
**Scope**: 完整（US1 模型架构 + US2 数据准备 + US3 蒸馏训练 + US4 推理生成）

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

## Phase 2: User Story 2 — 数据准备与 Tokenizer 集成

### 设计要点

**数据流**: HuggingFace `wikipedia/20231101.zh` → 文本提取 → Tokenizer 编码 → 拼接 → 按 seq_len=512 分块 → input_ids/labels 对 → DataLoader

**关键设计决策**:

1. **Tokenizer 来源**: 直接复用 Qwen2.5-0.5B 的 AutoTokenizer（vocab_size=151936），无需训练自定义 Tokenizer
2. **数据预处理策略**: 文本拼接法（concatenate-then-chunk）— 将所有文章 token 拼接为一维长序列，再按固定 seq_len 切分，避免短文本填充浪费
3. **Labels 构造**: 因果语言建模标准做法，labels = input_ids 右移一位，首位填充 -100（cross_entropy 会忽略该位置）
4. **数据集拆分**: 90% 训练 / 10% 验证，通过取前 N 条原始文章控制总量在 ~50MB
5. **pad_token 处理**: Qwen2.5 Tokenizer 默认无 pad_token，需设置 pad_token = eos_token

### 显存与性能注意事项

- 数据预处理在 CPU 上完成，不占用 GPU 显存
- DataLoader 使用 num_workers=2（Colab 限制），pin_memory=True
- 预处理后的数据集缓存到内存（~50MB token 数据完全放得下）

### 与 Phase 1 的接口

- **依赖 Phase 1**: ModelConfig.max_seq_len 和 ModelConfig.vocab_size 决定数据分块长度和 Tokenizer 兼容性
- **下游消费者**: Phase 3 (DistillationTrainer) 的 train_loader/val_loader

## Phase 3: User Story 3 — 知识蒸馏训练

### 设计要点

**训练流**: 加载教师模型(FP16) → 加载学生模型(FP32) → 训练循环(在线蒸馏) → 检查点保存 → 验证评估

**关键设计决策**:

1. **蒸馏损失**: L = α·T²·KL(softmax(s/T) ‖ softmax(t/T)) + (1-α)·CE(s, labels)，α=0.5, T=2.0
2. **教师模型加载**: AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=float16)，设 eval() 并冻结全部参数
3. **优化器**: AdamW(lr=3e-4, weight_decay=0.01) + cosine scheduler with warmup
4. **检查点策略**: 每 save_interval 步保存完整状态（模型、优化器、scheduler、epoch/step、best_val_loss）
5. **梯度安全**: gradient clipping=1.0，NaN/Inf 检测与警告日志
6. **评估指标**: 验证集 loss + perplexity (ppl = exp(loss))

### 显存运行时分析

教师模型前向传播使用 `torch.no_grad()` 且 FP16，不产生计算图。学生模型 FP32 训练。总显存预计 ~9-11GB，T4 15GB 留有余量。

### 与前置 Phase 的接口

- **依赖 Phase 1**: StudentModel (model.py)
- **依赖 Phase 2**: train_loader, val_loader (data.py), load_tokenizer
- **下游消费者**: Phase 4 (TextGenerator) 加载训练后的检查点

## Phase 4: User Story 4 — 模型推理与文本生成

### 设计要点

**推理流**: 加载检查点 → 构建 StudentModel(eval) → 编码 prompt → 自回归生成 → 解码输出

**关键设计决策**:

1. **解码策略**: 支持 greedy（argmax）、top-k（筛选前 k 个 token 再采样）、top-p/nucleus（累积概率阈值采样）
2. **温度控制**: temperature 参数缩放 logits，<1 更确定，>1 更随机（仅采样策略生效）
3. **停止条件**: 达到 max_new_tokens 或生成 eos_token
4. **模型加载**: 从检查点中提取 model_state_dict，加载到新建的 StudentModel 实例

### 与前置 Phase 的接口

- **依赖 Phase 1**: StudentModel, ModelConfig
- **依赖 Phase 2**: load_tokenizer
- **依赖 Phase 3**: 训练后的检查点文件

## Phase 5: 集成与 Notebook

### Colab 主 Notebook (notebooks/main.ipynb)

**Cell 顺序**:
1. 环境检查 & pip install
2. 从 src/ 导入所有模块
3. 加载 Tokenizer & 构建数据集
4. 构建学生模型 & 打印参数量
5. 加载教师模型 & 显存检查
6. 执行蒸馏训练 & 绘制 loss 曲线
7. 文本生成演示（多种解码策略对比）
8. 保存最终模型到 Google Drive

## Complexity Tracking

无宪法违规，无需追踪。
