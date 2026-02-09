# Feature Specification: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Feature Branch**: `001-pytorch-distillation`
**Created**: 2026-02-09
**Status**: Draft
**Input**: User description: "我需要从头使用pytorch编写 qwen/deepseek 模型，并使用小量数据训练蒸馏，环境可以使用colab"

## Clarifications

### Session 2026-02-09

- Q: 使用哪个具体的教师模型？ → A: Qwen2.5-0.5B（显存友好，可在线蒸馏）
- Q: 蒸馏方式：在线蒸馏还是离线缓存？ → A: 在线蒸馏（训练时同时加载教师和学生，实时计算教师 logits）
- Q: 使用哪个具体的训练数据集？ → A: Wikipedia 中文子集（HuggingFace 直接加载，高质量结构化语料）
- Q: 项目交付格式：Notebook 还是模块化文件？ → A: .py 模块 + 主 Notebook（核心代码为独立 .py 文件，Notebook 负责编排调用）
- Q: 默认最大序列长度？ → A: 512 tokens（平衡显存和上下文长度）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 从零构建 Transformer 模型架构 (Priority: P1)

作为一名深度学习研究者，我希望能够从零使用 PyTorch 实现一个与 Qwen/DeepSeek 架构对齐的小型 Transformer 语言模型，以便深入理解模型内部工作原理并获得完全可控的模型代码。

**Why this priority**: 模型架构是整个项目的基础，没有模型就无法进行后续的训练和蒸馏。这是最小可行产品的核心交付物。

**Independent Test**: 可以通过构建模型、输入随机张量并验证输出形状正确来独立测试。模型能够完成前向传播即证明架构正确。

**Acceptance Scenarios**:

1. **Given** 已实现的模型代码, **When** 输入一组随机 token ID 序列, **Then** 模型输出与词表大小匹配的 logits 张量，形状为 (batch_size, seq_len, vocab_size)
2. **Given** 模型实例化完成, **When** 打印模型参数量, **Then** 参数量约为 120M（小型模型）
3. **Given** 模型代码, **When** 检查架构组件, **Then** 包含多头注意力（支持 GQA）、RMSNorm、旋转位置编码（RoPE）、SwiGLU FFN 等核心组件

---

### User Story 2 - 数据准备与 Tokenizer 集成 (Priority: P2)

作为一名研究者，我希望能够加载和处理小规模中文/英文文本数据集，并使用与教师模型兼容的 Tokenizer 进行编码，以便为蒸馏训练提供高质量的训练数据。

**Why this priority**: 数据是训练的燃料，Tokenizer 兼容性直接影响蒸馏效果。必须在训练前解决。

**Independent Test**: 可以通过加载数据集、对文本进行编码/解码并验证 token 序列正确性来独立测试。

**Acceptance Scenarios**:

1. **Given** 一个小规模文本数据集（约 10MB-100MB）, **When** 使用 Tokenizer 进行编码, **Then** 所有文本被正确转换为 token ID 序列且能无损解码还原
2. **Given** 编码后的数据, **When** 构建 DataLoader, **Then** 能够按批次输出固定长度的 token 序列，支持因果语言建模格式
3. **Given** 数据加载管道, **When** 在 Colab 环境运行, **Then** 数据加载不会成为训练瓶颈，GPU 利用率保持在 80% 以上

---

### User Story 3 - 知识蒸馏训练 (Priority: P3)

作为一名研究者，我希望使用预训练的 Qwen/DeepSeek 教师模型对我从零构建的学生模型进行知识蒸馏训练，使学生模型学习到教师模型的知识表示能力。

**Why this priority**: 蒸馏训练是项目的核心目标，但依赖模型架构和数据准备两个前置步骤。

**Independent Test**: 可以通过观察训练 loss 下降曲线、比较蒸馏前后模型在验证集上的困惑度（perplexity）来独立验证蒸馏效果。

**Acceptance Scenarios**:

1. **Given** 学生模型和教师模型, **When** 执行蒸馏训练, **Then** 蒸馏损失（KL 散度 + 交叉熵的加权组合）持续下降
2. **Given** 蒸馏训练完成, **When** 在验证集上评估, **Then** 学生模型的困惑度显著低于随机初始化后直接训练的基线
3. **Given** Colab 免费 GPU 环境（T4 16GB）, **When** 执行完整蒸馏流程, **Then** 训练能够在合理时间内完成（数小时级别）而不会 OOM

---

### User Story 4 - 模型推理与文本生成 (Priority: P4)

作为一名研究者，我希望使用蒸馏训练后的学生模型进行文本生成推理，验证模型学到了有意义的语言能力。

**Why this priority**: 推理是验证整个蒸馏流程有效性的最终手段，优先级低于核心训练流程。

**Independent Test**: 可以通过给定提示词生成文本并人工评估连贯性来独立测试。

**Acceptance Scenarios**:

1. **Given** 蒸馏训练完成的模型, **When** 输入一段中文/英文提示词, **Then** 模型生成语法基本正确、语义连贯的续写文本
2. **Given** 推理代码, **When** 使用不同解码策略（贪心、top-k、top-p）, **Then** 生成文本质量和多样性有明显差异

---

### Edge Cases

- 当训练数据量过小（<1MB）时，模型可能严重过拟合，需要早停（early stopping）机制
- 当 Colab 运行时会话中断时，需要从检查点恢复训练而不丢失进度
- 当教师模型（Qwen2.5-0.5B）因意外显存压力无法加载时，需降低 batch size 或序列长度作为回退策略
- 当梯度出现 NaN/Inf 时，训练循环需要能够检测并记录异常而非静默失败
- 当 Tokenizer 遇到未知字符时，需要有合理的回退处理策略

## Requirements *(mandatory)*

遵从SpecKit宪法原则：
- 代码必须保持整洁，格式统一，遵循一致的编码风格
- 系统架构必须逻辑清晰，模块划分合理
- 所有关键代码必须配备完整注释
- 所有技术文档必须使用中文编写
- 代码设计必须考虑长期维护需求

### Functional Requirements

- **FR-001**: 系统必须实现完整的 Transformer Decoder-Only 模型架构，包含多头注意力机制（支持 Grouped-Query Attention）、RMSNorm 归一化、旋转位置编码（RoPE）和 SwiGLU 前馈网络
- **FR-002**: 系统必须支持可配置的模型超参数，包括层数、隐藏维度、注意力头数、KV 头数、词表大小、最大序列长度（默认 512 tokens）等
- **FR-003**: 系统必须提供数据加载管道，支持从文本文件或 HuggingFace 数据集加载数据，并转换为因果语言建模所需的训练格式
- **FR-004**: 系统必须集成与教师模型兼容的 Tokenizer，确保学生模型和教师模型使用相同的 token 编码方案
- **FR-005**: 系统必须实现知识蒸馏训练循环，支持 KL 散度损失和标准交叉熵损失的加权组合
- **FR-006**: 系统必须采用在线蒸馏方式，训练时同时加载教师模型（Qwen2.5-0.5B）和学生模型，实时计算教师 logits 用于蒸馏损失
- **FR-007**: 系统必须实现模型检查点的保存与加载功能，支持从中断处恢复训练
- **FR-008**: 系统必须提供文本生成推理功能，支持贪心搜索、top-k 采样和 top-p（nucleus）采样策略
- **FR-009**: 系统必须记录训练指标（loss、learning rate、perplexity），支持通过日志或可视化工具监控训练进度
- **FR-010**: 系统必须能在 Google Colab 免费 GPU 环境（T4 16GB VRAM）中完整运行，学生模型目标规模约 120M 参数（小型模型，类似 GPT-2 Small），不超出显存限制

### Key Entities

- **模型配置（ModelConfig）**: 定义模型架构的所有超参数——层数、维度、头数、词表大小、最大序列长度（默认 512）等
- **学生模型（StudentModel）**: 从零实现的小型 Transformer Decoder-Only 模型，是蒸馏训练的目标
- **教师模型（TeacherModel）**: 预训练的 Qwen2.5-0.5B 模型，作为知识蒸馏的知识来源
- **训练数据集（TrainingDataset）**: Wikipedia 中文子集（约 50MB 采样），经 Tokenizer 编码后用于训练
- **蒸馏训练器（DistillationTrainer）**: 管理训练循环、损失计算、优化器状态和检查点保存
- **文本生成器（TextGenerator）**: 加载训练后的模型权重，执行自回归文本生成

### Assumptions

- 教师模型使用 Qwen2.5-0.5B，FP16 加载约 1GB 显存，可与学生模型同时在线加载
- 训练数据使用 Wikipedia 中文子集，通过 HuggingFace Datasets 加载，取约 50MB 规模的采样
- 使用 HuggingFace Transformers 库加载教师模型和 Tokenizer（不从零实现 Tokenizer）
- Colab 免费版 T4 GPU（15GB VRAM）为硬性约束，所有组件必须在此限制内运行
- 蒸馏训练的目标是让学生模型学习通用语言建模能力，而非特定下游任务
- 项目产出为 .py 模块文件（模型、训练器、数据加载、生成）加一个主 Notebook 编排调用，便于在 Colab 中运行且核心组件可独立测试

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 学生模型架构从零实现后，能够对任意输入 token 序列完成前向传播并输出正确形状的 logits
- **SC-002**: 蒸馏训练完成后，学生模型在验证集上的困惑度（perplexity）比随机初始化直接训练的基线至少降低 30%
- **SC-003**: 完整蒸馏训练流程（含数据准备、在线蒸馏训练）能在 Colab 免费 GPU 环境中 6 小时内完成
- **SC-004**: 蒸馏后的学生模型能够根据提示词生成语法正确、语义连贯的文本续写（中文或英文）
- **SC-005**: 所有代码以模块化方式组织，核心组件（模型、训练器、数据加载、生成）可独立测试
