# Research: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Phase 0 产出** | **Date**: 2026-02-09

## R1: 学生模型架构参数配置

**Decision**: 采用以下配置实现 ~123M 参数的学生模型

| 超参数 | 学生模型 | Qwen2.5-0.5B（教师） | 说明 |
|--------|----------|---------------------|------|
| hidden_size | 512 | 896 | 缩小至教师的 57% |
| num_layers | 12 | 24 | 缩小至教师的 50% |
| num_heads | 8 | 14 | 保持 head_dim=64 |
| num_kv_heads | 2 | 2 | 与教师一致，保留 GQA 结构 |
| intermediate_size | 2048 | 4864 | SwiGLU FFN 中间维度 |
| vocab_size | 151936 | 151936 | 与教师完全一致（共享 Tokenizer） |
| max_seq_len | 512 | 32768 | 受 Colab 显存限制 |
| rope_theta | 1000000.0 | 1000000.0 | 与教师一致 |

**参数量估算**:
- Embedding: 151936 × 512 = 77.8M（权重共享 lm_head，不重复计算）
- 每层 Attention (GQA): Q(512×512) + K(512×128) + V(512×128) + O(512×512) = 0.66M
- 每层 SwiGLU FFN: gate(512×2048) + up(512×2048) + down(2048×512) = 3.15M
- 每层 RMSNorm × 2: 1024（可忽略）
- 12 层合计: 12 × (0.66M + 3.15M) = 45.7M
- 最终 RMSNorm: 512（可忽略）
- **总计**: 77.8M + 45.7M ≈ **123.5M 参数**

**Rationale**: 保持与 Qwen2.5 相同的架构特性（GQA、RoPE、SwiGLU），仅缩小规模。共享词表确保蒸馏时 logits 空间对齐。

**Alternatives considered**:
- hidden_size=384 + 更多层: 词表嵌入层占比过大，模型表达能力分配不均
- hidden_size=768 + 6 层: 嵌入层 ~117M 已接近总预算，Transformer 层过浅

## R2: 蒸馏损失函数设计

**Decision**: 使用标准 logit-level 蒸馏损失

```
L = α * T² * KL(softmax(s/T) || softmax(t/T)) + (1 - α) * CE(s, labels)
```

- α（蒸馏权重）: 0.5（蒸馏损失与标准语言建模损失等权）
- T（温度）: 2.0（软化 logits 分布，暴露更多教师知识）
- KL 散度使用 `F.kl_div(log_softmax(s/T), softmax(t/T))`
- T² 缩放因子补偿温度对梯度的缩放效应

**Rationale**: 这是 Hinton et al. (2015) 经典蒸馏方法，简单有效，适合首次实现。α=0.5 在小数据场景下提供足够的标签监督信号防止过拟合。

**Alternatives considered**:
- 纯 KL 蒸馏（无标签损失）: 小数据场景下容易发散
- Feature-level 蒸馏（中间层匹配）: 实现复杂度高，且教师/学生维度不同需额外投影层
- CKD/GKD 等高级方法: 超出当前学习目标范围

## R3: 显存预算分析

**Decision**: batch_size=8, seq_len=512 可在 T4 上安全运行

| 组件 | 估算显存 | 说明 |
|------|----------|------|
| 教师模型（FP16, eval） | ~1.0 GB | 0.5B 参数 × 2 bytes |
| 学生模型（FP32, train） | ~0.5 GB | 123M 参数 × 4 bytes |
| AdamW 优化器状态 | ~1.0 GB | 2 × 学生参数 × 4 bytes (m, v) |
| 梯度 | ~0.5 GB | 学生参数 × 4 bytes |
| 学生激活值 | ~3-4 GB | batch=8, seq=512, 12 layers |
| 教师前向激活值 | ~2-3 GB | batch=8, seq=512, 24 layers (no_grad) |
| PyTorch 开销 | ~1 GB | CUDA context, fragmentation |
| **总计** | **~9-11 GB** | T4 15GB，余量 4-6GB |

**Rationale**: 留有 30-40% 显存余量，可应对碎片和峰值。若显存紧张，可先降 batch_size 至 4。

## R4: 训练超参数

**Decision**: 采用以下训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| optimizer | AdamW | 标准选择，支持权重衰减 |
| learning_rate | 3e-4 | 小模型标准学习率 |
| weight_decay | 0.01 | 轻度正则化 |
| warmup_steps | 500 | 学习率线性预热 |
| scheduler | cosine | 余弦退火至 lr_min=1e-5 |
| batch_size | 8 | 受显存约束 |
| gradient_clip | 1.0 | 防止梯度爆炸 |
| mixed_precision | 否 | 学生模型较小，FP32 训练即可 |
| epochs | 根据数据量动态 | 约 50MB 数据，预计 3-5 epochs |

**Rationale**: 保守配置，优先稳定性。小模型不需要混合精度训练，FP32 更易调试。

## R5: 数据集与 Tokenizer

**Decision**: 使用 `wikipedia` 数据集的 `20231101.zh` 版本子集

- HuggingFace 数据集 ID: `wikipedia`, config `20231101.zh`
- 取前 N 条记录（约 50MB 文本量）
- 按 90/10 比例拆分训练集/验证集
- Tokenizer: `Qwen/Qwen2.5-0.5B` 的 AutoTokenizer
- 数据预处理: 文本拼接 → 按 512 tokens 分块 → 构造 input_ids 和 labels（右移一位）

**Rationale**: Wikipedia 中文质量高、结构化好，HuggingFace 直接提供 streaming 和分片下载。Qwen2.5 Tokenizer 原生支持中英文。
