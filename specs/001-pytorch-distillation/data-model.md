# Data Model: PyTorch 知识蒸馏

**Phase 1 产出** | **Date**: 2026-02-09

## 实体关系概览

```text
ModelConfig ──配置──→ StudentModel
                         ↑
TrainingConfig ──配置──→ DistillationTrainer ←──加载── TeacherModel (Qwen2.5-0.5B)
                              ↑
WikiDataset ──提供数据──→ DataLoader
                              ↓
                     TextGenerator ←──加载权重── StudentModel (训练后)
```

## 实体定义

### ModelConfig

模型架构超参数的不可变配置对象。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| hidden_size | int | 512 | 隐藏层维度 |
| num_layers | int | 12 | Transformer 层数 |
| num_heads | int | 8 | 注意力头数 |
| num_kv_heads | int | 2 | KV 头数（GQA） |
| intermediate_size | int | 2048 | SwiGLU FFN 中间维度 |
| vocab_size | int | 151936 | 词表大小（与 Qwen2.5 一致） |
| max_seq_len | int | 512 | 最大序列长度 |
| rope_theta | float | 1000000.0 | RoPE 旋转基数 |
| norm_eps | float | 1e-6 | RMSNorm epsilon |
| dropout | float | 0.0 | Dropout 率（蒸馏时通常为 0） |

**验证规则**:
- hidden_size 必须能被 num_heads 整除
- num_heads 必须能被 num_kv_heads 整除
- vocab_size 必须 > 0
- max_seq_len 必须 > 0

### TrainingConfig

训练超参数配置对象。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| batch_size | int | 8 | 批大小 |
| learning_rate | float | 3e-4 | 初始学习率 |
| weight_decay | float | 0.01 | 权重衰减 |
| warmup_steps | int | 500 | 学习率预热步数 |
| num_epochs | int | 3 | 训练轮数 |
| gradient_clip | float | 1.0 | 梯度裁剪阈值 |
| alpha | float | 0.5 | 蒸馏损失权重（vs 标签损失） |
| temperature | float | 2.0 | 蒸馏温度 |
| checkpoint_dir | str | "checkpoints/" | 检查点保存路径 |
| log_interval | int | 50 | 日志打印步数间隔 |
| eval_interval | int | 500 | 验证评估步数间隔 |
| save_interval | int | 1000 | 检查点保存步数间隔 |

### StudentModel

从零实现的 Decoder-Only Transformer 模型。

**组件结构**:
- `embedding`: nn.Embedding(vocab_size, hidden_size)
- `layers`: nn.ModuleList of TransformerBlock × num_layers
  - 每个 TransformerBlock 包含:
    - `attention_norm`: RMSNorm(hidden_size)
    - `attention`: GQAAttention(hidden_size, num_heads, num_kv_heads)
    - `ffn_norm`: RMSNorm(hidden_size)
    - `ffn`: SwiGLUFFN(hidden_size, intermediate_size)
- `norm`: RMSNorm(hidden_size) — 最终归一化
- `lm_head`: nn.Linear(hidden_size, vocab_size, bias=False) — 与 embedding 权重共享

**状态转换**:
- 初始化 → 前向传播就绪
- 训练中 → 参数更新中
- 检查点保存 → 可恢复状态
- 训练完成 → 推理模式（eval）

### TeacherModel

外部加载的 Qwen2.5-0.5B 预训练模型。

| 属性 | 说明 |
|------|------|
| model_id | "Qwen/Qwen2.5-0.5B" |
| precision | FP16 (torch.float16) |
| mode | eval only, requires_grad=False |
| 显存占用 | ~1GB |

**约束**: 教师模型在整个训练过程中冻结参数，仅做前向传播提供 logits。

### WikiDataset

Wikipedia 中文子集数据集。

| 属性 | 说明 |
|------|------|
| source | HuggingFace `wikipedia` / `20231101.zh` |
| sample_size | ~50MB 文本 |
| split_ratio | 训练 90% / 验证 10% |
| tokenizer | Qwen2.5-0.5B AutoTokenizer |
| seq_len | 512 tokens |
| format | input_ids (causal LM: labels = input_ids shifted right) |

**数据处理流水线**:
1. 加载原始文本 → 2. Tokenizer 编码 → 3. 文本拼接 → 4. 按 seq_len 分块 → 5. 构造 input_ids/labels 对

### DistillationTrainer

管理完整训练生命周期。

| 状态 | 说明 |
|------|------|
| initialized | 模型、优化器、数据加载器就绪 |
| training | 训练循环执行中 |
| evaluating | 验证集评估中 |
| checkpointing | 保存/加载检查点 |
| completed | 训练结束 |

**检查点内容**:
- student_model.state_dict()
- optimizer.state_dict()
- scheduler.state_dict()
- current_epoch, current_step
- best_val_loss
- training_config

### TextGenerator

加载训练后的学生模型进行自回归文本生成。

| 属性 | 说明 |
|------|------|
| model | 加载了训练权重的 StudentModel (eval mode) |
| tokenizer | Qwen2.5-0.5B AutoTokenizer |
| strategies | greedy, top_k, top_p |
| max_new_tokens | 可配置，默认 256 |
