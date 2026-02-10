# Tasks: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Input**: Design documents from `/specs/001-pytorch-distillation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Scope**: 完整 — US1（模型架构）+ US2（数据准备）+ US3（蒸馏训练）+ US4（推理生成）+ 集成 Notebook

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目初始化和基础结构搭建

- [x] T001 创建项目目录结构：`src/`, `tests/`, `notebooks/`，按 plan.md 中的 Source Code 布局创建所有目录
- [x] T002 创建 `requirements.txt`，包含依赖：torch, transformers, datasets, pytest
- [x] T003 [P] 创建 `src/__init__.py` 和 `tests/__init__.py`，确保 Python 包结构正确

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 配置数据类，所有模块共享的基础设施

**⚠️ CRITICAL**: 所有 User Story 的实现依赖此阶段的配置类

- [x] T004 [SETUP] 实现 `ModelConfig` 数据类 in `src/config.py`
  - 字段：hidden_size=512, num_layers=12, num_heads=8, num_kv_heads=2, intermediate_size=2048, vocab_size=151665, max_seq_len=512, rope_theta=1e6, norm_eps=1e-6, dropout=0.0
  - 验证规则：hidden_size % num_heads == 0, num_heads % num_kv_heads == 0
  - 参考：data-model.md ModelConfig 实体定义
  - **注意**: vocab_size 从研究阶段的 151936 更正为实际值 151665（Qwen2.5-0.5B Tokenizer 实测）
- [x] T005 [P] [SETUP] 实现 `TrainingConfig` 数据类 in `src/config.py`
  - 字段：batch_size=8, learning_rate=3e-4, weight_decay=0.01, warmup_steps=500, num_epochs=3, gradient_clip=1.0, alpha=0.5, temperature=2.0, checkpoint_dir, log_interval=50, eval_interval=500, save_interval=1000
  - 参考：data-model.md TrainingConfig 实体定义

**Checkpoint**: 配置基础就绪，所有 User Story 可以开始

---

## Phase 3: User Story 1 - 从零构建 Transformer 模型架构 (Priority: P1) 🎯 MVP

**Goal**: 从零使用 PyTorch 实现一个 ~120M 参数的 Decoder-Only Transformer 模型，对齐 Qwen2.5 架构特性

**Independent Test**: 构建模型 → 输入随机 token ID → 验证输出 logits 形状为 (batch, seq_len, vocab_size)，参数量约 123M

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T006 [P] [US1] 编写配置验证测试 in `tests/test_config.py`
  - 测试 ModelConfig 默认值正确性
  - 测试 hidden_size % num_heads != 0 时抛出 ValueError
  - 测试 num_heads % num_kv_heads != 0 时抛出 ValueError
- [x] T007 [P] [US1] 编写模型架构测试 in `tests/test_model.py`
  - 测试 RMSNorm: 输入输出形状一致，归一化后均值接近 0
  - 测试 RotaryEmbedding: 输出形状不变，不同位置编码不同
  - 测试 GQAAttention: 输入 (batch, seq, hidden) → 输出形状一致
  - 测试 SwiGLUFFN: 输入 (batch, seq, hidden) → 输出形状一致
  - 测试 TransformerBlock: 输入输出形状一致，残差连接有效
  - 测试 StudentModel: input_ids (batch, seq) → logits (batch, seq, vocab_size)
  - 测试 StudentModel.count_parameters() ≈ 123M (±5%)
  - 测试 lm_head 与 embedding 权重共享（是同一个张量对象）

### Implementation for User Story 1

- [x] T008 [P] [US1] 实现 `RMSNorm` in `src/model.py`
  - Root Mean Square Layer Normalization
  - 参数：可学习的缩放权重 gamma (hidden_size,)
  - 公式：x * rsqrt(mean(x²) + eps) * gamma
  - 参考：contracts/model.md RMSNorm 接口
- [x] T009 [P] [US1] 实现 `RotaryEmbedding` in `src/model.py`
  - 预计算旋转频率矩阵 freqs = 1 / (theta^(2i/dim))
  - 根据 position_ids 生成 cos/sin 位置编码
  - 对 Q/K 张量的前半/后半维度应用旋转变换
  - 参考：contracts/model.md RotaryEmbedding 接口
- [x] T010 [US1] 实现 `GQAAttention` in `src/model.py`（depends on T008, T009）
  - Q 投影: hidden_size → num_heads * head_dim
  - K/V 投影: hidden_size → num_kv_heads * head_dim
  - KV 头扩展（repeat_kv）：将 num_kv_heads 扩展到 num_heads
  - 缩放点积注意力 + 因果遮罩（上三角 -inf mask）
  - 对 Q、K 应用 RoPE
  - O 投影: num_heads * head_dim → hidden_size
  - 参考：contracts/model.md GQAAttention 接口
- [x] T011 [P] [US1] 实现 `SwiGLUFFN` in `src/model.py`
  - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
  - up_proj: Linear(hidden_size, intermediate_size, bias=False)
  - down_proj: Linear(intermediate_size, hidden_size, bias=False)
  - 公式：down_proj(SiLU(gate_proj(x)) * up_proj(x))
  - 参考：contracts/model.md SwiGLUFFN 接口
- [x] T012 [US1] 实现 `TransformerBlock` in `src/model.py`（depends on T008, T010, T011）
  - Pre-norm 架构：norm → attention → residual → norm → ffn → residual
  - 参考：contracts/model.md TransformerBlock 接口
- [x] T013 [US1] 实现 `StudentModel` in `src/model.py`（depends on T012）
  - embedding → layers × N → final_norm → lm_head
  - **权重共享**: lm_head.weight = embedding.weight
  - 自动生成 position_ids 和因果 attention_mask
  - count_parameters() 方法
  - 参考：contracts/model.md StudentModel 接口
- [x] T014 [US1] 运行 `tests/test_model.py` 和 `tests/test_config.py` 验证所有测试通过

**Checkpoint**: User Story 1 完成。学生模型架构从零实现，能够接受 token 输入并输出正确形状的 logits。

---

## Phase 4: User Story 2 - 数据准备与 Tokenizer 集成 (Priority: P2)

**Goal**: 加载 Wikipedia 中文子集，使用 Qwen2.5 Tokenizer 编码，构建因果语言建模格式的 DataLoader

**Independent Test**: 加载数据 → Tokenizer 编码/解码 → 验证 DataLoader 输出形状为 (batch, seq_len)

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T015 [P] [US2] 编写数据管道测试 in `tests/test_data.py`
  - 测试 load_tokenizer 返回有效 Tokenizer 且 vocab_size=151936
  - 测试 Tokenizer 的 pad_token 已设置
  - 测试 WikiDataset.__len__() > 0
  - 测试 WikiDataset.__getitem__() 返回 dict 含 "input_ids" 和 "labels"
  - 测试 input_ids shape = (seq_len,), dtype = torch.long
  - 测试 labels[0] == -100（首位忽略标记）
  - 测试 labels[1:] == input_ids[:-1] 的右移关系
  - 测试 create_dataloaders 返回两个 DataLoader
  - 测试 DataLoader 输出 batch shape = (batch_size, seq_len)

### Implementation for User Story 2

- [x] T016 [US2] 实现 `load_tokenizer()` in `src/data.py`
  - 使用 AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
  - 设置 pad_token = eos_token（Qwen2.5 默认无 pad_token）
  - 参考：contracts/data.md load_tokenizer 接口
- [x] T017 [US2] 实现 `WikiDataset` in `src/data.py`（depends on T016）
  - 从 HuggingFace 加载 `wikipedia` 数据集，config `20231101.zh`
  - 提取 "text" 字段，使用 tokenizer 批量编码
  - 拼接所有 token 为一维长序列（concatenate-then-chunk 策略）
  - 按 seq_len 分块为等长样本
  - 构造 labels：input_ids 右移一位，首位填充 -100
  - 支持 max_samples 参数控制数据量（~50MB）
  - 支持 "train"/"validation" 拆分（90/10）
  - 参考：contracts/data.md WikiDataset 接口
- [x] T018 [US2] 实现 `create_dataloaders()` in `src/data.py`（depends on T017）
  - 创建 train/val WikiDataset 实例
  - 构建 DataLoader：shuffle(train=True, val=False), num_workers=2, pin_memory=True
  - 参考：contracts/data.md create_dataloaders 接口
- [x] T019 [US2] 运行 `tests/test_data.py` 验证所有测试通过

**Checkpoint**: User Story 2 完成。数据管道就绪，能输出正确格式的训练/验证 batch。

---

## Phase 5: User Story 3 - 知识蒸馏训练 (Priority: P3)

**Goal**: 使用 Qwen2.5-0.5B 教师模型对学生模型进行在线知识蒸馏，训练 loss 持续下降，验证集 perplexity 优于基线

**Independent Test**: 执行少量 step 训练 → 验证 loss 下降 → 保存/加载检查点 → 验证恢复正确

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T020 [P] [US3] 编写训练模块测试 in `tests/test_trainer.py`
  - 测试 distillation_loss 返回 (total_loss, metrics_dict)
  - 测试 total_loss 为标量张量且 > 0
  - 测试 metrics_dict 包含 "kl_loss", "ce_loss", "total_loss"
  - 测试 alpha=0 时 total_loss ≈ ce_loss
  - 测试 alpha=1 时 total_loss ≈ T²·kl_loss
  - 测试 load_teacher_model 返回冻结的模型（requires_grad=False）
  - 测试 DistillationTrainer.save_checkpoint/load_checkpoint 往返一致
  - 测试 DistillationTrainer 训练 10 步后 loss 下降

### Implementation for User Story 3

- [x] T021 [US3] 实现 `distillation_loss()` in `src/trainer.py`
  - KL 散度：F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction="batchmean")
  - 交叉熵：F.cross_entropy(student.view(-1, V), labels.view(-1), ignore_index=-100)
  - 加权组合：α·T²·KL + (1-α)·CE
  - 返回 (total_loss, {"kl_loss": ..., "ce_loss": ..., "total_loss": ...})
  - 参考：contracts/trainer.md distillation_loss 接口
- [x] T022 [US3] 实现 `load_teacher_model()` in `src/trainer.py`
  - AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
  - model.eval()，冻结所有参数 param.requires_grad_(False)
  - 移动到指定 device
  - 参考：contracts/trainer.md load_teacher_model 接口
- [x] T023 [US3] 实现 `DistillationTrainer.__init__()` in `src/trainer.py`（depends on T021, T022）
  - 初始化 AdamW 优化器（lr, weight_decay）
  - 初始化 cosine scheduler with linear warmup
  - 冻结教师模型，设置 eval 模式
  - 初始化训练状态变量（epoch, step, best_val_loss）
  - 参考：contracts/trainer.md DistillationTrainer 接口
- [x] T024 [US3] 实现 `DistillationTrainer.train()` in `src/trainer.py`（depends on T023）
  - 训练循环：遍历 epoch × batches
  - 每步：教师前向(no_grad) → 学生前向 → 计算蒸馏 loss → 反向传播 → 梯度裁剪 → 优化器 step → scheduler step
  - 每 log_interval 步打印 loss, lr, step
  - 每 eval_interval 步调用 evaluate()
  - 每 save_interval 步调用 save_checkpoint()
  - NaN/Inf 梯度检测：torch.isnan/isinf 检查，记录警告
  - 训练结束保存最终检查点
  - 返回训练历史字典
- [x] T025 [US3] 实现 `DistillationTrainer.evaluate()` in `src/trainer.py`
  - 学生模型 eval 模式遍历验证集
  - 计算平均 val_loss 和 perplexity (exp(loss))
  - 恢复学生模型 train 模式
  - 返回 {"val_loss": float, "val_ppl": float}
- [x] T026 [US3] 实现 `save_checkpoint()` 和 `load_checkpoint()` in `src/trainer.py`
  - 保存：student_model.state_dict(), optimizer, scheduler, epoch, step, best_val_loss, config
  - 加载：恢复所有状态，支持从中断处继续训练
  - 使用 torch.save/torch.load
- [x] T027 [US3] 运行 `tests/test_trainer.py` 验证所有测试通过

**Checkpoint**: User Story 3 完成。蒸馏训练能力就绪，可端到端训练并保存模型。

---

## Phase 6: User Story 4 - 模型推理与文本生成 (Priority: P4)

**Goal**: 加载蒸馏后的学生模型，支持贪心/top-k/top-p 解码策略生成文本

**Independent Test**: 加载检查点 → 输入提示词 → 生成文本 → 验证非空且可解码

### Tests for User Story 4 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T028 [P] [US4] 编写生成模块测试 in `tests/test_generate.py`
  - 测试 load_trained_model 返回 eval 模式的 StudentModel
  - 测试 TextGenerator.generate(strategy="greedy") 返回非空字符串
  - 测试 TextGenerator.generate(strategy="top_k") 返回非空字符串
  - 测试 TextGenerator.generate(strategy="top_p") 返回非空字符串
  - 测试生成文本长度 ≤ len(prompt_tokens) + max_new_tokens
  - 测试不同 temperature 下生成结果有差异（temperature=0.1 vs 1.5）

### Implementation for User Story 4

- [x] T029 [US4] 实现 `load_trained_model()` in `src/generate.py`
  - 创建 StudentModel(config)
  - 从检查点加载 model_state_dict
  - 设置 eval() 模式并移动到 device
  - 参考：contracts/generate.md load_trained_model 接口
- [x] T030 [US4] 实现 `TextGenerator.__init__()` in `src/generate.py`
  - 保存 model, tokenizer, device 引用
  - 确认 model 处于 eval 模式
- [x] T031 [US4] 实现 `TextGenerator.generate()` in `src/generate.py`（depends on T030）
  - 编码 prompt → input_ids
  - 自回归生成循环：
    - 前向传播获取 next_token_logits
    - 根据 strategy 选择 next_token：
      - greedy: argmax
      - top_k: 保留前 k 个 logit，softmax 后采样
      - top_p: 按概率降序累积至阈值 p，在范围内采样
    - temperature 缩放：logits / temperature
    - 拼接 next_token 到序列
    - 遇到 eos_token 或达到 max_new_tokens 时停止
  - 解码生成的 token 序列为文本
  - 参考：contracts/generate.md TextGenerator 接口
- [x] T032 [US4] 运行 `tests/test_generate.py` 验证所有测试通过

**Checkpoint**: User Story 4 完成。推理生成能力就绪，可根据提示词生成文本。

---

## Phase 7: Integration & Notebook

**Purpose**: 创建 Colab 主 Notebook，整合所有模块，端到端演示

- [x] T033 [INTEG] 创建 `notebooks/main.ipynb` — Colab 主 Notebook
  - Cell 1: 环境检查 & pip install（torch, transformers, datasets）
  - Cell 2: 挂载 Google Drive（可选，用于持久化检查点）
  - Cell 3: 从 src/ 导入所有模块，打印版本信息
  - Cell 4: 加载 Tokenizer & 构建 WikiDataset & DataLoader
  - Cell 5: 构建 StudentModel & 打印参数量 & 显存状态
  - Cell 6: 加载教师模型 & 显存检查
  - Cell 7: 创建 DistillationTrainer & 执行训练
  - Cell 8: 绘制 loss 曲线（matplotlib）
  - Cell 9: 文本生成演示（greedy / top-k / top-p 对比）
  - Cell 10: 保存最终模型

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 代码质量收尾

- [x] T034 [P] 检查所有 .py 文件中文注释完整性（宪法要求）
- [x] T035 [P] 运行全量测试 `pytest tests/ -v` 确认通过
- [x] T036 验证 quickstart.md 流程可复现（本地 + Colab）

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1 (Setup)
    ↓
Phase 2 (Foundation: config.py)
    ↓
Phase 3 (US1: model.py) ──→ Phase 4 (US2: data.py) ──→ Phase 5 (US3: trainer.py) ──→ Phase 6 (US4: generate.py)
                                                                                            ↓
                                                                                     Phase 7 (Notebook)
                                                                                            ↓
                                                                                     Phase 8 (Polish)
```

### Within Each User Story

- 测试先行（TDD），先写测试确认 FAIL
- 按依赖顺序实现各组件
- 最后运行测试确认全部 PASS

### Parallel Opportunities

- T006 & T007: 不同测试文件，可并行
- T008, T009, T011: 独立模型组件，可并行
- T015, T020, T028: 不同测试文件，可并行编写（但实现需按 US 顺序）
- T034, T035: 独立收尾任务，可并行

---

## Task Summary

| Phase | User Story | Tasks | 文件 |
|-------|-----------|-------|------|
| 1 | Setup | T001-T003 | 目录, requirements.txt, __init__.py |
| 2 | Foundation | T004-T005 | src/config.py |
| 3 | US1 模型架构 | T006-T014 | src/model.py, tests/test_model.py, tests/test_config.py |
| 4 | US2 数据准备 | T015-T019 | src/data.py, tests/test_data.py |
| 5 | US3 蒸馏训练 | T020-T027 | src/trainer.py, tests/test_trainer.py |
| 6 | US4 推理生成 | T028-T032 | src/generate.py, tests/test_generate.py |
| 7 | Integration | T033 | notebooks/main.ipynb |
| 8 | Polish | T034-T036 | 全部文件 |
| **Total** | | **36 tasks** | |

---

## Notes

- 所有代码需配备中文注释（宪法要求）
- 关键算法（RoPE、GQA、SwiGLU、KL 蒸馏损失）需行内注释说明数学原理
- 先写测试再实现（TDD，宪法工作流程要求）
- 每完成一个 Task 后建议提交一次 commit
- User Story 间按 P1→P2→P3→P4 顺序实现，因为存在依赖链
