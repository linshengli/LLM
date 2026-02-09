# Tasks: PyTorch 从零实现 Qwen/DeepSeek 模型知识蒸馏

**Input**: Design documents from `/specs/001-pytorch-distillation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Scope**: Phase 1 only — User Story 1（从零构建 Transformer 模型架构）

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目初始化和基础结构搭建

- [ ] T001 创建项目目录结构：`src/`, `tests/`, `notebooks/`，按 plan.md 中的 Source Code 布局创建所有目录
- [ ] T002 创建 `requirements.txt`，包含依赖：torch, transformers, datasets, pytest
- [ ] T003 [P] 创建 `src/__init__.py` 和 `tests/__init__.py`，确保 Python 包结构正确

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 配置数据类，所有模块共享的基础设施

**⚠️ CRITICAL**: User Story 1 的模型实现依赖此阶段的配置类

- [ ] T004 [SETUP] 实现 `ModelConfig` 数据类 in `src/config.py`
  - 字段：hidden_size=512, num_layers=12, num_heads=8, num_kv_heads=2, intermediate_size=2048, vocab_size=151936, max_seq_len=512, rope_theta=1e6, norm_eps=1e-6, dropout=0.0
  - 验证规则：hidden_size % num_heads == 0, num_heads % num_kv_heads == 0
  - 参考：data-model.md ModelConfig 实体定义
- [ ] T005 [P] [SETUP] 实现 `TrainingConfig` 数据类 in `src/config.py`
  - 字段：batch_size=8, learning_rate=3e-4, weight_decay=0.01, warmup_steps=500, num_epochs=3, gradient_clip=1.0, alpha=0.5, temperature=2.0, checkpoint_dir, log_interval=50, eval_interval=500, save_interval=1000
  - 参考：data-model.md TrainingConfig 实体定义

**Checkpoint**: 配置基础就绪，User Story 1 模型实现可以开始

---

## Phase 3: User Story 1 - 从零构建 Transformer 模型架构 (Priority: P1) 🎯 MVP

**Goal**: 从零使用 PyTorch 实现一个 ~120M 参数的 Decoder-Only Transformer 模型，对齐 Qwen2.5 架构特性

**Independent Test**: 构建模型 → 输入随机 token ID → 验证输出 logits 形状为 (batch, seq_len, vocab_size)，参数量约 123M

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T006 [P] [US1] 编写配置验证测试 in `tests/test_config.py`
  - 测试 ModelConfig 默认值正确性
  - 测试 hidden_size % num_heads != 0 时抛出 ValueError
  - 测试 num_heads % num_kv_heads != 0 时抛出 ValueError
- [ ] T007 [P] [US1] 编写模型架构测试 in `tests/test_model.py`
  - 测试 RMSNorm: 输入输出形状一致，归一化后均值接近 0
  - 测试 RotaryEmbedding: 输出形状不变，不同位置编码不同
  - 测试 GQAAttention: 输入 (batch, seq, hidden) → 输出形状一致
  - 测试 SwiGLUFFN: 输入 (batch, seq, hidden) → 输出形状一致
  - 测试 TransformerBlock: 输入输出形状一致，残差连接有效
  - 测试 StudentModel: input_ids (batch, seq) → logits (batch, seq, vocab_size)
  - 测试 StudentModel.count_parameters() ≈ 123M (±5%)
  - 测试 lm_head 与 embedding 权重共享（是同一个张量对象）

### Implementation for User Story 1

- [ ] T008 [P] [US1] 实现 `RMSNorm` in `src/model.py`
  - Root Mean Square Layer Normalization
  - 参数：可学习的缩放权重 gamma (hidden_size,)
  - 公式：x * rsqrt(mean(x²) + eps) * gamma
  - 参考：contracts/model.md RMSNorm 接口
- [ ] T009 [P] [US1] 实现 `RotaryEmbedding` in `src/model.py`
  - 预计算旋转频率矩阵 freqs = 1 / (theta^(2i/dim))
  - 根据 position_ids 生成 cos/sin 位置编码
  - 对 Q/K 张量的前半/后半维度应用旋转变换
  - 参考：contracts/model.md RotaryEmbedding 接口
- [ ] T010 [US1] 实现 `GQAAttention` in `src/model.py`（depends on T008, T009）
  - Q 投影: hidden_size → num_heads * head_dim
  - K/V 投影: hidden_size → num_kv_heads * head_dim
  - KV 头扩展（repeat_kv）：将 num_kv_heads 扩展到 num_heads
  - 缩放点积注意力 + 因果遮罩（上三角 -inf mask）
  - 对 Q、K 应用 RoPE
  - O 投影: num_heads * head_dim → hidden_size
  - 参考：contracts/model.md GQAAttention 接口
- [ ] T011 [P] [US1] 实现 `SwiGLUFFN` in `src/model.py`
  - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
  - up_proj: Linear(hidden_size, intermediate_size, bias=False)
  - down_proj: Linear(intermediate_size, hidden_size, bias=False)
  - 公式：down_proj(SiLU(gate_proj(x)) * up_proj(x))
  - 参考：contracts/model.md SwiGLUFFN 接口
- [ ] T012 [US1] 实现 `TransformerBlock` in `src/model.py`（depends on T008, T010, T011）
  - Pre-norm 架构：norm → attention → residual → norm → ffn → residual
  - attention_norm + attention + residual
  - ffn_norm + ffn + residual
  - 参考：contracts/model.md TransformerBlock 接口
- [ ] T013 [US1] 实现 `StudentModel` in `src/model.py`（depends on T012）
  - embedding: nn.Embedding(vocab_size, hidden_size)
  - layers: nn.ModuleList of TransformerBlock × num_layers
  - norm: 最终 RMSNorm
  - lm_head: nn.Linear(hidden_size, vocab_size, bias=False)
  - **权重共享**: lm_head.weight = embedding.weight
  - 自动生成 position_ids 和因果 attention_mask
  - count_parameters() 方法
  - 参考：contracts/model.md StudentModel 接口
- [ ] T014 [US1] 运行 `tests/test_model.py` 验证所有测试通过
  - 前向传播形状验证
  - 参数量验证 (~123M)
  - 权重共享验证
  - 各组件独立测试

**Checkpoint**: User Story 1 完成。学生模型架构从零实现，能够接受 token 输入并输出正确形状的 logits。可独立验证。

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，可立即开始
- **Foundational (Phase 2)**: 依赖 Setup 完成，**阻塞** User Story 1
- **User Story 1 (Phase 3)**: 依赖 Foundational 完成

### Within User Story 1

```text
T006, T007 (测试先行，可并行)
    ↓
T008 (RMSNorm) ──┐
T009 (RoPE) ─────┤── 可并行
T011 (SwiGLUFFN) ┘
    ↓
T010 (GQAAttention) ── depends on T008, T009
    ↓
T012 (TransformerBlock) ── depends on T008, T010, T011
    ↓
T013 (StudentModel) ── depends on T012
    ↓
T014 (运行测试验证)
```

### Parallel Opportunities

- T006 & T007: 测试文件不同，可并行编写
- T008, T009, T011: 独立组件，不同类，可并行实现
- T004 & T005: 同文件但不同类，可并行（或顺序实现更安全）

---

## Notes

- 所有代码需配备中文注释（宪法要求）
- 关键算法（RoPE 旋转变换、GQA 头扩展、SwiGLU 激活）需行内注释说明数学原理
- 先写测试再实现（TDD，宪法工作流程要求）
- 每完成一个 Task 后提交一次 commit
- Phase 2-4（数据、训练、生成）将在后续迭代中规划
