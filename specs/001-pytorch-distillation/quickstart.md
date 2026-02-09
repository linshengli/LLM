# 快速启动指南

**Phase 1 产出** | **Date**: 2026-02-09

## 本地开发环境

### 前置条件

- Python 3.10+
- pip
- GPU 可选（本地开发可用 CPU 运行单元测试）

### 安装

```bash
# 克隆项目
git clone <repo-url> && cd LLM
git checkout 001-pytorch-distillation

# 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # 本地 CPU 版
pip install transformers datasets pytest
```

### 运行测试

```bash
pytest tests/ -v
```

### 项目结构

```text
src/
├── config.py      # 配置数据类
├── model.py       # 模型架构
├── data.py        # 数据加载
├── trainer.py     # 蒸馏训练
└── generate.py    # 文本生成

tests/             # 单元测试（镜像 src/ 结构）
notebooks/
└── main.ipynb     # Colab 主 Notebook
```

## Google Colab 环境

### 启动步骤

1. 上传项目文件到 Google Drive 或直接从 GitHub 克隆
2. 打开 `notebooks/main.ipynb`
3. 运行时选择 **T4 GPU**
4. Notebook 第一个 Cell 会自动安装依赖并验证环境

### Notebook 预期 Cell 顺序

1. 环境检查 & 依赖安装
2. 加载 Tokenizer & 准备数据
3. 构建学生模型 & 验证架构
4. 加载教师模型 & 验证显存
5. 执行蒸馏训练
6. 评估 & 文本生成演示

### 显存监控

```python
import torch
print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 关键配置

| 参数 | 值 | 可调节 |
|------|-----|--------|
| hidden_size | 512 | 按需缩小以省显存 |
| num_layers | 12 | 按需减少 |
| batch_size | 8 | OOM 时降至 4 或 2 |
| max_seq_len | 512 | OOM 时降至 256 |
| learning_rate | 3e-4 | 可微调 |
| temperature | 2.0 | 蒸馏温度，1-4 范围内调节 |
| alpha | 0.5 | 蒸馏权重，0-1 范围内调节 |
