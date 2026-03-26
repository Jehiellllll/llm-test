# SentiMM 论文方法复现（Framework Reproduction）

> 目标：构建**可运行、可扩展、可审计**的 SentiMM 五阶段复现框架。  
> 声明：当前仓库是**框架复现**，不是“完整原始 SentiMMD 数据 + 原论文全部闭源骨干”的严格逐点复现。

## 1. 论文事实对齐

- 五阶段 pipeline：Text Analyst → Image Analyst → Fusion Inspector → KB Assistant → Classifier Aggregator
- 7 类标签：Like / Happiness / Anger / Disgust / Fear / Sadness / Surprise
- 论文数据设定（文献描述）：SentiMMD 3500（3150 train / 350 test）
- 论文主干含 GPT-4o / Qwen2.5-VL-7B，本仓库通过统一模块接口支持替换，不强绑定闭源 API

## 2. 项目结构

- `src/` 核心实现
- `configs/` 主要实验配置
- `scripts/` 训练/评测/消融/预处理脚本
- `tests/` pytest 测试
- `docs/` 设计说明
- `outputs/` 结果输出

## 3. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 4. 数据准备

### 4.1 检查本地是否存在原始 SentiMMD

请自行放置数据；仓库不包含也不会伪造原始 SentiMMD。

### 4.2 兼容数据格式（JSONL）

每行一个样本，字段：

```json
{
  "id": "sample-001",
  "text": "post text",
  "image_path": "images/1.jpg",
  "video_path": null,
  "label": "Happiness",
  "kb_text": "optional external knowledge",
  "metadata": {"source": "custom"}
}
```

- `image_path` 与 `video_path` 至少有一个可用（也可都给，优先 video）
- `label` 必须是 7 类之一

## 5. 训练脚本

```bash
PYTHONPATH=src python scripts/train_text_only.py --train-jsonl data/train.jsonl
PYTHONPATH=src python scripts/train_image_only.py --train-jsonl data/train.jsonl
PYTHONPATH=src python scripts/train_multimodal.py --train-jsonl data/train.jsonl
PYTHONPATH=src python scripts/train_multimodal_with_retrieval.py --train-jsonl data/train.jsonl
```

## 6. 评测与消融

```bash
PYTHONPATH=src python scripts/evaluate.py --model outputs/multimodal_retrieval.joblib --eval-jsonl data/test.jsonl
PYTHONPATH=src python scripts/ablate.py --train-jsonl data/train.jsonl --eval-jsonl data/test.jsonl
```

指标包含：
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix

支持消融：
- no KB Assistant
- no Fusion Inspector
- no Image Analyst
- no Text Analyst
- no Classifier Aggregator

## 7. 预处理脚本

```bash
PYTHONPATH=src python scripts/preprocess_text.py --input-jsonl data/raw.jsonl --output-jsonl data/text_norm.jsonl
PYTHONPATH=src python scripts/preprocess_image.py --input-dir data/images_raw --output-dir data/images_224 --size 224
PYTHONPATH=src python scripts/sample_video_frames.py --video data/video.mp4 --output-dir data/frames --stride 10 --max-frames 16
```

## 8. 可复现性与审计

- 所有关键假设写入 `REPRO_ASSUMPTIONS.md`
- 训练/评测写入 `outputs/*.json`
- 统一随机种子配置（`configs/default.yaml`）

## 9. 当前边界与限制

- 未包含原始 SentiMMD 数据
- 未直接复用 GPT-4o / Qwen2.5-VL-7B 原论文推理链路
- 当前是开源友好的工程近似复现；严格复现实验结果需官方数据、模型和完整提示词/超参细节

## 10. 下一步 TODO

- 增加 HuggingFace 预训练编码器（文本/视觉）可选后端
- 对接真实 SentiMMD 后复现实验表格
- 增加更细粒度日志（tensorboard/wandb）
