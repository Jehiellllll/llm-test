# SentiMM (Framework Reproduction)

这是对论文 **SentiMM: A Multimodal Multi-Agent Framework for Sentiment Analysis in Social Media** 的**可运行、模块化、研究友好**复现工程。

> 注意：如果原始 SentiMMD 数据不可用，本仓库提供的是**任务兼容接口 + 框架复现**，不是官方原始数据的完整复刻。

## 项目结构

- `src/`：核心实现
- `configs/`：配置
- `scripts/`：训练/评测/消融/冒烟脚本
- `tests/`：单元测试
- `docs/`：说明文档
- `outputs/`：输出目录

## 五阶段模块

1. `Text Analyst`
2. `Image Analyst`
3. `Fusion Inspector`
4. `KB Assistant`
5. `Classifier Aggregator`

## 7 类情绪

- Like
- Happiness
- Anger
- Disgust
- Fear
- Sadness
- Surprise

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 数据接口（SentiMMD-like）

每行一个 JSON（jsonl）：

```json
{"id":"1","text":"...","image_path":"images/1.jpg","kb_text":"...","label":"Happiness"}
```

## 训练

```bash
PYTHONPATH=src python scripts/train.py --train-jsonl data/train.jsonl --model-out outputs/model.joblib
```

## 评测

```bash
PYTHONPATH=src python scripts/evaluate.py --model outputs/model.joblib --eval-jsonl data/test.jsonl --metrics-out outputs/metrics.json
```

输出指标包含：
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix

## 消融实验

```bash
PYTHONPATH=src python scripts/run_ablation.py --train-jsonl data/train.jsonl --eval-jsonl data/test.jsonl --output outputs/ablation_results.json
```

已支持：
- no KB Assistant
- no Fusion Inspector
- no Image Analyst
- no Text Analyst
- no Classifier Aggregator

## 复现假设

见 `REPRO_ASSUMPTIONS.md`。
