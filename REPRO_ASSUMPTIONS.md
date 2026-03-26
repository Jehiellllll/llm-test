# REPRO_ASSUMPTIONS

本文档记录《SentiMM: A Multimodal Multi-Agent Framework for Sentiment Analysis in Social Media》在公开信息不足情况下的“框架复现”假设。

## 1) 数据层
1. 若原始 SentiMMD 数据集不可直接获取，本仓库不伪造官方数据，仅定义 SentiMMD-like JSONL 接口。
2. 假设每条样本可包含：`text`、`image_path`、`label`、可选 `kb_text`。
3. 7 类标签固定为：Like, Happiness, Anger, Disgust, Fear, Sadness, Surprise。

## 2) 五阶段 Agent 实现映射
1. Text Analyst：使用 TF-IDF(1-2gram) 表示文本。
2. Image Analyst：使用可本地运行的手工视觉特征（RGB 直方图 + 通道统计），避免依赖闭源或远程 API。
3. Fusion Inspector：使用跨模态一致性手工特征（范数、均值差、余弦近似）。
4. KB Assistant：将外部知识文本 `kb_text` 以独立 TF-IDF 建模。
5. Classifier Aggregator：以各子模块概率输出作为元特征，训练 Logistic Regression 进行聚合。

## 3) 分类与训练
1. 将问题视为单标签 7 分类，多分类器统一使用可复现的 sklearn LogisticRegression。
2. 无论文明确超参数时，默认采用可运行的中等规模超参（如 max_iter=1000，C=1.0）。

## 4) 消融定义
1. `no KB Assistant`：移除 KB 特征与其子分类头。
2. `no Fusion Inspector`：移除 fusion 特征与其子分类头。
3. `no Image Analyst`：移除视觉特征与其子分类头。
4. `no Text Analyst`：移除文本特征与其子分类头。
5. `no Classifier Aggregator`：不训练聚合器，改为可用子分类头概率均值投票。

## 5) 复现边界声明
1. 本项目目标是“框架复现（framework reproduction）”，非“完整原始数据+原始权重逐点复现”。
2. 若未来获得官方数据与更多实现细节，可在当前模块化骨架上替换特征提取器与训练策略。
