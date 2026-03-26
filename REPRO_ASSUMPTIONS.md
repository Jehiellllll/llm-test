# REPRO_ASSUMPTIONS

本文档仅记录“论文未明确写清或无法直接获得”的实现细节与复现假设，避免把假设误写成论文原始实现。

## A. 数据相关假设

1. 论文提到 SentiMMD 3500 样本（3150/350），但当前仓库未内置原始数据。  
   - 假设：提供 `SentiMMD-like` JSONL 接口以兼容任务。  
   - 原因：遵循“不伪造原始数据”原则。
2. 样本结构中增加 `video_path` 与 `metadata` 字段。  
   - 假设：这属于工程扩展，不代表论文原始字段。

## B. Text Analyst 假设

1. 论文未完整公开 text encoder 和训练细节。  
   - 假设：第一版用 TF-IDF + PyTorch MLP head 本地可运行。
2. `rationale` 采用轻量规则（文本长度）占位。  
   - 假设：解释字段是工程可审计增强，不等同论文 prompt-based rationale。

## C. Image Analyst 假设

1. 论文未给出公开视频处理细节。  
   - 假设：video 采用定步长抽帧 + 平均池化特征。
2. 视觉特征采用 RGB histogram + channel stats。  
   - 假设：作为开源低门槛基线，可替换为更强视觉编码器。

## D. Fusion Inspector 假设

1. 论文未完整定义 fusion function 与 discrepancy 公式。  
   - 假设：特征拼接 + MLP score + discrepancy threshold refinement hook。
2. refinement 规则：若 discrepancy 超阈值，则向单模态均值回拉。  
   - 假设：务实稳定策略，不宣称论文同款。

## E. KB Assistant 假设

1. 论文未详述检索索引构建与检索特征。  
   - 假设：默认以训练集 `kb_text` 建库，向量为 TF-IDF。
2. 检索后端：优先 FAISS，失败时回退 sklearn KNN。  
   - 假设：满足本地可运行与可部署性。
3. retrieval summary 输出近邻索引与距离。  
   - 假设：增强审计能力。

## F. Classifier Aggregator 假设

1. 论文未明确最终分类器完整公式。  
   - 假设：默认 `combined = alpha * multimodal + beta * retrieved`。
2. 同时提供 learned mode（MLP）作为可扩展选项。  
   - 假设：研究扩展，不代表论文原始实现。

## G. 训练与复现性假设

1. 统一设置随机种子（numpy/torch/python）。
2. 默认设备 CPU。
3. 训练超参数（epochs/lr/hidden）来自工程合理默认，不声称论文官方超参。

## H. 复现边界声明

1. 当前实现是“方法框架复现”，不是“原论文结果的严格逐点复现”。
2. 若后续获得官方数据、prompt、模型权重、全量实验设置，可在本工程骨架上继续逼近严格复现。
