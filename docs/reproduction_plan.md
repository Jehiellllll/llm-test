# Reproduction Plan

## Phase 1 仓库审查（已完成）
当前仓库已有基础框架，但与新要求存在差距：
- 缺少指定脚本名：`train_text_only.py` 等
- 缺少 video 支持
- 缺少 FAISS/本地检索模块
- 缺少明确的 discrepancy-refinement 机制
- 缺少 alpha/beta 聚合公式的显式实现
- 缺少预处理脚本与更完整日志

## 论文模糊点（待假设化）
- fusion function 与 discrepancy 公式未公开完整细节
- KB 构建、索引与检索策略不详
- 最终聚合器结构不详
- GPT-4o / Qwen2.5-VL-7B 具体提示词与训练细节不详

## Minimal Reproducible Version (MRV)
1. 五模块接口完整
2. 本地开源可运行（TF-IDF + handcrafted vision + PyTorch heads）
3. 支持 7 类分类、训练、评测、五项消融
4. 输出标准指标 + confusion matrix

## Stretch Version
1. HuggingFace 文本/视觉编码器可插拔
2. learned aggregator 强化
3. 更强视频关键帧抽取
4. 统一实验追踪（wandb/tensorboard）
