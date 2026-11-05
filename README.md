# 全国高校计算机能力挑战赛 · 胸片多标签分类

使用官方胸片数据进行 10 类多标签分类，按 Study 汇总概率。当前数据清单、缓存和加载接口已通过本地验收。

| 目录 | 内容 |
| --- | --- |
| [论文](01_literature/README.md) | PDF、文献索引、引用与来源记录 |
| [官方数据集](03_datasets/README.md) | 公开 Release、下载与还原步骤、类别表和校验记录 |
| [数据代码](dataset/README.md) | 清单、缓存、审计、DataLoader、评估和测试日志 |
| [模型实验](model01/README.md) | MambaOut历史优化、5种新骨干各5轮对照、固定组合与官方结构验证结果 |
| [队友模型](model02/) | 队友后续上传另一套模型的入口 |
| [赛题相关资料](赛题相关资料/README.md) | 任务说明、两份正式规则及一页备赛方案 |

官方数据共 33,984 张图和 3 份 CSV，ZIP 已发布为 52 卷。[下载数据集](https://github.com/yuqirui618/computer-competence/releases/tag/official-cxr-first-2026-11-03)

数据接口继续通过 `from dataset import ...` 使用。运行命令见[数据模块说明](dataset/README.md)，模型起点与对照安排见[备赛方案](赛题相关资料/备赛方案.md)。
