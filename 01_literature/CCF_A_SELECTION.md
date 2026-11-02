# CCF-A 精选阅读清单

已核实 **15 篇 CCF-A 主会论文**，其中下表 **12 篇**作为重点阅读对象，超过不少于 10 篇的要求。另有 3 篇 A 类论文因权重许可或任务匹配原因作为边界参考。

分级依据：[CCF 官方人工智能类目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)，核查日期 2026-11-02；本清单统计 CVPR、ICCV、ICML 的主会正式论文，不将 Workshop、MICCAI、未核实等级的 ICLR/TMLR 或纯预印本计入 A 类数量。CCF 分级针对会议/期刊，并不代表每篇论文的单独质量评分。

| 阅读顺序 | ID / 方法 | 正式发表 | 与本赛关系 | 文件 |
| --- | --- | --- | --- | --- |
| 1 | L01 MambaOut: Do We Really Need Mamba for Vision? | CVPR 2025 · CCF-A | 当前主候选；门控 CNN 分类骨干 | [官方原文](https://arxiv.org/pdf/2405.07992v3)；本地 `L01_MambaOut.pdf` |
| 2 | L10 A ConvNet for the 2020s | CVPR 2022 · CCF-A | 稳定基线；标准卷积骨干 | [PDF](01_backbones/L10_ConvNeXt.pdf) |
| 3 | L11 Asymmetric Loss For Multi-Label Classification | ICCV 2021 · CCF-A | 正负样本不平衡；与 BCE 对照 | [官方原文](https://arxiv.org/pdf/2009.14119v4)；本地 `L11_Asymmetric_Loss.pdf` |
| 4 | L07 RepViT: Revisiting Mobile CNN From ViT Perspective | CVPR 2024 · CCF-A | 轻量化与部署备选 | [PDF](01_backbones/L07_RepViT.pdf) |
| 5 | L19 EfficientNetV2: Smaller Models and Faster Training | ICML 2021 · CCF-A | 高效分类与渐进训练 | [PDF](01_backbones/L19_EfficientNetV2.pdf) |
| 6 | L17 Multi-Label Image Recognition With Graph Convolutional Networks | CVPR 2019 · CCF-A | 疾病共现与标签图建模 | [官方原文](https://arxiv.org/pdf/1904.03582v1)；本地 `L17_ML_GCN.pdf` |
| 7 | L18 General Multi-Label Image Classification With Transformers | CVPR 2021 · CCF-A | 标签 Transformer 与缺失标签训练 | [PDF](02_training_methods/L18_C_Tran.pdf) |
| 8 | L20 Training data-efficient image transformers & distillation through attention | ICML 2021 · CCF-A | 数据效率与合规蒸馏 | [PDF](01_backbones/L20_DeiT.pdf) |
| 9 | L21 EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | ICML 2019 · CCF-A | 分辨率、宽度与深度的计算预算 | [PDF](01_backbones/L21_EfficientNet.pdf) |
| 10 | L03 Edge-RecViT: Efficient Vision Transformer via Semantic-Refined Dynamic Recursion | CVPR 2026 · CCF-A | 2026 参数共享与动态计算研究 | [官方原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.pdf)；本地 `L03_Edge_RecViT.pdf` |
| 11 | L05 DARC: Dual Adjustment Reasoning with Counterfactuals for Trustworthy Chest X-ray Classification | CVPR 2026 · CCF-A | 2026 胸片共现和捷径混杂研究 | [官方原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Liao_DARC_Dual_Adjustment_Reasoning_with_Counterfactuals_for_Trustworthy_Chest_X-ray_CVPR_2026_paper.pdf)；本地 `L05_DARC.pdf` |
| 12 | L06 Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification | CVPR 2026 · CCF-A | 2026 通用表征迁移研究；原任务为 3D | [PDF](03_recent_research/L06_AnyMC3D.pdf) |

## 其余 3 篇 A 类边界参考

- L12 ConvNeXt V2（CVPR 2023）：阅读架构与训练设计；原始 NC 权重不进入比赛。
- L13 MambaVision（CVPR 2025）：阅读混合骨干设计；原始 NC 权重不进入比赛。
- L14 MedCLIPSeg（CVPR 2026）：研究视觉语言适配；医学/NC 权重及分割任务不作为本赛起点。

## 阅读与实施顺序

先读 L01、L10、L11、L07、L19 建立可运行骨干和损失对照；再读 L17/L18 研究标签关系；有预算时考虑 L20 蒸馏、L21 计算预算，以及 L03/L05/L06 的 2026 年研究设计。

论文被选入 CCF-A 清单，不等于允许复用其全部数据、权重、代码或推理设定。标签图、类别统计、阈值及训练监督只用官方训练/验证数据；推理不提供隐藏标签；蒸馏师生来源和延迟须遵守赛事规则。
