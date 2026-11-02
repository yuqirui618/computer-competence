# 文献调研：对齐 10 类胸片多标签分类

检索截止 2026-11-02。优先检索 CVPR 2026、MICCAI 2026 及 CVPR/ICCV/ICLR 已发表工作；结合真实权重、许可和部署要求筛选。L01–L16 对应 [文献清单](../06_references/papers.csv)和 [BibTeX](../06_references/references.bib)。

## 1. 选型结论

推荐 **MambaOut-Tiny + 多标签分类头 + Study 内聚合**。先跑 BCE，再对照 ASL；最新改进分支优先看 MICCAI 2026 的 RGB Loss。损失替换不增加部署网络结构，适合本赛的时间与文件大小预算。

2026 年论文已找到，但“论文新”与“可直接套用”是两件需要分别核查的事。CHASE 需要额外层级标注；DARC 引入外部混杂因素标注；Edge-RecViT 的公开例子不能直接证明论文权重和 T4 速度已就绪。建议把这些工作用于设计依据，先保证可运行的 2025 主干与完整提交链。

## 2. 2026 最新工作

| ID / 方法 | 发表状态与来源 | 核心机制 | 本赛可迁移部分与缺口 |
| --- | --- | --- | --- |
| L02 RGB Loss | [MICCAI 2026 主会](https://papers.miccai.org/miccai-2026/0870-Paper0566.html) | 利用训练 batch 内正确预测作为参考，平衡负梯度并增强低频正类 | 与胸片长尾多标签最贴近；有代码。只取损失设计并用合规骨干，默认医学预训练示例不能搬用；小 batch 稳定性需测 |
| L03 Edge-RecViT | [CVPR 2026 主会论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.pdf) | 参数共享的递归 Transformer，按 token 特征分配计算 | [公开代码](https://github.com/RicardoLee510520/Edge-RecViT)为 Apache-2.0；需核对原论文权重及训练来源，动态路由在 T4 上不一定更快 |
| L04 CHASE | [MICCAI 2026 主会](https://papers.miccai.org/miccai-2026/0172-Paper1023.html) | 解剖—子区域—病变层级分类与一致性约束 | 可参考标签依赖设计；赛事没有论文完整层级标签，不能直接搬完整训练设定；作者代码链接本次返回 404 |
| L05 DARC | [CVPR 2026 主会](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_DARC_Dual_Adjustment_Reasoning_with_Counterfactuals_for_Trustworthy_Chest_X-ray_CVPR_2026_paper.html) | 处理疾病共现和非病理视觉混杂造成的捷径 | 借鉴训练集内设备分层分析与鲁棒性检验；其外部 CheXconf 标注不纳入本赛训练 |
| L06 AnyMC3D | [CVPR 2026 主会](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Revisiting_2D_Foundation_Models_for_Scalable_3D_Medical_Image_Classification_CVPR_2026_paper.html) | 用通用 2D 基础模型配轻量插件进行 3D 分类 | 支持研究通用视觉表征迁移；原任务是 3D，不能据此声称已验证二维胸片性能 |

RGB 的 [官方实现](https://github.com/xup6YJ/RGB-Loss)采用 GPL-3.0；公开衍生代码需履行相应许可条件。README 的示例涉及医学预训练，且入口与根目录文件存在需要核对之处，不能把整条命令当成可直接运行的本赛复现脚本。

## 3. 能较快落地的骨干与训练方法

| ID / 方法 | 发表状态 | 主要价值 | 本项目安排 |
| --- | --- | --- | --- |
| L01 MambaOut | [CVPR 2025 主会](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html) | 使用门控 CNN，去掉状态空间模块；官方训练的 ImageNet 权重已被 timm 收录 | 主候选 Tiny，约 26.5M 参数；[模型卡](https://huggingface.co/timm/mambaout_tiny.in1k)明确 ImageNet-1K、Apache-2.0，无 gated |
| L07 RepViT | [CVPR 2024 主会](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html) | 适合部署的轻量 CNN 与结构重参数化 | 速度备选；选择 ImageNet 分类权重，别混用 RepViT-SAM；手机延迟不能当成 T4 成绩 |
| L08 Registers | [ICLR 2024](https://openreview.net/pdf?id=2dnO3LLiJ1) | 添加寄存器 token，减少 ViT 特征中的高范数伪影 | 对照自然图像预训练 DINOv2 Small with Registers；效果须在官方验证集测 |
| L09 DINOv2 | [作者论文](https://arxiv.org/abs/2304.07193)，2023 预印本，后获 TMLR 接收 | 无监督通用视觉特征 | 强异构对照；只使用原始自然图像模型，排除同仓库的 XRay-DINO、Cell-DINO 等分支 |
| L10 ConvNeXt | [CVPR 2022 主会](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) | 由标准卷积模块组成的现代 CNN | Tiny 做稳定基线；代码与权重来源逐项记录 |
| L11 ASL | [ICCV 2021 主会](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) | 对正负样本采用不同聚焦策略，减轻大量简单负类的影响 | 作为 BCE 后第一项损失对照；[官方代码](https://github.com/Alibaba-MIIL/ASL)为 MIT |

这些论文的 ImageNet top-1 或其他公开数据集 AUC 不能横向拼接成本赛模型排名。胜出模型由同一官方验证协议、大小及 T4 延迟共同决定。

## 4. 已调研但不采用的权重

| ID | 工作 | 出处 | 本次排除理由 |
| --- | --- | --- | --- |
| L12 | ConvNeXt V2 | CVPR 2023 | [官方说明](https://github.com/facebookresearch/ConvNeXt-V2)明确其 ImageNet 预训练/微调权重采用 CC-BY-NC；代码 MIT 不改变权重限制 |
| L13 | MambaVision | CVPR 2025 | [官方说明](https://github.com/NVlabs/MambaVision)的预训练权重为 CC-BY-NC-SA-4.0 |
| L14 | MedCLIPSeg | CVPR 2026 | 医学预训练；[任务权重](https://huggingface.co/TahaKoleilat/MedCLIPSeg)另有 NC 条款；任务为分割 |
| L15 | Dino U-Net | MICCAI 2026 | 分割任务；默认 DINOv3 主干获取有授权门槛，且需额外编译；不作为本赛起点 |
| L16 | DINOv3 | [作者论文/代码](https://github.com/facebookresearch/dinov3)，2025 起 | 官方权重入口需接受获取授权，按本赛 gated 限制排除；不把它与 DINOv2 混同 |

MedSAM、BiomedCLIP、RETFound、CheXzero 等医学预训练权重也不进入本赛方案。可以阅读其论文，但不能用这些模型的输出给学生蒸馏，或将医学权重换个文件名继续使用。[规则依据](../06_references/COMPETITION_RULES.md)。

## 5. 读完文献后应确认什么

先读 L01/L10 建立骨干对照，再读 L11/L02 设计损失消融；L08/L09 用于异构表征对照；L03/L04/L05 用于有预算的研究分支。

逐篇核对：预训练数据、权重许可、是否无需登录下载、是否额外需要医学标注、推理网络是否增加、是否支持当前依赖、能否在 batch=1 计入完整前后处理后满足时限。本报告给出可验证的设计依据，未宣称实现文献中的精度。
