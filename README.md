# computer-competence｜胸部 X 光片多标签分类竞赛

全国高校计算机能力挑战赛人工智能赛道备赛仓库。调研日期：**2026-11-02**。依据：[官方赛题](https://www.ncccu.org.cn/index/Paper/case1.html)及其规则、2026-10-27 补充说明。

## 推荐路线

**自然图像预训练 MambaOut-Tiny（CVPR 2025）→ 官方数据有监督微调 → 10 个独立 Sigmoid 输出 → 同次检查多视角聚合 → 验证集选择损失与阈值 → T4 上验证推理 → 生成 submission.csv。**

以 ConvNeXt-Tiny 做基线，DINOv2 Small + Registers 做异构对照；最新方法重点阅读 **RGB Loss（MICCAI 2026）**，先与 BCE、ASL 比较。Edge-RecViT、CHASE 等 2026 年工作已筛查，但代码、权重或任务条件限制使其更适合作为研究分支。这里的推荐尚未在官方数据上验证。

已核实的主候选入口：[MambaOut 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html)、[官方代码](https://github.com/yuweihao/MambaOut)、[ImageNet-1K 权重与 Apache-2.0 模型卡](https://huggingface.co/timm/mambaout_tiny.in1k)、[RGB Loss 会议页面](https://papers.miccai.org/miccai-2026/0870-Paper0566.html)。

## 赛题决定的实施边界

这是 **10 类多标签分类、Study_id 级提交**。比赛提供训练/验证/测试数据，主要指标为 Macro-AUC，辅助指标为 Macro-F1。只使用许可匹配的非医学预训练权重；医学预训练、NC 和 gated 权重排除。实际推理模型与必要文件合计须 ≤500MB，并在主办方 T4 16GB 上满足平均单图端到端 ≤100ms。

细则、类别与提交格式统一见 [规则解读](06_references/COMPETITION_RULES.md)和 [数据说明](03_datasets/README.md)。区域赛/省赛截止：**2026-11-13 23:59，北京时间**。截至2026-11-03约有10天，优先建立可提交基线。

## 文件夹说明

| 文件夹 | 讲什么 | 关键文件 |
| --- | --- | --- |
| [01_literature](01_literature/README.md) | 21 篇论文原文；15 篇 CCF-A，精选 12 篇；按四类子目录存放 | 文献调研报告 |
| [02_models](02_models/README.md) | 代码、权重、许可、大小估算、实际复用步骤 | 模型表、REPRODUCTION_GUIDE.md |
| [03_datasets](03_datasets/README.md) | 官方数据结构、10 类映射、Study/Subject 分组、CSV 规范 | class_mapping.csv、datasets.csv |
| [04_technical_route](04_technical_route/README.md) | 分类模型、多视角融合、公式、技术路线图与截止前排期 | pipeline.mmd |
| [05_experiments](05_experiments/README.md) | 对比/消融、AUC/F1、速度与大小检查、结果记录模板 | experiment_plan.csv、results_template.csv |
| [赛题相关资料](赛题相关资料/README.md) | 任务说明及正式规则原件 | 任务描述.docx、参赛规则.pdf、参赛补充规则.pdf |
| [06_references](06_references/README.md) | 论文链接/BibTeX、官方规则来源、检索记录、公开权重元数据 | papers.csv、references.bib、COMPETITION_RULES.md |

建议顺序：`规则解读 → 01 文献 → 02 模型 → 03 数据 → 04 路线 → 05 实验`。

## 当前交付状态

已完成赛题核对、文献与公开实现筛查、技术路线和实验设计。**已下载并校验主候选的公开 ImageNet 权重；已取得官方比赛数据并核对划分规模；尚未安装训练环境或训练模型，没有实测分数、速度与显存结论。** 本机检测到 RTX 5060 Laptop、8151 MiB 显存；部署硬件以赛方 T4 为准。

当前逐项复核见 [赛题符合性复核](06_references/COMPLIANCE_REVIEW.md)，官方数据入口与本地接收步骤见 [下载说明](03_datasets/DOWNLOAD_GUIDE.md)。

论文已下载到 `01_literature/` 的四类子目录：本地共 21 篇，核实 15 篇 CCF-A，精选阅读 12 篇；10 篇有明确再分发许可的 A 类原文同步公开仓库，其余 11 篇仅在本地保存并提供官方链接。见 [CCF-A 精选](01_literature/CCF_A_SELECTION.md)、[PDF 总索引](01_literature/PDF_INDEX.md)。2026-11-03已接收E盘官方数据：33,984张图像及3份CSV；按仓库所有者确认的授权范围以[Release分卷](https://github.com/yuqirui618/computer-competence/releases/tag/official-cxr-first-2026-11-03)发布，状态见[数据清单](06_references/official_dataset.json)。数据不复制到C盘。赛题原件见[赛题相关资料](赛题相关资料/README.md)。预测文件、模型权重及下载凭据不上传。
