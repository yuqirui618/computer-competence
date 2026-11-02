# computer-competence｜胸部 X 光片多标签分类竞赛

全国高校计算机能力挑战赛人工智能赛道备赛仓库。调研日期：**2026-11-02**。依据：[官方赛题](https://www.ncccu.org.cn/index/Paper/case1.html)及其规则、2026-10-27 补充说明。

## 推荐路线

**自然图像预训练 MambaOut-Tiny（CVPR 2025）→ 官方数据有监督微调 → 10 个独立 Sigmoid 输出 → 同次检查多视角聚合 → 验证集选择损失与阈值 → T4 上验证推理 → 生成 submission.csv。**

以 ConvNeXt-Tiny 做基线，DINOv2 Small + Registers 做异构对照；最新方法重点阅读 **RGB Loss（MICCAI 2026）**，先与 BCE、ASL 比较。Edge-RecViT、CHASE 等 2026 年工作已筛查，但代码、权重或任务条件限制使其更适合作为研究分支。这里的推荐尚未在官方数据上验证。

已核实的主候选入口：[MambaOut 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html)、[官方代码](https://github.com/yuweihao/MambaOut)、[ImageNet-1K 权重与 Apache-2.0 模型卡](https://huggingface.co/timm/mambaout_tiny.in1k)、[RGB Loss 会议页面](https://papers.miccai.org/miccai-2026/0870-Paper0566.html)。

## 赛题决定的实施边界

这是 **10 类多标签分类、Study_id 级提交**。比赛提供训练/验证/测试数据，主要指标为 Macro-AUC，辅助指标为 Macro-F1。只使用许可匹配的非医学预训练权重；医学预训练、NC 和 gated 权重排除。实际推理模型与必要文件合计须 ≤500MB，并在主办方 T4 16GB 上满足平均单图端到端 ≤100ms。

细则、类别与提交格式统一见 [规则解读](06_references/COMPETITION_RULES.md)和 [数据说明](03_datasets/README.md)。区域赛/省赛截止：**2026-11-13 23:59，北京时间**。当前约有 11 天，优先建立可提交基线。

## 文件夹说明

| 文件夹 | 讲什么 | 关键文件 |
| --- | --- | --- |
| [01_literature](01_literature/README.md) | 16 项模型/方法文献：2026 新工作、实用骨干、损失函数及淘汰理由 | 文献调研报告 |
| [02_models](02_models/README.md) | 代码、权重、许可、大小估算、实际复用步骤 | 模型表、REPRODUCTION_GUIDE.md |
| [03_datasets](03_datasets/README.md) | 官方数据结构、10 类映射、Study/Subject 分组、CSV 规范 | class_mapping.csv、datasets.csv |
| [04_technical_route](04_technical_route/README.md) | 分类模型、多视角融合、公式、技术路线图与截止前排期 | pipeline.mmd |
| [05_experiments](05_experiments/README.md) | 对比/消融、AUC/F1、速度与大小检查、结果记录模板 | experiment_plan.csv、results_template.csv |
| [06_references](06_references/README.md) | 论文链接/BibTeX、官方规则来源、检索记录、公开权重元数据 | papers.csv、references.bib、COMPETITION_RULES.md |

建议顺序：`规则解读 → 01 文献 → 02 模型 → 03 数据 → 04 路线 → 05 实验`。

## 当前交付状态

已完成赛题核对、文献与公开实现筛查、技术路线和实验设计。**尚未获取官方比赛数据、下载模型权重、安装训练环境或训练模型；没有实测分数、速度与显存结论。** 本机检测到 RTX 5060 Laptop、8151 MiB 显存；部署硬件以赛方 T4 为准。

公开仓库只保存自主整理的调研、公开链接和模板。比赛原始数据、患者标识、预测文件及模型权重默认不上传。论文原文通过链接访问。
