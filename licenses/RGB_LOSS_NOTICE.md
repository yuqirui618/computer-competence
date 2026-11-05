# RGB Loss 改进说明（Echo-12345 / feat/model）

- 论文：Lin et al., "Reference-Guided Gradient Balancing Loss for Long-Tailed Multi-label Medical Image Classification", MICCAI 2026.
- 官方实现：https://github.com/xup6YJ/RGB-Loss （GPL-3.0）
- 本项目用途：仅作为训练损失；在 `train.py` 中通过 `LOSS=RGB` 启用，默认仍为 BCE。
- 数据来源：仅使用赛方官方训练集 batch 统计；未使用任何外部医学数据或医学预训练权重。
- 实现说明：`losses.py` 按论文公式重新实现，未复制官方仓库的医学预训练/外部数据配置。
- 合规提示：如提交包包含本实现，请保留本说明及 GPL-3.0 许可信息，并如实申报。
