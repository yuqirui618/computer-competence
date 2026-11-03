# MambaOut-Tiny 权重合规材料（陈负责存档）

- 模型名：`timm/mambaout_tiny.in1k`（MambaOut-Tiny）
- 用途：比赛分类骨干，ImageNet-1K 预训练，非医学数据
- 许可证：Apache-2.0（全文见同目录 `LICENSE`）
- 获取方式：非 gated，无需登录 / 单独授权

## 权重信息（已核验，2026-11-02）

| 项目 | 值 |
|---|---|
| HuggingFace | https://huggingface.co/timm/mambaout_tiny.in1k |
| 固定 revision | d080941663a60c42d015ecf8d8fad86d12694208 |
| 权重文件 | model.safetensors |
| 实际大小 | 106,197,008 字节（约 106.2 MB） |
| SHA-256 | ac9535c5b9fe1d325c5bf58dd4ffc3201c1fc04992b37e393977a7526cccafb4 |
| 参数量 / FLOPs | 26.5M / 4.5G @224 |

## 官方仓库（作者 yu weihao，备查）

- 仓库：https://github.com/yuweihao/MambaOut
- LICENSE：Apache-2.0
- 作者原版权重（备查，本方案不使用）：
  https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth
- 官方 ImageNet 训练配方（仅参考，不是微调配置）：
  lr 4e-3 / adamw / batch 4096 / warmup 20 epochs / drop_path 0.2 / AMP