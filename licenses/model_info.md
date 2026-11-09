# CAFormer-S18-384 权重合规材料（Echo-12345 / feat/model）

- 模型名：`caformer_s18.sail_in1k_384`（CAFormer-S18，原生 384）
- 用途：比赛分类骨干，ImageNet-1K 预训练，非医学数据
- 许可证：Apache-2.0（全文见同目录 `LICENSE`）
- 获取方式：非 gated，无需登录 / 单独授权

## 预训练权重信息

| 项目 | 值 |
|---|---|
| HuggingFace | https://huggingface.co/timm/caformer_s18.sail_in1k_384 |
| 预训练数据 | ImageNet-1K（native 384） |
| 许可证 | Apache-2.0 |
| SHA-256 | 629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806 |
| 参数量 | 约 26M（CAFormer-S18） |

## 训练/续训权重

- 正式候选：`CAFormer-S18-384-EMA-RGB/`（本地两阶段 + EMA + RGB Loss）
- 原版 BCE 冠军权重/提交仍保留在仓库 `feat/model01-mambaout` 分支，不在本分支覆盖
- 训练得到的 EMA 权重是本地训练产物，不属于预训练权重，不在此静态申报

## RGB Loss 说明

- 论文：Lin et al., MICCAI 2026, "Reference-Guided Gradient Balancing Loss for Long-Tailed Multi-label Medical Image Classification"
- 官方实现：https://github.com/xup6YJ/RGB-Loss （GPL-3.0）
- 仅用于训练，推理不使用；无外部医学数据/医学预训练权重
- 详见同目录 `RGB_LOSS_NOTICE.md`

## 当前选定输入分辨率

- 选定：**320 × 320**（2000 张 / 2 轮小样本对比：Study Macro-AUC 0.7003，优于 224 的 0.6773，与 384 的 0.6995 基本持平）
- 配置文件：`configs/caformer_rgb_320.yaml`
- 本地 checkpoint：`checkpoints/rgb_320.pth`（92.8MB，未提交，`.gitignore` 已忽略）
- 完整 384 两阶段 EMA 配方仍保留在 `CAFormer-S18-384-EMA-RGB/`，未覆盖
