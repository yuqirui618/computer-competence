# CAFormer-S18-384-EMA-RGB

本目录是 **CAFormer-S18-384 + EMA + RGB Loss** 的本地实验/候选代码，位于 Echo-12345 的 `feat/model` 分支。

## 与原 CAFormer-S18-384-EMA 的关系

- 原版 **BCE+EMA**（验证 AUC 0.8061307765、已生成正式 test submission）保留在 `feat/model01-mambaout` 分支的 `model01/CAFormer-S18-384-EMA/`，本分支不覆盖它。
- 本目录只把第二阶段（EMA 续训）的损失从 BCE 换成 **RGB Loss**，其余训练流程、输入 384、Study 概率均值、患者 OOF 阈值协议保持一致。

## 文件

| 路径 | 用途 |
|---|---|
| `training/ema_continue.py` | 第二阶段入口，`--arm RGB_EMA` |
| `training/ema_core.py` | 训练/验证核心，新增 RGB 分支 |
| `training/losses.py` | RGB Loss（官方 ClassAsyDiff 的移植） |
| `training/ema_recipe.json` | 训练配方（含 RGB 参数） |
| `infer.py` / `gates.json` | 固定发布模型的推理与阈值（原 BCE 版参考；RGB 训练后需重新生成） |
| `weights_manifest.json` | 原 BCE+EMA 权重来源与 SHA 参考；RGB 训练会生成新权重与 SHA |
| `README_RGB_VARIANT.md` | RGB 实验的详细运行步骤 |

## 重要

- 本目录是**实验代码**，当前没有 RGB 训练完成的新权重/新 submission。
- `gates.json`、`weights_manifest.json` 中的 SHA/阈值来自原 BCE 版本，**不能**直接当作 RGB 模型结果。
- 训练数据、权重、检查点、预测 NPZ 不入 Git。
- RGB Loss 来源：Lin et al., MICCAI 2026；官方实现 GPL-3.0；详见 `training/losses.py` 与 `README_RGB_VARIANT.md`。
