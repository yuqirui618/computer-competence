# 当前模型：CAFormer-S18 + 384（原生分辨率）

## 为什么改回 384

2000 张 / 2 轮小样本对比：

| 分辨率 | Study Macro-AUC | Image Macro-AUC |
|---|---:|---:|
| 224 | 0.6773 | 0.6920 |
| 320 | 0.7003 | 0.7188 |
| **384** | 0.6995 | 0.7097 |

320 与 384 基本持平（差 0.0008，属于单种子噪声）；384 是 CAFormer-S18 **原生预训练分辨率**，因此改回 384 作为默认。

## 损失函数对比（384 / 2000 张 / 2 轮）

| LOSS | Study Macro-AUC | Image Macro-AUC | Study F1@0.5 |
|---|---:|---:|---:|
| RGB | 0.6995 | 0.7097 | 0.0689 |
| BCE | 0.7043 | 0.7243 | 0.0697 |
| **ASL** | **0.7176** | **0.7317** | 0.2827 |
| WEIGHTED_BCE | 0.7024 | 0.7210 | **0.3348** |

**当前默认：ASL**（Macro-AUC 最好；小样本单种子，后续需多种子/更大数据确认）。

`losses.py` 支持：

| LOSS | 说明 |
|---|---|
| `BCE` | 基线 |
| `RGB` | Reference-Guided Gradient Balancing（已试，未稳定提升） |
| `ASL` | Asymmetric Loss（γ⁻=4, γ⁺=1, clip=0.05） |
| `WEIGHTED_BCE` | 按官方训练集 Study 频率加权 |
| `FOCAL` | Focal Loss（γ=2, α=0.25） |

## 用法

```powershell
# 默认已是 ASL；如需切换：$env:LOSS = "BCE" / "WEIGHTED_BCE" / "FOCAL" / "RGB"
python train.py --config configs/caformer_rgb_384.yaml --subset 2000 --epochs 2 --ckpt checkpoints/asl_384.pth
python evaluate.py --ckpt checkpoints/asl_384.pth --split val --config configs/caformer_rgb_384.yaml
```
