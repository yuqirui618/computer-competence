# 当前模型：CAFormer-S18 + RGB Loss + 320

## 结论

在 2000 张 / 2 轮的小样本对比中：

| 分辨率 | Study Macro-AUC | Image Macro-AUC |
|---|---:|---:|
| 224 | 0.6773 | 0.6920 |
| **320** | **0.7003** | **0.7188** |
| 384 | 0.6995 | 0.7097 |

320 与 384 基本持平，且 320 略好、更快，因此当前默认选 **320**。

## 训练/推理

```powershell
# 训练（RGB Loss 默认）
python train.py --config configs/caformer_rgb_320.yaml --subset 2000 --epochs 2 --ckpt checkpoints/rgb_320.pth

# 验证
python evaluate.py --ckpt checkpoints/rgb_320.pth --split val --config configs/caformer_rgb_320.yaml

# 推理
python predict.py --images D:/cxr/raw/cxr_first/valid --ckpt checkpoints/rgb_320.pth --out preds.csv --img-size 320
```

## 说明

- 架构仍是 CAFormer-S18，预训练权重为 ImageNet-1K 原生 384（Apache-2.0、非医学、非 gated）；
- 320 只是训练/推理输入分辨率，数据集文件不变；
- checkpoint `checkpoints/rgb_320.pth` 是本地训练产物，未提交；
- 完整 384 两阶段 EMA 配方仍在 `CAFormer-S18-384-EMA-RGB/`，未覆盖。
