# 模型与权重复用清单

截至 2026-11-02。选择自然图像预训练骨干，在官方胸片上有监督微调；所有候选的赛事适用性最终仍以主办方核验为准。

| 模型 | 可检查的实现与权重 | 预训练/许可观察 | 安排 |
| --- | --- | --- | --- |
| MambaOut-Tiny | [作者代码](https://github.com/yuweihao/MambaOut)、[固定模型卡](https://huggingface.co/timm/mambaout_tiny.in1k) | ImageNet-1K；Apache-2.0；HF API 返回 gated=false | 主候选，先单模型 |
| ConvNeXt-Tiny | [作者代码](https://github.com/facebookresearch/ConvNeXt)、[torchvision 权重文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) | 官方自然图像权重；原项目 MIT；实施时保存所选发布渠道的完整材料 | 基线 |
| DINOv2 ViT-S/14 with Registers | [代码/下载表](https://github.com/facebookresearch/dinov2) | 原始 DINOv2 自然图像权重为 Apache-2.0；排除 XRay-DINO/Cell-DINO | 异构对照；是否集成由验证决定 |
| RepViT-M1.0/M1.5 | [代码/分类权重](https://github.com/THU-MIG/RepViT) | ImageNet-1K 分类模型，项目 Apache-2.0；实施前保存具体权重许可 | 延迟不足时备选 |
| Edge-RecViT | [代码](https://github.com/RicardoLee510520/Edge-RecViT) | Apache-2.0；本次未确认完整原论文权重与上游初始化链 | 2026 研究分支，暂不排入首轮主模型 |

## 组件选择

- **BCEWithLogitsLoss**：最先实现的基线。输出 10 个 logits，推理独立 Sigmoid；不用单标签 Softmax。
- **ASL**：[MIT 官方实现](https://github.com/Alibaba-MIIL/ASL)。作为损失替换对照，保留代码出处与许可。
- **RGB Loss（MICCAI 2026）**：[GPL-3.0 官方实现](https://github.com/xup6YJ/RGB-Loss)。只迁移训练损失，在官方训练数据上重新计算类别统计；不搬医学预训练及外部数据配置。小 batch 下类参考不足是本机实验风险。

## 大小和速度预算

MambaOut-Tiny 模型卡约 26.5M 参数，纯参数存储量粗估 FP32 106MB、FP16 53MB；实际文件包含缓冲区、头部及元数据，应按解压后所有推理必需文件统计。该估算不是导出包实测大小。

先用 224/288 输入作工程起点，再在验证集与 T4 上比较 320/384。模型名中的 Tiny 不能保证 100ms；高分辨率、多视角、TTA、集成及文件读写都会增加耗时。建议内部目标留出余量，如 ≤80ms、≤450MB，这只是工程目标。

本机 RTX 5060 Laptop 8GB 可用于小 batch 试验；正式部署目标为 T4 16GB。先测训练前向/反向，再确定 batch；梯度累计不会让 batch 相关的 RGB 参考集合等价于真正大 batch。

## 权重落实时必须保存

确切模型名称、公开下载链接、发布方、上游训练数据说明、许可证全文、下载文件 SHA-256、代码提交、依赖版本。已实际下载主候选固定版本的权重并核对发布方 SHA-256；精确大小和哈希见 [weight_metadata.json](../06_references/weight_metadata.json)，逐项证据见 [符合性复核](../06_references/COMPLIANCE_REVIEW.md)。权重只保存在本地忽略目录，完整模型训练和部署仍待验证。

加载与适配步骤见 [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)。
