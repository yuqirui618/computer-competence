# 从公开骨干到本赛分类模型

这是后续实施指南，本次未安装环境或运行模型。比赛数据须从参赛账号取得，仅存本地。先建立一个能完整训练、验证和导出 CSV 的单模型基线。

## 1. 环境

官方部署环境列出 Python 3.12、PyTorch 2.10.0/CUDA 12.8、T4 16GB，优先在兼容环境验证最终包；镜像见 [赛题页](https://www.ncccu.org.cn/index/Paper/case1.html)。本机 RTX 50 系还需检查 Blackwell 构建支持。[PyTorch 官方说明](https://pytorch.org/blog/pytorch-2-7/)。不直接沿用旧论文的 CUDA 版本。

至少验证真实 CUDA 张量运算、完整模型前向和反向，再保存精确依赖。安装或升级任何包之前先用隔离环境，避免覆盖其他项目。

## 2. 公开权重的明确选择

首选 `timm/mambaout_tiny.in1k`，模型卡说明为论文作者的 ImageNet-1K 训练权重，Apache-2.0、非 gated。本次核实 revision：`d080941663a60c42d015ecf8d8fad86d12694208`。[模型卡](https://huggingface.co/timm/mambaout_tiny.in1k)。

该 revision 的具体权重、config、模型卡与作者许可证已保存于本地 `weights/pretrained/mambaout_tiny.in1k/`，SHA-256 与发布方一致。后续配置应加载这一已校验副本；下面的 `hf-hub:` 入口仅用于说明接口，最终运行不依赖远程临时变化。

以下仅示意公开接口，不是本仓库已实现的训练脚本：

```python
import timm
import torch

model = timm.create_model(
    'hf-hub:timm/mambaout_tiny.in1k',
    pretrained=True,
    num_classes=10,
)
logits = model(images)                  # images: B x 3 x H x W
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
probabilities = logits.sigmoid()       # B x 10，每类独立
```

`num_classes=10` 会重新初始化任务头，骨干仍来自选定的自然图像权重；必须确认确实加载而非静默随机初始化。最终保存任务训练后的完整 state_dict 与预处理配置。

## 3. 数据与监督

先读官方标签文件，构造每次检查 10 维标签和对应图像列表；核查缺失/不确定标记的定义，不把它们自动改为阴性。有多个视角时，可先逐图预测并按 Study 聚合；更严格的训练分支在 Study 级聚合后计算损失，避免默认每个投照方向都能看到全部检查标签。

不执行作者原本的 ImageNet/COCO 训练命令，也不照抄 RGB 仓库医学预训练示例。ASL/RGB 只是训练目标候选，所有医学样本来自官方数据。

## 4. 适配与对照顺序

1. ConvNeXt-Tiny + BCE 完成一次训练/验证；保存逐检查概率。
2. MambaOut-Tiny + BCE，对齐图像尺寸、训练预算与验证集。
3. 同骨干分别替换 ASL、RGB；统计总体、逐类 AUC、低频类和训练稳定性。
4. 比较 Study 内 mean-probability、mean-logit 等固定聚合；仅在验证集选择。
5. 有时间再跑 DINOv2 Small；集成只在速度与大小仍有余量时保留。

## 5. 推理与导出

`model.eval()` + `torch.inference_mode()`；固定归一化、resize、模型、聚合和阈值。关闭随机增强和 dropout。TTA 是固定多次前向，不是测试时训练；同样纳入耗时。

按 [类别表](../03_datasets/class_mapping.csv)输出 1–10 类，不按 Python 索引 0–9 直接提交。Subject_id、Study_id 按字符串保存，保留前导零，每个 Study 的 10 个类别完整覆盖。

`cost.py` 应导入与 `predict.py` 相同的公共推理函数。计时包含读图、预处理、前向、聚合和结果写出；不能用缓存概率替代实际推理。完整计时与一致性要求见 [规则解读](../06_references/COMPETITION_RULES.md)。

## 6. 复现记录

记录预训练文件哈希、最终权重哈希、代码版本、数据版本、Subject 分组检查、训练和验证用途、随机种子、图像尺寸、有效 batch 与真实 batch、训练步数、损失参数、聚合方式和阈值。所有训练/评价/速度均未执行；不把此指南当成跑通证明。
