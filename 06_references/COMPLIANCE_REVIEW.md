# 推荐路线与赛题要求复核

复核日期：2026-11-02。范围为当前方案及实际下载的主候选权重；不替代主办方最终核验，也不代表已完成训练、计时或比赛提交。

**结论：胸片 10 类多标签分类、自然图像预训练 MambaOut-Tiny、仅官方数据微调、Study 内固定聚合的方向，与已核实赛题和规则相容。** 权重公开获取、来源、许可和文件完整性已落实；最终推理包大小、T4 延迟和图像可解码性与提交模板仍待验证。

## 1. 逐项对应

| 要求 | 当前方案及证据 | 状态 |
| --- | --- | --- |
| 胸片 10 类多标签分类 | 视觉骨干后接 10 个独立 logits，推理使用 Sigmoid | 设计匹配；任务头尚未训练 |
| 以 Study_id 提交 | 对官方同一次检查的图像固定聚合，输出每类概率 | 实际CSV已包含Study_id，图像数与检查数已核对；聚合代码待实现 |
| 医学样本来自赛方 | 训练/验证/测试均采用官方版本；外部医学数据不进入流程 | 已接收官方ZIP及解压数据，划分检查数与官网一致 |
| 非医学预训练主数据 | `timm/mambaout_tiny.in1k` 模型卡明确为作者训练的 ImageNet-1K 权重 | 已核实 |
| 公开获取、非 gated | 固定 revision 的权重已在未提供认证凭据的情况下成功下载；API 标记 gated=false | 已核实 |
| 非 NC、允许再分发 | 权重模型卡标注 Apache-2.0；作者许可证全文已本地保存 | 已核实来源及许可；提交时附完整材料 |
| 记录权重哈希 | 实际文件 SHA-256 与发布方 LFS SHA-256 相同 | 已核实 |
| 不利用测试集调参 | 模型、聚合、阈值仅用训练/验证确定；测试固定 eval 前向 | 实施约束 |
| 部署文件总大小 ≤500MB | 原始骨干权重实际约 106.2MB；最终统计需包含实际推理必需文件 | 原始权重已测；最终包待测 |
| T4 平均单图端到端 ≤100ms | 按指定 100 图、batch=1、3 轮及另 5 图预热测完整流程 | 待 T4 实测，不能提前保证 |
| 原创与知识产权 | 引用论文和复用代码来源、保留许可，完成自己的数据与训练/推理实现，不把已有损失称为原创 | 实施约束 |

官方依据：[赛题页面](https://www.ncccu.org.cn/index/Paper/case1.html)、[参赛规则](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)、[2026-10-27 补充说明](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)。

同一 Study 的固定均值聚合不拟合测试统计量，与补充说明关于确定性聚合的允许范围一致。实现时仍须使用官方分组字段，并如实计入各图像前向与聚合耗时。

## 2. 确切权重与已完成检查

- 模型：`timm/mambaout_tiny.in1k`。
- 来源：[模型卡](https://huggingface.co/timm/mambaout_tiny.in1k)。
- 固定 revision：`d080941663a60c42d015ecf8d8fad86d12694208`。
- 文件：[固定版本 model.safetensors](https://huggingface.co/timm/mambaout_tiny.in1k/resolve/d080941663a60c42d015ecf8d8fad86d12694208/model.safetensors)。
- 实际字节数：`106197008`。
- 实际 SHA-256：`ac9535c5b9fe1d325c5bf58dd4ffc3201c1fc04992b37e393977a7526cccafb4`。
- 发布方 LFS 哈希：与上项一致。
- [作者 Apache-2.0 许可证](https://github.com/yuweihao/MambaOut/blob/main/LICENSE)全文与固定版本模型卡、config 已保存在本地权重目录。

本地路径：`weights/pretrained/mambaout_tiny.in1k/`，已被 Git 忽略，模型大文件未上传公开仓库。可公开核查的元数据见 [weight_metadata.json](weight_metadata.json)。

另已读取 safetensors 头部确认文件为 F32 参数，原始分类头为 1000 类。这是 ImageNet 骨干，需要重新建立本赛 10 类头并微调；该文件还不是训练完成的胸片模型。没有运行完整模型前向、反向或性能评估。

## 3. 新损失与可选分支

BCE/ASL/RGB 仅在训练阶段使用，类别统计和参考样本来自官方训练集，不在推理阶段重新拟合。ASL 代码使用 MIT；RGB 官方代码使用 GPL-3.0，直接复用时遵守该许可，且排除其 README 里的医学预训练权重和外部数据配置。

当前先实施单模型 BCE，再做 ASL/RGB 对照。自监督、蒸馏、集成是可选项；每项实施前按规则落实数据范围、初始化来源、声明材料和部署预算。没有把这些尚未实现的分支视为已验证收益。

## 4. 规则版本复核

此次从赛题页面所列 URL 重新下载的两份 PDF，与前次保存的文件逐字节哈希相同：

| 附件 | SHA-256 |
| --- | --- |
| 参赛规则 | `ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6` |
| 补充说明 | `836c4f5d3202f2dc65830fd919ca195797c12b7507d750121786a21672dc7a44` |

后续若赛方更新链接、模板或公告，应重新核查。本次尚缺的下载入口与落地步骤见 [官方数据获取说明](../03_datasets/DOWNLOAD_GUIDE.md)。
