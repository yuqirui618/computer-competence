# 第三方来源与归档说明

本目录保存项目实际执行的比较代码与聚合报告；本说明不为项目自有脚本另行设定许可。模型实现通过timm等已安装库调用，没有把完整第三方库源码、预训练权重或微调权重打包在本目录。依赖版本见[model01/requirements.txt](../requirements.txt)。

## 五份实际预训练渠道

以下许可来自对应下载渠道的模型卡记录，不能将库代码许可替代权重许可。归档保留了实际下载时的模型卡、配置、HF元数据、固定revision与权重SHA。未来重新取得权重时仍需核对这些信息及比赛规则。

| 精确模型名 | 权重训练来源 | 模型卡许可 | 归档材料 |
| --- | --- | --- | --- |
| convnext_tiny.fb_in1k | ImageNet-1k | apache-2.0 | [模型卡](sources/新模型对照_20261105/pretrained/convnext_tiny/README.md) · [配置](sources/新模型对照_20261105/pretrained/convnext_tiny/config.json) · [许可全文](sources/新模型对照_20261105/pretrained/LICENSE.apache-2.0.txt) |
| tf_efficientnetv2_s.in1k | ImageNet-1k | apache-2.0 | [模型卡](sources/新模型对照_20261105/pretrained/efficientnetv2_s/README.md) · [配置](sources/新模型对照_20261105/pretrained/efficientnetv2_s/config.json) · [许可全文](sources/新模型对照_20261105/pretrained/LICENSE.apache-2.0.txt) |
| caformer_s18.sail_in1k | ImageNet-1k | apache-2.0 | [模型卡](sources/顶刊三模型对照_20261105/pretrained/caformer_s18/README.md) · [配置](sources/顶刊三模型对照_20261105/pretrained/caformer_s18/config.json) · [许可全文](sources/顶刊三模型对照_20261105/pretrained/LICENSE.apache-2.0.txt) |
| densenet121.ra_in1k | ImageNet-1k | apache-2.0 | [模型卡](sources/顶刊三模型对照_20261105/pretrained/densenet121/README.md) · [配置](sources/顶刊三模型对照_20261105/pretrained/densenet121/config.json) · [许可全文](sources/顶刊三模型对照_20261105/pretrained/LICENSE.apache-2.0.txt) |
| hrnet_w18.ms_aug_in1k | ImageNet-1k | mit | [模型卡](sources/顶刊三模型对照_20261105/pretrained/hrnet_w18/README.md) · [配置](sources/顶刊三模型对照_20261105/pretrained/hrnet_w18/config.json) · [许可全文](sources/顶刊三模型对照_20261105/pretrained/LICENSE.hrnet-MIT.txt) |

实际.safetensors和.pth均保留本地。Apache-2.0与HRNet MIT许可全文按原路径保留；模型卡与元数据为原文件字节副本。原始Apache文件记录在各实验的license_source(s).json中，HRNet MIT许可来自作者官方HRNet-Image-Classification仓库。

## 代码与共同依赖

- 本目录八份实际存在的Python/PowerShell脚本按原字节保存，来源与SHA见[source_manifest.json](source_manifest.json)。两模型比较中的权重下载当时通过临时命令完成，没有独立下载脚本可归档；这里没有伪造“当时执行过”的文件。三模型比较的download_pretrained.py是实际文件。
- 两个训练入口复用同一套项目DataLoader、预处理、Study评估及旧训练模块。八个直接与传递依赖已在[optimization源码归档](../optimization/README.md)中保存，本包不重复复制；source_manifest.json的shared_dependencies记录它们的原路径、归档路径、字节数与SHA。
- 部署时先按optimization说明部署共享源码，再运行本包deploy_sources.py。默认只读检查，只有显式--apply才补齐缺失文件；不同文件拒绝覆盖。比较包的部署预检还会确认共享依赖与记录的SHA一致。
- 训练入口依赖Windows、CUDA及E盘原目录拓扑；在导入时设置缓存并创建temp。download_pretrained.py没有main保护，导入会联网并可能重写下载元数据。因此归档审查使用AST和哈希，不能批量import或直接从仓库sources运行。
- sources代码归档不包含原始影像、预训练或微调权重、含真值的原始NPZ，也未收录全部本地审计输入。整个model_comparison另在[validation](validation/README.md)下公开按官方结构导出的Study级CSV；其中保留必要的官方匿名Subject_id/Study_id及预测，不含真值列。源码快照可审阅，并不等于仅clone后就能重现全部历史结果；冻结路径、输入SHA和患者隔离检查均保留。

## 文献与报告

期刊作者PDF的下载索引被保留用于指向原文，但PDF文件只作原E盘个人阅读，没有上传。本地三份PDF的索引明确记录了这一范围。公开报告使用相对仓库文档链接或作者原文链接，实验数值没有修改；原文件与公开副本记录见[报告副本校验](reports/报告副本校验.json)。

报告中的“未上传”或本地目录描述属于其生成时的历史状态。本次发布范围是代码、来源配置、聚合验证结果与validation下的官方格式Study级预测CSV；原始含真值NPZ、影像和模型二进制继续留在本地。三组组合为固定50:50的概率平均，不是参数融合；未来推理仍需要两份绑定权重，并分别遵守它们的来源与许可。
