# 五种骨干模型与固定组合验证对照

2026-11-05完成5种新骨干模型的全量初筛：ConvNeXt-Tiny、EfficientNetV2-S、CAFormer-S18、DenseNet-121、HRNet-W18，每种5轮，共25轮。每轮使用30,344张官方训练图；验证集1,422张图，按检查汇总为859次Study、228位患者。这里只用官方验证集评分，没有新增训练或用测试集调参。

本目录集中保存本轮实际代码、逐轮总结和按官方五列结构导出的验证预测。MambaOut此前86轮与优化源码继续保留在[optimization](../optimization/README.md)，没有挪动或覆盖原有模型。

| 目录/文件 | 内容 |
| --- | --- |
| [sources](sources/) | 两轮实验实际执行的脚本、固定组合计划、预训练来源及模型配置；保留原始字节与E盘目录拓扑 |
| [reports](reports/) | 两阶段实验总结、25轮指标、5个骨干与3组合的同口径比较及文献核查 |
| [validation](validation/README.md) | 5骨干、3组合、2个MambaOut参照的官方结构验证CSV、专属Gate、来源与回读检查 |
| [tools/export_validation.py](tools/export_validation.py) | 从已核实的本地验证NPZ导出CSV并回读校验，不运行训练或测试集推理 |
| [source_manifest.json](source_manifest.json) | 源码及配置SHA256、目标拓扑、共享依赖 |
| [deploy_sources.py](deploy_sources.py) | 仅校验/部署历史源码；默认dry-run，不导入训练模块 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 上游来源、ImageNet权重渠道及Apache-2.0/MIT许可 |

## 结果怎么看

下表是Study Macro AUC及同口径患者五折阈值OOF Macro F1，均来自本机已经完成的验证记录。每个新骨干按AUC优先、固定0.5的F1次优选出最佳轮；组合只采用这些已选轮次，没有按组合成绩重新选轮或搜索比例。

| 方案 | 最佳轮/来源 | 验证AUC | 患者五折OOF F1 |
| --- | --- | --- | --- |
| 原保留MambaOut冠军 | 保护性微调权重 | 0.80197696 | 0.451055 |
| Gamma＋轻噪声MambaOut候选 | 原实验第1轮 | 0.80217748 | 0.442491 |
| ConvNeXt-Tiny | 5 | 0.78945253 | 0.430902 |
| EfficientNetV2-S | 4 | 0.78514468 | 0.413895 |
| CAFormer-S18 | 3 | 0.79119136 | 0.429735 |
| DenseNet-121 | 4 | 0.77265703 | 0.421592 |
| HRNet-W18 | 5 | 0.78645840 | 0.426004 |
| Gamma MambaOut＋CAFormer | 固定50:50概率均值 | 0.80307234 | 0.460976 |
| Gamma MambaOut＋DenseNet | 固定50:50概率均值 | 0.79669534 | 0.449988 |
| Gamma MambaOut＋HRNet | 固定50:50概率均值 | 0.80514173 | 0.437889 |

5个新骨干在该5轮预算下均未超过现有MambaOut。HRNet组合的AUC点估计最高，但相对Gamma的患者配对差值95%区间为[-0.002708,+0.010844]，跨0；CAFormer组合AUC差值区间也跨0。不能认定稳定突破，也不能把双模型组合分数归给新单模型。现有MambaOut经历了更多轮次和方案选择，本次不是等预算架构优劣的因果证明。

完整结果见[二十五轮总明细](reports/二十五轮总明细.csv)、[顶刊三模型实验总结](reports/顶刊三模型实验总结.md)、[前次两模型总结](reports/新模型对照总结.md)和[固定组合对照](reports/等权组合对照.md)。患者OOF阈值在每折其余患者上拟合，但最佳轮及历史方案已经反复使用官方验证集；这仍是内部评估，不能替代独立测试。

## 验证CSV与官方提交结构

每份CSV严格使用以下5列，无额外索引、真值或分折列：

```csv
Subject_id,Study_id,Predict_class,Probability,Gate
```

859次官方验证Study，每次输出1至10类，故每份8,590行数据，另加1行表头。ID保留官方原值，按官方验证CSV首次Study出现顺序排列，每次内部按类1至10排列。Probability是同Study各图Sigmoid概率均值；组合则在两条Study概率上取固定50:50均值。概率和Gate保留可往返的浮点精度，Gate与相应权重SHA绑定。

**这些文件属于验证集，不能直接作为正式测试集提交。** 官方测试集有1,414次Study，正式测试CSV需14,140行数据。本轮没有生成新测试预测，也未改写既有submission.csv；模型文件规模和T4端到端时延还需按赛事环境验收。

提交结构的Gate采用各类最终阈值，在全验证集上拟合并对同类固定。用这些Gate回算验证F1是拟合内指标；上表患者OOF F1采用折外Gate，两者不能互换。每份导出的实际AUC、最终Gate的F1和独立回读检查见validation中的检查记录。

## 环境、源码与复核

实际环境为Windows、Python 3.13、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、timm 1.0.22，完整版本见[model01/requirements.txt](../requirements.txt)。沿用E盘既有环境，没有新建C盘环境。源码中的E盘路径及冻结SHA保留用于追溯；不能只把训练脚本移到任意目录就认为已完成环境与输入准备。

源码共享依赖来自此前optimization归档。先使用[optimization部署工具](../optimization/deploy_sources.py)核对/部署依赖，再使用本目录部署工具；两者默认只检查，仅加--apply时部署缺失且已验证的源码，不覆写不同文件。实际运行还需要官方数据、清单、各模型预训练权重和历史对照的匹配权重/预测；它们没有随本次代码归档上传。初始化均为公开ImageNet权重，不使用外部医学数据预训练权重。

sources保留实际执行代码；reports是适合GitHub阅读的报告副本，链接改为仓库相对路径。原实验报告及JSON中关于“当时未上传”的字段记录训练完成时状态，本次按用户要求发布代码、聚合总结与验证CSV，模型权重仍仅保留E盘。含真值的NPZ、训练检查点、原始影像、论文PDF和临时缓存不属于本次提交。
