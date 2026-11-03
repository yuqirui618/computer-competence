# 官方胸片数据集

| 划分 | 图片 | Study | 标签 |
| --- | ---: | ---: | --- |
| 训练 | 30,344 | 18,513 | 有 |
| 验证 | 1,422 | 859 | 有 |
| 测试 | 2,218 | 1,414 | 由赛方保留 |

原始 ZIP 共 55.23 GB，已公开发布为 52 卷。[数据集 Release](https://github.com/yuqirui618/computer-competence/releases/tag/official-cxr-first-2026-11-03)

- [下载和还原步骤](DOWNLOAD_GUIDE.md)与[还原脚本](restore_dataset.py)。
- [数据及分卷清单](official_dataset.json)、[发布校验记录](dataset_release_receipt.json)与[本地整理记录](dataset_cleanup_receipt.json)。
- [10 类机器可读类别表](class_mapping.csv)。
- [数据代码](../dataset/README.md)和[全量审计结果](../dataset/data_audit.md)。

## 提交数据口径

官网样例列为 `Subject_id,Study_id,Predict_class,Probability,Gate`，最终格式以报名系统模板为准。每个 Study 覆盖 1–10 类，概率为 [0,1] 的有限数，保留官方 ID 原样。按当前 1,414 个测试 Study 计算，完整长表为 14,140 行。Gate 可留空或使用验证集确定的固定阈值。[官方提交格式样例](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261019/4dc233268b61702b4343f5a2d6385a78.png)
