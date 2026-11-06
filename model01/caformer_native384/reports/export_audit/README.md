# 原始导出审计快照

本目录保留2026-11-06正式测试导出时的原始JSON字节，以便核对哈希。JSON中的本机E盘路径用于记录当时输入位置；使用便携入口时通过命令参数指定自己的路径。

- `export_receipt.json`：正式导出及验证集精确复现记录。
- `independent_validation.json`：独立CPU重建Study均值、官方ID、阈值与CSV舍入检查。
- `model_config.json`：当时权重、预处理、精度、聚合与数据清单绑定。
- `validation_recheck.json`：全部1,422张验证图的重算，图级与Study级差异均为0。

导出收据中的 `github_uploaded:false` 记录的是导出完成、尚未发布时的状态，保留原值以维持历史哈希。当前文件是否已发布以所在分支提交为准。

深度审计用到的逐图/Study NPZ保留在本地交付目录的work子目录。GitHub的 `submission/verification.json` 提供不依赖本机路径的摘要；轻量校验器只验结构/ID/阈值，不能单凭CSV重算模型预测。
