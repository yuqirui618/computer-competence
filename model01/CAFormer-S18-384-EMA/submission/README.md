# 官方测试预测

仅提交本目录的 **submission.csv**。官方任务描述要求文件名为submission.csv、编码UTF-8；模型名体现在外层CAFormer-S18-384-EMA目录。

本文件已通过完整推理及独立CPU校验：2218张测试图、1414个Study、每Study10类，共14140行；五列顺序、官方ID/顺序、float32图概率均值及Gate均已核对。无测试标签，不报告测试AUC/F1。

SHA256：8c3749feda0a592c823df234b5ebc8e470701eb1fa7a5ebc1b322719562d524c，681599字节。源推理曾生成模型命名CSV，本文件与其逐字节相同；仓库只保留官方命名这一份预测文件。

verification.json为精简摘要，reports/export_audit保存三份原始证据快照。快照内本机路径用于追溯；github_uploaded:false表示导出时尚未上传，不代表最终分支状态。逐图NPZ和权重仍在本地，不在Git中。
