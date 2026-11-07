# model03：本轮CAFormer-S18原生384模型

[完整代码、复现命令与模型说明](CAFormer-S18-384/README.md)

|模型|验证Macro AUC|官方测试结果|
|---|---:|---|
|BCE普通权重第5轮|0.8074637227|[submission.csv](CAFormer-S18-384/submissions/BCE-Raw/submission.csv)|
|BCE EMA第5轮|0.8076126371|[submission.csv](CAFormer-S18-384/submissions/BCE-EMA/submission.csv)|
|照片增强普通权重第5轮|0.8083684013|[submission.csv](CAFormer-S18-384/submissions/Photo-Raw/submission.csv)|

三份文件均为官方五列、1414次检查×10类、14140行，并通过独立格式及逐图概率汇总校验。上述AUC均来自验证集，不是隐藏测试集成绩。代码与数据使用方法见模型README；原model01、model02和main保持原样。
