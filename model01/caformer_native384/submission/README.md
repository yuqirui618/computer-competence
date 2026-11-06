# 正式测试预测

本目录的 `submission.csv` 来自原生384 CAFormer-S18第5轮，覆盖官方全部2,218张测试图、1,414次Study；每次检查10类，共14,140行数据。文件为UTF-8无BOM，列为 `Subject_id,Study_id,Predict_class,Probability,Gate`。

SHA256：`2b7ddeec8f605e1fef96807ecdd17d042a8ba06e2163d4a4433f12a35f38ba66`。

`verification.json` 为独立校验摘要：官方ID、类别完整性、阈值与权重绑定、图像概率按Study做float32均值、12位小数舍入均通过。测试集没有公开标签，未计算测试AUC或F1。本文件属于候选结果，未替换本地原冠军结果。

[官方CSV要求](https://www.ncccu.org.cn/index/Paper/case1.html)。本次验收范围为预测CSV及其来源一致性；比赛要求的合规材料和T4延迟另行验收。
