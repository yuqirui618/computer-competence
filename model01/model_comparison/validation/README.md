# 验证集预测结果：官方五列格式

**这里是验证集结果，不是可直接交到赛方的测试集submission.csv。** 每份包含859次验证检查×10类别＝8,590条数据，另加1行表头；正式测试集是1,414次检查×10＝14,140条数据。没有进行新的测试集推理。

精确表头：`Subject_id,Study_id,Predict_class,Probability,Gate`。依据仓库赛题相关资料的任务描述.docx内嵌示例和已核验的原submission.csv。CSV为UTF-8无BOM；保留官方ID字符串，按valid.csv首次Study出现顺序及类别1至10排序。概率与Gate用17位有效数字保存，独立回读后与原冻结概率数值完全一致。

| 文件key | 对应AUC最佳轮 | Study AUC | CSV最终Gate的F1（拟合内） | 患者五折OOF F1（内部估计） |
| --- | --- | --- | --- | --- |
| mambaout_original | protected_original | 0.80197696 | 0.46650799 | 0.45105496 |
| mambaout_gamma | 1 | 0.80217748 | 0.46751476 | 0.44249117 |
| convnext_tiny | 5 | 0.78945253 | 0.45381445 | 0.43090181 |
| efficientnetv2_s | 4 | 0.78514468 | 0.43879586 | 0.41389517 |
| caformer_s18 | 3 | 0.79119136 | 0.45069922 | 0.42973496 |
| densenet121 | 4 | 0.77265703 | 0.43613561 | 0.42159206 |
| hrnet_w18 | 5 | 0.78645840 | 0.44305432 | 0.42600438 |
| ensemble_caformer_s18 | 3 | 0.80307234 | 0.47507651 | 0.46097585 |
| ensemble_densenet121 | 4 | 0.79669534 | 0.46954591 | 0.44998822 |
| ensemble_hrnet_w18 | 5 | 0.80514173 | 0.46751400 | 0.43788878 |

`predictions/<key>.csv`保存五列预测；`gates/<key>.json`保存最终逐类阈值及权重SHA绑定；`export_manifest.json`记录全部输入/输出SHA和来源；`export_checks.json`记录独立回读的格式、AUC、F1及每类TP/FP/FN。

**F1口径必须分开。** CSV每一类在全部Study使用同一个最终Gate，来自该模型/组合在整个验证集上拟合的final_gates_by_class_id。它没有使用每折不同的OOF阈值，因此CSV回读的F1是拟合内分数，不能当作泛化提升。患者五折OOF F1另行列出，每折只在拟合患者选阈值；模型和轮次仍曾用同一验证集选择，所以也是内部估计，不能当作隐藏测试分数。AUC从原连续概率计算，不用Gate。

三个ensemble是现有MambaOut Gamma候选与对应顶刊模型的固定50:50概率均值，先各自按Study平均图像概率，再按模型均值。阈值JSON同时绑定两份checkpoint SHA及0.5/0.5权重，不能当成任何一个单模型的阈值。

导出前用本机官方valid.csv重新构建检查级10类真值，并逐项核对manifest和NPZ中的subject、Study、图像与标签。所有10份CSV均独立回读验证：无重复subject/Study/类别键，每个Study完整10类，概率有限且在[0,1]，各类Gate全局固定，指标和原预测一致。真值标签、逐患者NPZ和模型权重没有复制到此目录。

在原E盘实验产物齐全时，从仓库根目录重导出：

```powershell
& 'E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe' -B model01\model_comparison\tools\export_validation.py --data-root 'E:\比赛数据集' --output-dir 'E:\比赛数据集\验证结果重导出'
```

该工具仅使用numpy/scikit-learn及标准库，不导入训练模块、不使用GPU、不改原实验、不运行训练或测试推理。需要源实验的实际权重/NPZ以核对SHA，因此只克隆仓库不会自动得到这些本地产物。
