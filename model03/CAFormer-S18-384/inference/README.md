# 三份已训练CAFormer-S18-384的便携推理

仅支持本发布包注册的三份完整权重，不训练、不下载预训练权重。代码不依赖本机历史实验目录、验证NPZ、保护清单或训练runner。需使用仓库requirements.txt匹配的PyTorch/timm环境。

| --model | 权重轨迹 | 仓库提交子目录 |
|---|---|---|
| a0_raw | A0_BCE_EMA/raw，第5轮 | BCE-Raw |
| a0_ema | A0_BCE_EMA/ema，第5轮 | BCE-EMA |
| photo_raw | A2_photo/raw，第5轮 | Photo-Raw |

权重SHA及各自Gate固定在profiles.json。--gates必须传对应模型的JSON，文件里历史checkpoint路径不参与定位；真正权重由--checkpoint显式提供。完整训练权重已包含骨干与10类完整head，不需要再次下载ImageNet权重。

## 推理命令

从本模型目录运行，替换下面所有示例路径：

```powershell
python -B inference/infer.py --model photo_raw --checkpoint /path/to/photo_best_raw.pth --gates /path/to/Photo-Raw/gates.json --raw-root /path/to/cxr_first --official-test-csv /path/to/cxr_first/test.csv --output-dir /path/to/new_prediction_dir
```

另外两份使用--model a0_raw或a0_ema及其对应权重/Gate。输出目录须在源码包及原始数据之外，且不得已有预测结果。默认只接受被冻结SHA的原始官方test.csv，防止换错数据或顺序。原始文件为raw-root/test/Subject_id/Study_id/Dicom_id.jpg；同时识别jpeg/png并拒绝同ID歧义。

输出submission.csv（UTF-8无BOM，LF，14140行五列），以及本地image_probabilities.npz、study_probabilities.npz和prediction_receipt.json便于审计。NPZ是本地复算证据，无需上传GitHub或提交比赛。公开代码不附模型权重或原始影像。

## 数值合同

RGB完整视野、保宽高比bicubic黑边pad384，ImageNet均值/标准差；完整timm S18隐藏层2048分类头。全局图像batch固定4、最后2，CUDA FP16前向后先转float32再sigmoid，同Study各图概率按原图序做float32均值；按官方test.csv首次出现的Study排序，类1..10，各自固定Gate。无随机推理增强、无测试标签、无阈值拟合。

CPU已证明原冻结manifest的2218张test图顺序与原始official test.csv逐行完全相同，含末batch2。默认因此无需manifest。可额外传--manifest /path/to/original_manifest.csv核验同序；不会读取历史val预测。便携改造不声称重新训练或获得新分数。

## CPU预检与结果校验

推理命令追加--check-only，仅CPU严格加载完整权重并检查元数据/原图路径，不前向图片、不初始化GPU；会在新输出目录写preflight.json。正式推理可接着使用这个目录。

```powershell
python -B inference/validate.py --model photo_raw --submission /path/to/output/submission.csv --gates /path/to/Photo-Raw/gates.json --official-test-csv /path/to/test.csv
```

上例只验证格式、官方ID顺序和Gate，不能单独证明概率来自模型。若本地保留推理NPZ和receipt，追加--prediction-dir /path/to/output可独立重算每个Study的FP32均值并检验12位舍入前后阈值判定。追加--checkpoint PATH还可核验实际权重文件SHA。校验器只用CPU，不重新推理、不修改结果。

```powershell
python -B inference/check_cpu.py
```

该检查使用合成数据验证交错图像顺序/Study均值、官方列顺序、12位精度，并拒绝缺失或重复ID、错顺序、错Gate、NaN、错误概率等。可额外传官方数据及权重根路径作CPU完整检查；参数详见--help。运行检查时请让当前工作目录或--output-report所在目录位于有空间的数据盘，临时缓存位于该目录的cpu_check_temp。

## 来源

provenance.json记录实际Photo-Raw导出源SHA与改造范围。GPU前向与预处理数学未改变；本模块自身新增的是三模型选择、可移植路径和本地CPU审计。实际GPU测试导出及比赛格式校验由各提交目录收据另行证明。CPU重聚合与序列化对照不等于重新执行GPU前向。

## 已完成的CPU对照

三份原始GPU导出的2218张逐图概率均经本模块重新按Study做FP32均值，1414份检查的概率与原Study NPZ最大差为0；新序列化CSV与各原始提交文件逐字节一致。详见cpu_checks.json。三份完整checkpoint已分别严格CPU加载，分类头为10×2048，CUDA未初始化。Photo-Raw另用独立validator完整核验原NPZ，并验证错模型receipt会被拒绝。

| 模型 | 对应CSV SHA256 |
|---|---|
| BCE-Raw | ac769124d862d48c0bc94c093246be0ed378819a73ee559e534a860eb508c638 |
| BCE-EMA | 1ee3fdce8d54c3e353d13583cd8313dcc37b6db04fe614ffd55e4056597a56f3 |
| Photo-Raw | 4a1e0277ef1260bab1a519a31b99dbf08436acb0e99c4320aae4a5ce2c2218e8 |

这些对照证明本模块的CPU聚合、顺序与CSV序列化一致；不把它描述为已重新执行改造版CLI的GPU前向。
