# CAFormer-S18 原生384：BCE、EMA与温和照片增强

本目录对应本轮 **fresh ImageNet-1K native384** 实验的三个独立交付模型。训练代码、阈值、验证指标和官方测试预测逐模型绑定；权重、训练数据、环境和NPZ不在仓库内。

|交付名|对应权重|验证Macro AUC|患者OOF F1|官方测试文件|
|---|---|---:|---:|---|
|BCE-Raw|A0_BCE_EMA/raw 第5轮|0.8074637227|0.4665867886|submissions/BCE-Raw/submission.csv|
|BCE-EMA|A0_BCE_EMA/ema 第5轮|0.8076126371|0.4596310207|submissions/BCE-EMA/submission.csv|
|Photo-Raw|A2_photo/raw 第5轮|0.8083684013|0.4664106733|submissions/Photo-Raw/submission.csv|

这些分数来自官方验证集859次Study、1422图、228名患者，**不是隐藏测试集成绩**。Photo-EMA是另一候选（AUC 0.8082075348、OOF F1 0.4780110207），本次不交付它的测试CSV；不能把其OOF F1拼到Photo-Raw上。原旧EMA参照AUC 0.8061307765 / OOF F1 0.4714205095仍保留，原目录和main未因本包改变。

本轮全部探索按用户条件早停后实际23轮、46份raw/EMA评估；这些是训练计数和验证记录数，不是隐藏test分数。提升点估计和配对区间仅供开发验证判断：使用了相同验证集筛选方案与选轮，bootstrap固定选中权重/OOF决策，没有重复选模或多重比较校正，不承诺统计显著提升或外部泛化领先。逐模型相对旧EMA的区间见 `validation_summary.json`。

## 文件与来源

- `train.py`：便携训练与CPU检查入口，仅支持本次A0_BCE_EMA与A2_photo两种5轮配方。
- `portable_model.py`：显式本地预训练路径；完整1000类strict加载后只替换最终fc2为10类。
- `portable_training.py`：本轮原训练循环、Study损失、EMA、数据加载、验证和阈值数学的独立副本。
- `data_pipeline.py`、`augmentation.py`：与冻结实验逐字节一致。
- `frozen_sources/`、`source_manifest.json`：原7模块的源码快照和SHA溯源；快照用于审核，便携入口不会导入原experiment_common/trainer或要求旧冠军、历史保护清单。
- `weights_manifest.json`：原生ImageNet来源与三个训练权重SHA；不包含任何权重。
- `thresholds_manifest.json`：三个权重各自的全验证Gate及患者OOF分数，class_id固定1..10。
- `inference/`：共享三模型推理/CPU校验入口，参数和使用方法见该目录说明。
- `submissions/BCE-Raw/`、`BCE-EMA/`、`Photo-Raw/`：各自的 `submission.csv`、模型配置与Gate。正式CSV由独立推理及审核通过后复制，不可互换权重或阈值。

## 复现训练

原环境Python 3.13.11、torch 2.10.0+cu128、torchvision 0.25.0+cu128、timm 1.0.22，其余版本见 `requirements.txt`。本包未包含或安装环境。请在自己的环境中准备合适的PyTorch CUDA构建及所列依赖；本轮本地整理只执行了CPU验收，没有重跑GPU训练。不同硬件/库构建不保证训练逐位相同。

准备官方manifest与原图根目录，并自行取得 [ImageNet-1K原生384预训练](https://huggingface.co/timm/caformer_s18.sail_in1k_384/tree/c46cc16d67a7d68806b7554b93e16b13506b7035)。必须是 `weights_manifest.json` 中SHA为 `629149fd...426806` 的 `model.safetensors`，不是旧CXR微调权重，也没有额外医学或ImageNet21k预训练。预训练上游许可为Apache-2.0。

manifest为官方CSV（SHA见代码），含 split、subject_id、study_id、image_path、label_ids；label_ids使用1..10并以 `|` 分隔。image_path相对 `--raw-root`。固定划分为train30344图/18513 Study/11055患者，val1422图/859 Study/228患者；test只读取元数据作划分检查，训练入口不读取test影像。数据与manifest请自行保存在包外。

从本目录执行以下单行命令，将路径替换为自己的绝对路径；Windows同样可用自己的盘符路径：

```sh
python train.py check --arm A0_BCE_EMA --manifest /data/manifest.csv --raw-root /data/cxr_first --pretrained /weights/model.safetensors --output /runs/cpu_check
python train.py train --arm A0_BCE_EMA --manifest /data/manifest.csv --raw-root /data/cxr_first --pretrained /weights/model.safetensors --output /runs/A0_BCE_EMA
python train.py train --arm A2_photo --manifest /data/manifest.csv --raw-root /data/cxr_first --pretrained /weights/model.safetensors --output /runs/A2_photo
```

中断后可在相同命令末尾加 `--resume`，恢复最后一个完整提交的训练轮，包含optimizer、GradScaler和EMA，最多总计5轮。若中断发生在轮内或验证中，未提交的该轮文件会阻止直接覆盖，需先由使用者检查并移出这些文件；不会把部分轮当完整轮。两个臂应使用不同output；output必须与源码、raw原图以及manifest/预训练文件目录分离。所有缓存和临时文件随output保存，便携入口没有E盘或旧实验目录依赖。

## 固定配方与评估

原图转RGB，等比全视野BICUBIC补边到384，ImageNet均值/方差归一化。训练轻仿射±5度、平移0.02、尺度0.95..1.05；无裁剪、翻转。A0使用none额外增强；A2 photo按各自0.5概率使用gamma/contrast 0.9..1.1、brightness 0.95..1.05，全零背景保持零；局部随机流由seed、epoch、图路径与操作名决定，不额外消耗全局随机流。验证/测试不增强。

完整2048隐藏MLP保留，骨干与隐藏头全部训练；仅末fc2重新初始化，偏置来自训练Study标签先验。seed42；每轮seed42+epoch。真实Study batch16、累积2，有效32，末17个Study按总数17归一化。所有有效图像完整通过分类器后，FP32 sigmoid概率取Study均值；按类别均值算BCE，再对Study等权，不能平均logit或embedding替代。

AdamW原骨干/隐藏头LR5e-5，新fc2 LR5e-4；bias/norm不衰减，其余0.05。hidden dropout0.2、outer dropout0、DropPath0.1、梯度检查点。每轮579次优化尝试：第1轮线性warmup系数0.1→1，后4轮cosine 1→0.1；AMP跳步仍推进尝试步调度。EMA仅在成功优化步更新，衰减min(0.999,(1+u)/(10+u))。

同一5轮训练同时维护raw和EMA，并分别按验证Macro AUC选轮，平局F1@0.5、再保留较早轮；不能把raw/EMA记为10轮训练。每臂输出完整history、检查点、验证预测、恢复状态及各自Gate。F1阈值网格0.025..0.975；患者GroupKFold5 shuffle seed42用于OOF，最终Gate则在全验证集拟合，两者评分含义分开。

## 预测文件

三份 `submission.csv` 是官方无标签test上的预测：1414个Study ×10类，共14140行，严格遵循官方模板ID顺序。模型分数、Gate、checkpoint SHA逐个配对；验证概率和隐藏测试真值均未放进本目录。不要根据这三份CSV本身声称隐藏测试AUC/F1。


共享推理入口从本目录执行（原始权重由使用者在包外提供，`--model`固定校验对应SHA）：

```sh
python -B inference/infer.py --model photo_raw --checkpoint /weights/Photo-Raw.pth --gates submissions/Photo-Raw/gates.json --raw-root /data/cxr_first --official-test-csv /data/official_test.csv --output-dir /runs/Photo-Raw-test
python -B inference/infer.py --model a0_raw --checkpoint /weights/BCE-Raw.pth --gates submissions/BCE-Raw/gates.json --raw-root /data/cxr_first --official-test-csv /data/official_test.csv --output-dir /runs/BCE-Raw-test
python -B inference/infer.py --model a0_ema --checkpoint /weights/BCE-EMA.pth --gates submissions/BCE-EMA/gates.json --raw-root /data/cxr_first --official-test-csv /data/official_test.csv --output-dir /runs/BCE-EMA-test
python -B inference/validate.py --model photo_raw --submission submissions/Photo-Raw/submission.csv --gates submissions/Photo-Raw/gates.json --official-test-csv /data/official_test.csv
```

`infer.py --check-only`仅做CPU权重及路径预检；可加 `--manifest`核对官方图像同序。validator提供 `--prediction-dir`时可重算本地FP32图→Study均值；没有本地预测时只检查CSV结构、ID顺序和Gate，不冒称重新核验概率。重新训练产生的checkpoint文件不保证拥有本次发布权重的相同文件SHA；三份已交付CSV对应的是manifest锁定的这三份原权重。
