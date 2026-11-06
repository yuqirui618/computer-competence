# CAFormer-S18-384-EMA

本目录归档当前最高单模型验证点的相关代码与官方测试预测。模型采用完整CAFormer-S18、384输入和EMA权重，**验证AUC 0.8061307765，患者OOF F1 0.4714205095**，选中EMA续训第1轮（骨干累计第6轮）。相对原生CAFormer第5轮，AUC差值约+0.0013521，患者配对95%区间 **[-0.0008183885, +0.0034409897]跨0**，尚不能确认可靠提升；原有冠军、原生模型和历史提交均保留。

官方测试预测已完成并通过独立核验，唯一发布位置为 **[submission/submission.csv](submission/submission.csv)**：2218图、1414个Study、14140行，681599字节；SHA256为 **8c3749feda0a592c823df234b5ebc8e470701eb1fa7a5ebc1b322719562d524c**。它含连续概率和Gate，是测试预测而非原始数据；测试标签没有公开，不报告测试AUC/F1。详见同目录verification.json及reports/export_audit证据。

## 来源与方法

来源链固定为：作者ImageNet-1K原生384权重 → 官方训练集完整全骨干训练5轮，选中第5轮 → 从该完整权重独立BCE续训5轮并同步维护EMA，按验证AUC选中EMA第1轮。没有医学预训练权重、MambaOut热启动或特征缓存。第二阶段不是继续第一阶段的optimizer状态，而是同一全模型权重上新建AdamW和EMA。

原图RGB、双三次等比补边到384、ImageNet mean/std；训练仅轻微仿射，验证/测试确定性预处理。每张图经过完整原分类头再sigmoid，按Study作float32概率均值。Study等权BCE用稳定logsigmoid/logsumexp实现。第二阶段全骨干及原隐藏头可训练，隐藏Dropout0.2、外层Dropout0、DropPath0.1；AdamW lr1e-5、weight decay0.05，偏置/归一化无衰减。

第二阶段真实batch为16 Study，累积2次，有效32。每轮30344图、18513 Study、579个优化窗口（578×32+17）；最后两微批16和1都除以17。学习率首轮579次attempt暖启动，之后4轮余弦衰减；AMP跳步仍推进attempt日程，EMA只在成功optimizer更新后递推。EMA衰减为min(0.999,(1+u)/(10+u))，u仅计成功步。推理只加载一份选中的EMA完整权重，不需要运行时多模型融合。

OOF F1按患者GroupKFold5、shuffle=True、seed42分折选择10类阈值。最终全验证集Gate与OOF评分分开；AUC不依赖判定阈值。同一验证集多次选轮/选模型，区间也固定已选权重和阈值，未校正多次探索；需要新患者或独立测试才能确认泛化。

## 文件说明

|位置|用途|
|---|---|
|infer.py|本次固定发布EMA的完整验证复验及官方测试导出；原字节复用实际导出源码|
|gates.json、weights_manifest.json、pretrained_config.json|绑定发布EMA权重SHA、10类Gate和原生归一化|
|training/ema_continue.py、ema_core.py、ema_common.py|参数化BCE+EMA续训入口、实际训练数学及可恢复完整轮状态|
|training/data_pipeline.py、training_utils.py、ema_recipe.json|原图预处理、Study聚合、患者阈值、固定16×2配方|
|training/native_stage/|原生ImageNet384起点5轮的独立便携源码；提供完整前置来源链|
|reports/|A五轮raw/EMA明细、精炼方法比较和代码CPU核验|
|tools/validate_export.py|独立深度核验，需要本地infer输出的逐图/Study NPZ，不是只凭CSV重算模型|
|submission/|仅官方submission.csv及小型校验记录，不含原始数据|

完整权重、数据、环境、逐图预测NPZ不入Git。固定发布EMA权重SHA256：**f1f862bd6c76235e59e96b17b4bdae92aa497846b4eef42e8fb33b2b34ddee85**。gates.json保留原审计来源字段，实际输入路径以命令参数为准。requirements.txt列出本机验证过的版本；本次打包复用了既有E盘环境，没有安装新环境。

## 固定发布模型的预测复现

下面从模型目录运行；路径均为显式参数。请使用新输出目录和E盘既有Python，替换示例的数据/权重位置。infer.py严格核对已发布EMA、Gate和选中验证NPZ的SHA；先重放全部1422张验证图，核对后再预测全部2218张测试图。

~~~powershell
$py = 'E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe'
$raw = 'E:\your_data\cxr_first'
$manifest = 'E:\your_data\manifest.csv'
$ema = 'E:\your_runs\A_BCE_EMA\best_ema.pth'
$emaVal = 'E:\your_runs\A_BCE_EMA\predictions\ema_epoch_01.npz'
$out = 'E:\your_outputs\CAFormer-S18-384-EMA'
& $py -B .\infer.py --checkpoint $ema --gates-json .\gates.json --pretrained-config .\pretrained_config.json --manifest $manifest --data-root $raw --official-test-csv (Join-Path $raw 'test.csv') --validation-predictions $emaVal --output-dir $out --check-only
# 同一输出目录，去掉--check-only执行实际GPU验证及测试推理
& $py -B .\infer.py --checkpoint $ema --gates-json .\gates.json --pretrained-config .\pretrained_config.json --manifest $manifest --data-root $raw --official-test-csv (Join-Path $raw 'test.csv') --validation-predictions $emaVal --output-dir $out
# 官方任务描述明确要求文件名submission.csv；仅改名，CSV字节和预测不变
Copy-Item -LiteralPath (Join-Path $out 'CAFormer-S18-384-EMA.csv') -Destination (Join-Path $out 'submission.csv')
& $py -B .\tools\validate_export.py --output-dir $out
$modelCsvSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $out 'CAFormer-S18-384-EMA.csv')).Hash
$officialCsvSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $out 'submission.csv')).Hash
if ($modelCsvSHA -ne $officialCsvSHA) { throw '官方命名副本字节不一致' }
~~~

必须提交名为submission.csv的文件，模型名体现在外层目录。官方CSV五列依次为Subject_id,Study_id,Predict_class,Probability,Gate；1414个Study各10类，共14140行。Probability是连续概率，Gate是连续阈值，Predict_class是1到10的类别编号。深度校验使用本地原导出目录及NPZ；仓库只保留小型摘要和官方命名的最终CSV。

## 两阶段训练复现

先获取作者原生384 safetensors和官方数据/manifest。所有运行产物写自己指定的新目录。原生第一阶段入口如下，依次CPU预检、GPU资源预检、正式5轮：

~~~powershell
$nativeOut = 'E:\your_runs\native384'
$nativeWeights = 'E:\your_weights\caformer_s18_native384.safetensors'
& $py -B .\training\native_stage\train.py --data-root $raw --manifest $manifest --pretrained-weights $nativeWeights --output-dir $nativeOut --check-only
& $py -B .\training\native_stage\train.py --data-root $raw --manifest $manifest --pretrained-weights $nativeWeights --output-dir $nativeOut --smoke
& $py -B .\training\native_stage\train.py --data-root $raw --manifest $manifest --pretrained-weights $nativeWeights --output-dir $nativeOut
~~~

本次发布来源的原生最佳恰为第5轮。第二阶段只接受完整epoch5、10类分类头和对应原生ImageNet来源标识；用第5轮checkpoint及其验证预测衔接。重新训练的SHA可能不同，需显式绑定新文件，不能声称任意硬件都逐字节重现本次权重：

~~~powershell
$initial = Join-Path $nativeOut 'checkpoints\epoch_05.pth'
$initialVal = Join-Path $nativeOut 'predictions\epoch_05.npz'
$initialSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath $initial).Hash.ToLowerInvariant()
$initialValSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath $initialVal).Hash.ToLowerInvariant()
$emaOut = 'E:\your_runs\ema_continuation'
$trainArgs = @('--data-root',$raw,'--manifest',$manifest,'--initial-checkpoint',$initial,'--initial-predictions',$initialVal,'--initial-sha256',$initialSHA,'--initial-predictions-sha256',$initialValSHA,'--output-dir',$emaOut)
& $py -B .\training\ema_continue.py @trainArgs --check-only
& $py -B .\training\ema_continue.py @trainArgs --smoke
& $py -B .\training\ema_continue.py @trainArgs --run
~~~

第二阶段固定5个完整续训轮，raw/EMA两轨独立选轮，但只有5轮训练。支持完整轮保存后的恢复和pause_after_epoch.flag暂停；中断的未完成轮不计完成，出现未提交的同轮artifact会拒绝覆盖，应先人工归档。新运行生成的新权重必须配其新Gate并重新验证；根infer.py锁定本次已验收发布权重，不能把新模型配旧Gate或旧验证NPZ直接当作本次提交导出。

本轮还完成RLDAM5轮、GLoRI启发4轮（按用户C4不高于C3条件停止），总计14个完整新训练轮、19份验证结果。这里只提供精炼对比，不打包无关模型依赖或大量历史目录。详见reports/validation_metrics.json。T4延迟及其他参赛材料需按比赛要求另行验收。
