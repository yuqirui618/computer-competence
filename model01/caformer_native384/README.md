# CAFormer-S18 原生384全骨干候选

本目录保存最新原生384模型的训练/预测代码、配置、五轮验证结果，以及官方完整测试集的预测文件。所有文件单独归档在个人分支，不替换原MambaOut冠军和HRNet融合结果。

本次从作者ImageNet-1K原生384预训练重新开始；仅将1000类末层替换为10类，stem、四个stage、原隐藏分类头全部参与训练。直接读取官方原图，RGB、双三次等比补边到384、ImageNet归一化；每张图经过完整分类头后做sigmoid，再按Study取float32概率均值。训练使用等Study权重的稳定BCE、真实32 Study batch、AdamW、隐藏头Dropout 0.2和DropPath 0.1，完整训练5轮。

最佳按验证AUC选择为第5轮：**AUC 0.8047787005，患者分折阈值F1 0.4578443406**。相对原保留MambaOut的AUC差值95%区间跨0，仍作为候选保留；详见[验证总结](reports/validation_summary.md)。

## 文件位置

| 文件或目录 | 用途 |
| --- | --- |
| `train.py`、`training_core.py`、`common.py`、`data_pipeline.py` | 参数化路径的全骨干训练；核心数学和预处理沿用真实实验 |
| `infer.py` | 完整checkpoint推理；支持官方test/val，图像microbatch 4，Study概率均值 |
| `recipe.json`、`pretrained_config.json` | 原训练配方、模型归一化与384配置 |
| `gates.json` | 与发布的第5轮权重SHA绑定的10类阈值 |
| `weights_manifest.json`、`source_provenance.json` | 权重来源、许可证、原始与发布代码的对应关系 |
| `requirements.txt` | 本机验证过的依赖版本 |
| `tools/check_submission.py` | 只需官方test元数据的CSV结构、ID和阈值检查 |
| `tools/export_released_model.py`、`tools/validate_export.py` | 已实际执行的严格导出与独立深度核验；需要本地验证预测和测试NPZ |
| `reports/` | 五轮明细、验证总结、代码验证与导出审计证据 |
| `submission/` | 正式测试 `submission.csv` 和独立校验摘要，非859次Study的验证CSV |

权重、原始影像、运行环境及逐图预测NPZ保留在本地，没有加入Git。完整权重为本地 `CAFormer384_原生全骨干_20261106/best.pth`，SHA256为 `e4396db33d9fa25751205fc4d848766ff894ec4d104d884b77847e4afd01b250`。推理只需这份完整权重，不依赖旧MambaOut或CAFormer特征缓存。作者预训练来源为 [timm/caformer_s18.sail_in1k_384](https://huggingface.co/timm/caformer_s18.sail_in1k_384)，固定来源与许可证见weights_manifest。

## 本机预测和检查

以下PowerShell示例从仓库根目录执行，复用既有E盘环境；输出与临时缓存均放E盘。输出目录须使用新的路径，已有结果会被拒绝覆盖。

```powershell
$cxrPython = 'E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe'
$modelDir = Join-Path (Get-Location) 'model01\caformer_native384'
$rawData = 'E:\比赛数据集\raw\cxr_first'
$manifestFile = 'E:\比赛数据集\喻数据产物\manifest.csv'
$checkpointFile = 'E:\比赛数据集\CAFormer384_原生全骨干_20261106\best.pth'
$gatesFile = Join-Path $modelDir 'gates.json'
$predictionDir = 'E:\比赛数据集\复现输出\CAFormer384_test'
& $cxrPython -B (Join-Path $modelDir 'infer.py') --checkpoint $checkpointFile --gates $gatesFile --data-root $rawData --manifest $manifestFile --official-test-csv (Join-Path $rawData 'test.csv') --output-dir $predictionDir
& $cxrPython -B (Join-Path $modelDir 'tools\check_submission.py') --submission (Join-Path $predictionDir 'submission.csv') --official-test-csv (Join-Path $rawData 'test.csv') --gates $gatesFile --checkpoint $checkpointFile
```

对仓库已交付文件做轻量验收，可把第二条命令的submission路径换成 `$modelDir\submission\submission.csv`。该检查只验结构/ID/阈值，无法单凭CSV复算模型。深度核验记录在[export_audit](reports/export_audit/README.md)；便携预测入口也在全部2,218张测试图上与正式导出逐值核对，图级、Study级概率差均0，CSV逐字节一致。该一致性是在本机相同依赖/CUDA配置下核验的。

## 从原生预训练重新训练

使用新的输出目录，按顺序执行CPU预检、GPU资源试跑、正式训练。预检和资源试跑不计训练轮次；正式入口固定5轮。下面命令仅供主动复现，本次打包没有新增训练轮次。

```powershell
$trainDir = 'E:\比赛数据集\复现实验\CAFormer384_native'
$nativeWeights = 'E:\比赛数据集\CAFormer384_原生全骨干_20261106\pretrained\model.safetensors'
& $cxrPython -B (Join-Path $modelDir 'train.py') --data-root $rawData --manifest $manifestFile --pretrained-weights $nativeWeights --output-dir $trainDir --check-only
& $cxrPython -B (Join-Path $modelDir 'train.py') --data-root $rawData --manifest $manifestFile --pretrained-weights $nativeWeights --output-dir $trainDir --smoke
& $cxrPython -B (Join-Path $modelDir 'train.py') --data-root $rawData --manifest $manifestFile --pretrained-weights $nativeWeights --output-dir $trainDir
```

原生预训练SHA为 `629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806`。代码拒绝把此前胸片训练权重用作这次原生初始化。新训练结束后，推理必须同时使用新输出目录的 `best.pth` 和 `gates.json`，不能配用这里的固定发布阈值。

正式测试文件遵循[官方提交格式](https://www.ncccu.org.cn/index/Paper/case1.html)：UTF-8、每Study完整10类概率和Gate。测试标签未公开，未计算测试AUC/F1；本目录的CSV核验不代表完成T4延迟及全部合规材料验收。
