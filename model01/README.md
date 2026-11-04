# MambaOut-Tiny 本地实验

本目录保存已验证的 MambaOut-Tiny 实验代码。模型由 `timm` 提供，采用 ImageNet-1K 公开预训练权重，输出 10 类多标签概率，按 Study 平均图像概率进行评估。[权重来源](https://huggingface.co/timm/mambaout_tiny.in1k)

100 轮计划目前完成 45 轮并保持暂停。最佳为第 3 轮，Study Macro AUC 为 **0.80014564**，第 45 轮为 **0.70247655**。后续轮次和最终阈值校准待运行。准备工具只生成复现实验文件。

| 路径 | 内容 |
| --- | --- |
| `src/run_local.py` | 小样本试跑、验证、100 图样例 CSV 与测速 |
| `src/train_full.py` | 官方全量训练集的 5 轮基线 |
| `src/train_and_optimize.py` | 100 轮训练、最佳权重复核与阈值校准 |
| `tools/prepare_local.py` | 在新的 E 盘目录生成原实验目录结构 |
| `configs/data_local.example.yaml` | 本机数据路径示例 |
| `licenses/` | 预训练权重来源、校验信息和上游许可证 |
| `reports/experiment_summary.json` | 当前暂停状态与指标摘要 |
| `reports/training_history.csv` | 已完成轮次的训练和验证指标 |
| `requirements.txt` | 本机实测依赖版本 |

## 环境与工作区

这些脚本保留原实验的 Windows、CUDA 和 E 盘要求。本机使用 Python 3.13、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128 和 timm 1.0.22。以下 PowerShell 命令从仓库根目录执行，虚拟环境放在 E 盘。

```powershell
$setupCache = 'E:\比赛数据集\temp\model01-setup'
New-Item -ItemType Directory -Path $setupCache -Force | Out-Null
$env:TEMP = $setupCache
$env:TMP = $setupCache
$env:TMPDIR = $setupCache
$env:PIP_CACHE_DIR = Join-Path $setupCache 'pip'
py -3.13 -m venv 'E:\比赛数据集\envs\mambaout-model01'
$python = 'E:\比赛数据集\envs\mambaout-model01\Scripts\python.exe'
& $python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r model01\requirements.txt
```

先按[数据模块说明](../dataset/README.md)准备官方数据和清单。准备工具默认读取配置示例中的当前 E 盘数据位置，可用 `--config` 指定另一份 YAML，也可通过 `--data-root`、`--manifest`、`--cache-dir` 覆盖路径。未使用缓存时将配置中的 `cache_dir` 设为 `null`。

`--work-dir` 必须显式指定新的 E 盘目录，避开原来暂停中的三个实验目录。`--weights` 指向已下载的本地预训练文件，工具会按 metadata 检查 SHA256。

```powershell
$work = 'E:\比赛数据集\复现实验\model01'
& $python model01\tools\prepare_local.py --work-dir $work --weights 'E:\比赛数据集\本地跑通\pretrain_model\model.safetensors' --data-root 'E:\比赛数据集\raw\cxr_first' --manifest 'E:\比赛数据集\喻数据产物\manifest.csv' --cache-dir 'E:\比赛数据集\喻数据产物\cache_512'
```

工具复制三份脚本、仓库 `dataset/`、类别表、配置及权重来源文件，并生成 `本地跑通/`、`全量基线/`、`MambaOut_100轮/` 和 `start.ps1`。仓库提供代码、来源与实验摘要，数据、权重、检查点和运行副本保留在 E 盘。

## 主动重跑实验

以下命令会开始训练。启动器将临时目录与 CUDA、Hugging Face、PyTorch 缓存放到 E 盘，并使用 `workers=0`。按顺序执行，小样本阶段默认使用 2,000 张图、2 轮、224 输入和 batch 12。

```powershell
& (Join-Path $work 'start.ps1') -Phase smoke
& (Join-Path $work 'start.ps1') -Phase baseline
& (Join-Path $work 'start.ps1') -Phase long
```

5 轮阶段读取试跑生成的 `本地跑通/outputs/metrics.json`。100 轮阶段完成后读取 `全量基线/results.json` 和 `全量基线/checkpoints/best.pth` 比较历史基线。每轮使用全部 30,344 张训练图，验证集为 1,422 张图、859 次 Study。按 Study Macro AUC 优先、固定阈值 0.5 的 Macro F1 次优选择权重。

新工作区从头运行，原实验第 45 轮的检查点继续保留在原目录。脚本在各自运行目录保存检查点、历史与状态。

100 轮完成后，脚本才会按患者分组进行 5 折逐类阈值校准。轮次选择已使用整个官方验证集，分折 F1 仍属于内部估计。阈值用于分类决策，AUC 始终从概率计算。试跑输出的 100 图 CSV 和测速用于本地格式与一致性检查，只覆盖样例检查，延迟以赛方设备实测为准。
