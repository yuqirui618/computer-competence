# 胸片数据读取与评估

数据模块将官方图片和 CSV 关联成逐图清单，生成可选 JPEG 缓存，并提供训练、验证和推理共用的预处理。

## 运行

在已激活的数据环境中，从仓库根目录运行。`DATA_ROOT` 指向包含 `train`、`valid`、`test` 及对应 CSV 的官方解压目录，`DATA_OUT` 指向数据产物目录。

```powershell
python dataset/tools/audit.py --data-root $env:DATA_ROOT --bad-images (Join-Path $env:DATA_OUT 'bad_images.csv')
python dataset/tools/build_manifest.py --output (Join-Path $env:DATA_OUT 'manifest.csv')
python dataset/tools/build_cache.py --manifest (Join-Path $env:DATA_OUT 'manifest.csv') --output (Join-Path $env:DATA_OUT 'cache_512')
```

审计汇总见 [data_audit.md](data_audit.md)。坏图和缓存失败明细随本地数据产物保存。缓存先将短边缩放至 512 像素，再补黑边成正方形，以 JPEG 质量 92 保存。方图边长取缩放后的长边，可以大于 512，模型输入尺寸由 `img_size` 控制。同一缓存目录可以重跑，脚本会跳过可正常解码的已有方图。

## 输入与输出

`manifest.csv` 列为 `study_id,subject_id,image_path,split,label_ids,view_position`。`image_path` 相对于 `DATA_ROOT`，编号保留官方前缀，`valid` 在清单中记为 `val`，标签以 `|` 分隔。测试集的标签留空。

`dataset.build_loaders(cfg)` 返回训练、验证两个 DataLoader。配置包括 `data_root`、`manifest`、`cache_dir`、`img_size`、`batch_size` 和 `num_workers`。可用 `DATA_ROOT` 环境变量提供 `data_root`，并通过配置传入 `manifest` 和 `cache_dir` 的本机位置。读取到可用缓存时优先使用缓存，其他图片从原始目录读取。每个 batch 的 `image` 为 `(B,3,img_size,img_size)`，`target` 为 `(B,10)`，并带有 `study_id` 和 `subject_id`。

`list_images(root)` 支持整个数据目录或单个划分。`build_transform(False, img_size)` 是预测和评估共用的确定性预处理，按比例缩放、补边后应用 ImageNet 归一化。

训练和推理继续使用 `from dataset import CXRDataset, build_loaders, build_transform, list_images`。官方数据的下载和还原见[数据集发布说明](../03_datasets/DOWNLOAD_GUIDE.md)。

## 验收

DataLoader 的 `img_size=384`、`batch_size=8` 应输出 `(8,3,384,384)` 和 `(8,10)`。读取与评估需要 Python、PyTorch、torchvision、Pillow、NumPy 和 scikit-learn。若传入团队 YAML，评估由 `utils.load_config` 加载配置。`--ckpt` 入口调用 `model.build_model` 和 `model.load_checkpoint`。

```powershell
python evaluate.py --ckpt checkpoints/smoke.pth --split val
python evaluate.py --preds preds.csv --split val --manifest (Join-Path $env:DATA_OUT 'manifest.csv')
```

预测表沿用 `study_id,image_path,predict_class,probability`。评估同时输出逐图和 Study 平均概率的每类 AUC、macro AUC 和 macro F1。F1 默认阈值为 0.5。缺少正样本或负样本的类，其 AUC 显示 `N/A`，并从 macro AUC 的平均值中排除。macro F1 覆盖全部 10 类。

[测试日志](logs/test_20261103_1705.md)记录全量数据审计、真实 DataLoader 验收和合成评估验收。
