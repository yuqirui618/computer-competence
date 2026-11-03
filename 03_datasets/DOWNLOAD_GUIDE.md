# 官方数据存储与下载

2026-11-03 已核验并发布官方 ZIP。原始文件为 55,232,226,819 字节，包含 33,984 张 JPG 和 3 份 CSV。每卷长度、SHA-256 及附件地址见[发布记录](dataset_release_receipt.json)。

## 本地目录

数据、分卷、缓存及解压文件均使用E盘；整理完成后的目录为：

```text
E:\比赛数据集\
  downloads\cxr_first.zip    原始ZIP
  raw\cxr_first\
    train\                  30,344张图像
    valid\                   1,422张图像
    test\                    2,218张图像
    train.csv                30,344行，18,513次检查，含标签
    valid.csv                 1,422行，859次检查，含标签
    test.csv                  2,218行，1,414次检查，无疾病标签
  manifests\                正式校验和上传记录
  reference_materials\       赛题资料原件副本
```

原始ZIP自带`cxr_first/`层级。整理只改变外层存储位置，保留官方文件名、文件内容及划分。原始影像和ZIP不进入C盘的Git对象库。

## GitHub数据附件

55.23 GB 原始 ZIP 分为 52 个二进制分卷，每卷最多 1 GiB。全部分卷已核对长度和 SHA-256。[GitHub Release](https://github.com/yuqirui618/computer-competence/releases/tag/official-cxr-first-2026-11-03)

下载全部`cxr_first.zip.partNNNN`以及`DATASET_MANIFEST.json`、`restore_dataset.py`至E盘同一目录，再运行：

```powershell
python restore_dataset.py --parts-dir "E:\比赛数据集\release_downloads" --output "E:\比赛数据集\downloads\cxr_first.zip"
```

脚本边合并边核对每卷及完整ZIP的SHA-256；校验成功才生成最终ZIP，已有同名文件时停止。分卷不能分别解压。下载和还原ZIP共需约103GiB，解压另需约52GiB。

`git clone`获取资料、代码和校验清单，完整数据须从Release附件另行下载。[GitHub文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 校验和授权范围

已核对 ZIP 中央目录、解压文件路径与长度、CSV 行数和唯一 Study 数量，并计算原始 ZIP 和分卷的 SHA-256。解压后的逐图解码结果见[数据审计](../dataset/data_audit.md)。

[官方赛题第八部分](https://www.ncccu.org.cn/index/Paper/case1.html)要求数据传播取得允许。仓库所有者于2026-11-03明确确认，已有主办方授权将本届训练、验证、测试原始影像及3份CSV向互联网用户公开下载。本次发布依据此声明，授权文件未独立核验，不另行授予下游数据使用许可。

提取码、临时令牌、网页脚本/HTML缓存及含样本标识的临时工作清单不发布；上传凭据只在进程内使用。上传经校验成功后清理临时分卷及缓存。
