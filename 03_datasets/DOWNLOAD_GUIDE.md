# 官方数据存储与下载

状态日期：2026-11-03。参赛者已下载并解压官方ZIP。原始文件为55,232,226,819字节，包含33,984张JPG和3份CSV。实际分卷发布状态见[数据清单](../06_references/official_dataset.json)的`publication.upload_status`。

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

55.23GB原始ZIP超出普通Git单文件100MiB及GitHub LFS单文件上限，采用[GitHub Release](https://github.com/yuqirui618/computer-competence/releases/tag/official-cxr-first-2026-11-03)附件存放52个二进制分卷，每卷最多1GiB。只有全部附件的长度和SHA-256核对后，才记录为`published_and_verified`；此前入口可能尚不可公开访问。

下载全部`cxr_first.zip.partNNNN`以及`DATASET_MANIFEST.json`、`restore_dataset.py`至E盘同一目录，再运行：

```powershell
python restore_dataset.py --parts-dir "E:\比赛数据集\release_downloads" --output "E:\比赛数据集\downloads\cxr_first.zip"
```

脚本边合并边核对每卷及完整ZIP的SHA-256；校验成功才生成最终ZIP，已有同名文件时停止。分卷不能分别解压。下载和还原ZIP共需约103GiB，解压另需约52GiB。

`git clone`获取资料、代码和校验清单，完整数据须从Release附件另行下载。[GitHub文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 校验和授权范围

已核对ZIP中央目录、全部解压文件的路径与长度，以及CSV行数和唯一Study数量；ZIP及分卷均计算SHA-256。尚未逐图检查像素或验证全部图像CRC，未训练模型或按测试数据调参。

[官方赛题第八部分](https://www.ncccu.org.cn/index/Paper/case1.html)要求数据传播取得允许。仓库所有者于2026-11-03明确确认，已有主办方授权将本届训练、验证、测试原始影像及3份CSV向互联网用户公开下载。本次发布依据此声明，授权文件未独立核验，不另行授予下游数据使用许可。

提取码、临时令牌、网页脚本/HTML缓存及含样本标识的临时工作清单不发布；上传凭据只在进程内使用。上传经校验成功后清理临时分卷及缓存。
