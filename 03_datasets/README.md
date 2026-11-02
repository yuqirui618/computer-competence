# 官方数据与提交结构

来源：[官方赛题](https://www.ncccu.org.cn/index/Paper/case1.html)，核查 2026-11-02。当前未取得参赛数据，以下按赛方页面与样例整理，下载后需用实际文件复核。

## 1. 唯一医学数据来源

| 划分 | 官网规模 | 用途 |
| --- | --- | --- |
| 训练集 | 18,513 次检查，含标签 | 监督训练；补充说明允许在官方训练图像上自监督训练 |
| 验证集 | 859 次检查，含标签 | 早停、选模型、选超参数/阈值；若最后并入训练，先冻结全部选择 |
| 测试集 | 1,414 次检查，无标签 | 最终固定前向预测与提交脚本检查 |

这些是**检查数，不是图像数或患者数**。一个 Study_id 可能有多张投照图；同一 Subject_id 也可能有多次检查。用官方清单检查三个划分是否存在患者重叠并如实记录，不自行改写赛方测试划分。

Kvasir、BUSI、MIMIC-CXR、CheXpert、医院数据等外部医学数据均不进入本赛的训练、自监督、伪标签和微调流程。不能因为数据未标注就当作允许。公开 GitHub 不上传官方数据、标签、患者标识和实际 submission.csv。

## 2. 10 类映射

以下逐项核对 [官网类别图片](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20260927/5385a992766a388da1480b3720b999a7.png)。机器可读版本：[class_mapping.csv](class_mapping.csv)。

| id | 中文 | 英文 |
| --- | --- | --- |
| 1 | 纵隔心影增大 | Enlarged Cardiomediastinum |
| 2 | 气胸 | Pneumothorax |
| 3 | 肺部实变 | Consolidation |
| 4 | 肺炎 | Pneumonia |
| 5 | 肺水肿 | Edema |
| 6 | 心影扩大 | Cardiomegaly |
| 7 | 肺不张 | Atelectasis |
| 8 | 肺部阴影 | Lung Opacity |
| 9 | 胸腔积液 | Pleural Effusion |
| 10 | 无异常发现 | No Finding |

No Finding 作为单独标签训练。是否施加互斥约束应先检查官方训练标签的一致性，不能直接令其概率等于 `1 - max(其他类别)`。

## 3. 需要建的本地清单

建议一张图一行：`Subject_id, Study_id, image_path, split, ViewPosition, PatientOrientation`；另建每个 Study 的 10 维标签表。保存图像尺寸、重复/近重复检查及缺失值情况。Support Devices 等元数据用于训练集内分层分析；其 -1/None 不能直接解释为疾病阴性。

训练增强建议从轻度几何与亮度变化起步，避免裁掉肺尖和肋膈角。胸片翻转、强旋转、重裁剪的影响需要验证；不能直接沿用自然图像分类默认强增强。归一化参数来自公开预处理配方或训练数据，不在测试集上重新拟合。

## 4. submission.csv

官网样例包含以下列，最终顺序和类型还需与报名系统实际模板核对：

```text
Subject_id,Study_id,Predict_class,Probability,Gate
```

样例图见 [官方提交格式](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261019/4dc233268b61702b4343f5a2d6385a78.png)。每个 Study 必须包含 1–10 全部类别，Probability 为 [0,1] 有限数。Gate 可空，默认 0.5；自定义阈值仅由验证集确定并固定。

若实际测试清单确实是 1,414 个唯一 Study，完整长表应有 **14,140 行数据**，不含表头。检查唯一键 `(Study_id, Predict_class)`、无缺漏/重复、Subject 关系匹配、UTF-8 和 ID 前导零。严禁用随机数或全零预测冒充已训练模型输出。
