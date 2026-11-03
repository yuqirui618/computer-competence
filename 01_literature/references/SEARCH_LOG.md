# 检索与核查记录

日期：2026-11-02。最终任务：用户提供的 NCCCU 2026 人工智能赛题，10 类胸片多标签分类。用户尚未提供官方数据与团队服务器信息；本机只作算力观察，不能替代部署 T4。

## 范围与查询

检索主会页面（CVF、MICCAI、OpenReview）、作者 arXiv、官方 GitHub 和公开模型卡。代表查询：

- `site:openaccess.thecvf.com/content/CVPR2026 multi-label classification chest`
- `site:papers.miccai.org/miccai-2026 chest X-ray classification label`
- `Reference-Guided Gradient Balancing Loss`
- `MambaOut CVPR 2025 license github`
- `Edge-RecViT github`
- `ConvNeXt V2 license weights CC BY NC`
- `Vision Transformers Need Registers`
- `Asymmetric Loss ICCV 2021`

初始泛医学图像调研还检查了分割基础模型；正式赛题明确后，最终报告以分类为主，分割仅保留在不采用权重的说明中。没有把最先检索到的分割方向继续当成赛题。

## 核查动作

1. 阅读赛题全文；下载并提取 6 页规则、5 页补充说明，渲染关键页检查；目视核对官网类别与提交格式图片。
2. 核实 MambaOut、RepViT、ConvNeXt、ASL、DINOv2、RGB 和 Edge-RecViT 的作者入口；检查原始权重与医学衍生分支的区别。
3. HF API 核实 `timm/mambaout_tiny.in1k` 为非 gated、Apache-2.0，记录 revision 与文件名，但不下载模型大文件。
4. 针对 ConvNeXt V2、MambaVision，单独读取权重许可，确认不能以代码许可证代替权重许可证。

## 访问限制与证据边界

- 部分 CVF 页面直接打开返回 403，但正式检索结果与对应 PDF/作者来源提供了题名和发表信息；保留正式链接供人工复查。
- CHASE 会议页面列出的作者仓库本次返回 404，不能承诺可直接复现。
- RGB 仓库有代码和 GPL-3.0，但默认 README 示例依赖医学预训练；代码入口、损失模块及批量行为仍需实施时审计。
- GitHub 未认证 API 后续请求失败；相关 JSON 如实标记 `api_unavailable`。正式网页能访问的仓库仍可作为来源，不将失败解释为仓库不存在。
- 规则补充 PDF 在网页读取工具中失败，改用从官网所链接的同一公开 URL 下载并本地提取、渲染；未改用非官方规则。

这是定向调研，不声称穷尽 2026 全部顶会，也未进行性能实证。最终模型合规由赛方核验，性能由官方数据实验与部署实测确定。

## 官方数据获取与方向复核补充

2026-11-02：重新检查官方赛题页，公开页未列出数据包直链，考生入口要求 QQ 登录；已向用户索取官方后台下载链接或已下载文件路径。重新获取的规则与补充说明 PDF 哈希与先前一致。主候选固定版本的 model.safetensors 已无需认证成功下载，实际 106197008 字节，SHA-256 与发布方 LFS 记录一致，详见[权重记录](../../赛题相关资料/weight_metadata.json)。当日官方比赛数据仍未下载。

## 论文原文下载与 CCF-A 筛选补充

2026-11-02：按用户新要求暂缓官方数据集下载，完成原 16 篇论文 PDF 下载，并补充 ML-GCN、C-Tran、EfficientNetV2、DeiT、EfficientNet，共 21 篇。对照 CCF 官方人工智能在线目录确认 CVPR、ICCV、ICML 主会为 A 类，本次核实 15 篇，精选 12 篇；未核实等级的其他会议/期刊和纯预印本不计入 A 类。下载版本可为作者 arXiv 版，正式发表证据另行记录，不能把 arXiv 本身当作 A 类 venue。

21 份文件均检查 PDF 文件头、可解析页数、首标题覆盖和 SHA-256。全部原 16 篇及补充 5 篇首面已渲染检查身份。10 份 A 类论文有明确 CC 再分发许可，保持原始字节公开同步；其余 11 份未核实公开镜像许可，存于对应 local_only/ 并由 Git 忽略。OpenReview 页面/接口及直接 PDF 返回验证挑战，未尝试绕过，采用作者公开 arXiv 版本。来源、许可证与哈希见[下载清单](paper_downloads.csv)，署名见[第三方论文说明](../THIRD_PARTY_PAPERS.md)。

## 2026-11-03 数据接收与资料同步

接收E盘55,232,226,819字节原始ZIP及33,987份解压文件，核对33,984张JPG、3份CSV，实测Study数为18,513 / 859 / 1,414。上传3份赛题原件，原始ZIP在E盘分为52卷。仓库所有者明确确认主办方允许原始影像及CSV公开下载；授权文件未独立核验。按此范围以Release附件发布，不发布分享令牌、提取码和临时网页缓存。完成状态见[数据清单](../../03_datasets/official_dataset.json)。
