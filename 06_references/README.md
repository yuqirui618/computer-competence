# 来源、引用与核查记录

| 文件 | 用途 |
| --- | --- |
| [papers.csv](papers.csv) | L01–L16 的完整题名、年份、发表状态、论文/代码链接与复用结论 |
| [references.bib](references.bib) | 16 项文献的 BibTeX；不编造未核实页码或 DOI |
| [COMPLIANCE_REVIEW.md](COMPLIANCE_REVIEW.md) | 推荐路线逐项复核、真实权重文件大小与哈希、仍待验证事项 |
| [COMPETITION_RULES.md](COMPETITION_RULES.md) | 官方赛题、规则和补充说明及其对技术方案的影响 |
| [SEARCH_LOG.md](SEARCH_LOG.md) | 检索范围、关键查询、核查方式和访问限制 |
| [upstream_repositories.json](upstream_repositories.json) | 上游仓库核查结果；API 失败不等于仓库不存在 |
| [weight_metadata.json](weight_metadata.json) | 模型卡 revision、许可与 gated 状态，以及实际下载权重的大小、SHA-256 和发布方校验结果 |

资料区分三层证据：论文正式页面确认发表状态，作者代码/模型卡确认公开材料，官方比赛规则决定可用范围。对未访问成功的链接保留限制说明，不把搜索到仓库名称当作完整复现证明。

主候选公开权重已下载，SHA-256 与发布方一致；官方数据仍未取得，因此没有实际数据核查结果或训练分数。引用采用公开链接与自主笔记；公开仓库不复制论文全文。
