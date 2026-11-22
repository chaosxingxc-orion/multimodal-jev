# JEV-209　0.6B r128 知识卡与融合 TPO（真人版归档）

**归档结案。** 真人版 Stage1 完成 160 步并收齐五面板结果。原定 Stage2 由混合数据版接续。以下是该端点实测成绩。

| 面板 | 录音数 | Top-1 正确率 | Top-1 macro-F1 | 平均 Top‑1 概率 |
|---|---:|---:|---:|---:|
| SLURP_clean_TEST60 | 12,991 | 85.43% | 81.54% | 0.6922 |
| MINDS_native14 | 8,168 | 67.24% | 65.42% | 0.1952 |
| SIB_FLEURS_eng_topic7 | 320 | 69.69% | 65.75% | 0.1870 |
| MMSU_polysemy | 98 | 66.33% | — | 0.1577 |
| SLURP_clean_SELECTION60 | 9,226 | 86.51% | 83.43% | 0.6971 |

MINDS 的 macro-F1 为 65.42%，低于历史 206 的 71.89%。话题识别为 65.75%，MMSU 答对 65/98。该组为不同训练方案的端点比较。独立评分收据 SHA-256 为 `991bd1c14c2eb52b773200c2de3ddabd44a4c69f58d6cb520df20a2a5ff7a0ad`。以下登记内容按当时记录保留。


北京时间 2026-11-22 20:04，181 上的 Stage1 正式训练完成 160/160 步，八卡末点已提交并收到审核收据。20:16 核验时，全 TRAIN 参考打分、融合 Stage2 及五面板评测均待派发。[实验台账](../../experiment-ledger.md) · [训练与评测协议](protocol.md)

北京时间 2026-11-22 20:24 核验，Stage1 step 160 评测已由统一入口派发至 181。八个源 TRAIN 校准实例分别占用 0 至 7 卡，使用冻结的 21,244 条录音。阈值拟合后，八实例共享五面板的 966 个样本块，共 30,803 条留出录音。SLURP 使用冻结知识卡与同源示例避让规则，MINDS、SIB-FLEURS、MMSU 使用原生规约。派发提交 `5a75286`，运行包 `jev-card-eval-209-v1`。融合 Stage2 待讨论及派发。

| 封存面板 | 人口 | 历史 206 | 历史 208 | 新 Stage1 | 融合 Stage2 |
|---|---:|---:|---:|---:|---:|
| SLURP selection | 9,226 | 83.18% | 84.21% | 待测 | 待测 |
| SLURP TEST | 12,991 | 81.58% | 81.99% | 待测 | 待测 |
| MINDS native14 | 8,168 | 71.89% | 71.72% | 待测 | 待测 |
| SIB-FLEURS 英语话题 | 320 | 72.02% | 67.44% | 待测 | 待测 |
| MMSU 多义词 | 98 | 48/98 | 46/98 | 待测 | 待测 |

前四项为 Top-1 macro-F1，MMSU 为答对题数。历史 208 的源域 selection 点估计较高，但配对区间跨零，话题识别也出现退化，当前保留 206 为正式对照。[206 已审成绩](../../reports/JEV-206.md) · [208 已审成绩](../../reports/JEV-208.md)

执行后分别记录知识卡在双向混淆上的净修复和新增错例、融合训练的实际人口、间隔项激活数与卡时、相对新 Stage1 的配对增益和相对历史 206 的晋级结果。对 MINDS 单列跨数据集排序与概率变化，对 SIB-FLEURS 和 MMSU 分别解释跨任务表现，并列出分母和不确定性。

2026-11-22 正式训练启动。源 TRAIN 的 60 类共抽取 904 条文本标注，每类 2 至 16 条。按种子 317 为每条录音随机选取示例组合，两轮分别使用纯规则与带示例的版本。370 处同源或同文冲突均换用独立示例。正式第一个批次的 512 条输入均与冻结清单的长度一致，其中 263 条使用随机示例，共 32 种组合。

| 正式更新 | 实际录音数 | 八卡平均 loss | 学习率 | 梯度范数 |
|---:|---:|---:|---:|---:|
| 1 | 512 | 4.389775 | 0.0000125 | 8.01960 |
| 2 | 512 | 4.342903 | 0.0000250 | 5.98540 |
| 3 | 512 | 4.310467 | 0.0000375 | 4.96451 |

每步核对八个 rank 的 Adam 步数，并按实际样本数从 InfoNCE、概率间隔与双侧约束分项重算 loss。八卡有限验收另行完成两次更新，256 条配对的训推概率最大绝对差为 0。

运行包为 `jev-card-stage1-209-v3`，派发提交为 `d02a824`，正式容器为 `c606b8858fcff6d53e8059bbd6d7f583a422d20ae19fa2b25cd18ba8abb43b80`。本地运行根目录下 `ops/first-updates.json` 的 SHA-256 为 `00441e39275016263bd84ebe34b7087ed1e578047b84e40d4890c3d19f6e30a7`。181 上同名包的 `formal/run/rank-*/update-000{1,2,3}.json` 保存逐卡记录。

首次封包在 CPU 准备阶段停止。第二个包通过 CPU 核验后，启动器在隔离容器内解析主机名失败。第三个包改用单机回环地址，复用已核验的初始策略与随机示例缓存。

最终策略 ID 为 `090f33ca1671dc7bcdea549eba7330ee8b4a3f1b8b7329aabacb8e978d687fd3`，策略清单 SHA-256 为 `81068bf333bcaee6a8f85ec628e296b60adc09bafd28f98d9bccd0da3398d3af`。训练收据 `jev-card-stage1-209-v3/ops/formal-collected.json` 的 SHA-256 为 `3eafd817ddbf2cd03dad40a0c26bea627a10cf40573e2a04f64848101f84fa95`。

<!-- experiment-package:begin -->
## 实验入口执行记录

队列状态 **in_progress**。领取提交 `5a75286c9e64eb835a2241a91c44d3645708f240`，台账行 SHA-256 `b4c6ca6a6dcf0f4f27b9b3c3cb6978546c86861b8fbbef4fa4f5dc978ba25cc6`，协议 SHA-256 `12cf1516add130c1412189d9d31aecf6939d66fd9599b147a74aa81d1ca5e295`。
本次配置 SHA-256 `be382b1252d1d1d3ce53929277905454d56013f486bbb8361a728249b5d80a8b`。

| 阶段 | 步骤 | 状态 | 执行方式 | 证据或待办 |
| --- | --- | --- | --- | --- |
| data | freeze-card-training-inputs | audited | reused | `card-review.json` SHA-256 `a0770bbae48fa316f725a8464998d80ba40bf0739a4001d2303120cf1fcda386`；`operations-local.json` SHA-256 `e6f79cffa64d8918ee8948c8189973c532fe263c58d667c48cbecd41b5155f7e` |
| train | train-card-stage1 | audited | reused | `prepare-collected.json` SHA-256 `64e0f27b6cd3aa8f1203b389e3ea51f5fb81d756e6c3e7939669049a48383b1a`；`finite-collected.json` SHA-256 `8765efabda3bddc33d1c20148f253086adcd20480c0ede1ac34bef7e52e9db07`；`formal-collected.json` SHA-256 `3eafd817ddbf2cd03dad40a0c26bea627a10cf40573e2a04f64848101f84fa95` |
| train | evaluate-stage1-before-stage2 | audited | executed | `score-collected.json` SHA-256 `991bd1c14c2eb52b773200c2de3ddabd44a4c69f58d6cb520df20a2a5ff7a0ad` |
| train | score-full-train | pending | — | 待绑定已审 Stage1 step 160 的八实例全 TRAIN 固定参考评分与完整收据。 |
| train | build-fused-stage2-view | pending | — | 待参考打分后冻结一份分层难例与正确回放视图、单轮批序、按真实批计算的投影目标及预算。 |
| train | train-fused-tpo | pending | — | 待绑定六卡单轮融合 TPO 的有限验收、正式训练与末点审核。 |
| evaluate | audit-curves-and-card-effects | pending | — | 待绑定源域曲线、卡片配对诊断与源 TRAIN 阈值校准。 |
| evaluate | evaluate-stage1-and-fused-stage2 | pending | — | 待绑定新 Stage1 与融合 Stage2 的五面板配对评测、历史 206 与 208 对照及台账回填。 |

尚有预登记项目待审核，研究结论待完整证据。
<!-- experiment-package:end -->
