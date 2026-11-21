# 2025 年起的语音语义判决参考集

正式队列移除 FSC，已冻结的历史实验保留。SLURP 和 MINDS 分别用于源任务和意图迁移评测。SIB-FLEURS 英语话题、MMSU 多义词和 WolBanking77 官方音频 TEST 纳入 [JEV-201](../../wiki/reports/JEV-201.md)。选择标准为基准首次公开于 2025 年起、语音输入、意义判决、可固定的有限答案和独立监督。原始录音或文本的年份与新任务发布年份分别看待。

| 用途 | 官方出处与已核验事实 | 采用边界 |
|---|---|---|
| 首选跨任务：FLEURS-SLU / SIB-FLEURS，2025 | [论文](https://arxiv.org/abs/2501.06117)、[分类数据](https://huggingface.co/datasets/WueNLP/sib-fleurs)、[作者代码](https://github.com/fdschmidt93/fleurs-slu)。将 FLEURS 朗读录音与 SIB-200 七类话题标注对齐，分类覆盖 102 种语言。HF 的 eng_Latn/test 为 177 个条目；同一文本条目可含多份录音，不能称作 177 条独立音频。 | 先英语，逐录音输入、七类完整面板、源域冻结，按原句聚类报告不确定性；监督字段用 category，不用网页 topic 或 URL。单句主题识别独立于 SLURP 命令意图。 |
| 跨数据集、跨语言：WolBanking77，2025 | [论文](https://arxiv.org/abs/2509.19271)、[作者仓库](https://github.com/abdoukarim/wolbanking77)、[音频与文本发布](https://huggingface.co/datasets/karim155/WolBanking77)。官方 audio/test 为 641 条录音、10 个意图、16 个说话人。音频、意图与说话人 ID 由同一条源记录提供。 | 候选来自官方音频问题词表，模型直接听沃洛夫语音频，不提供对应文字。报告 Top-1 macro-F1、准确率、平均最高概率及按说话人分组的配对区间。与 MINDS 同属银行域，语种差异和语义差异无法单独归因。 |
| 细粒度跨任务：MMSU，2025 首发 / ICLR 2026 | [论文](https://arxiv.org/abs/2506.04779)、[作者数据](https://huggingface.co/datasets/ddwang2000/MMSU)。基准共 5,000 个问题、47 个子任务；已核对发布字段 audio、question、choice_a–d、answer_gt、task_name。论文含 Polysemy Reasoning。 | 优先预登记多义词语境判别，按官方任务名取完整子集、四选一；统计子集实际规模、答案能否唯一映射选项，不能拿原基准 5,000 作分母。数据托管 split 名 train，这是外部研究基准，不称作官方 TEST。排除纯声学识别、多步推理；题目依赖外部知识时单独说明。 |
| 待标注发布：YUE-PUB-Speech，2026 | [作者音频仓](https://huggingface.co/datasets/Multilingual-NLP/YUE-PUB-Speech/tree/1d29fa1903aaee627ee946c2cfdf4b879f0347e1)、[作者代码仓](https://github.com/swaggy66/Yue-PUB-Speech/tree/b880751fc0e81896f28df1b50a55e4986a5c7283)、[论文](https://doi.org/10.21437/interspeech.2026-289)。粤语语用理解，论文报告 1,680 个实例、14 个任务。 | 核对上述发布时，音频仓缺少 README 所述的 `data/`，作者代码仓仅提供逐任务样例。完整录音与题目、候选、金标无法独立绑定，暂不派发全量评测。 |

协议：新旧端点使用同一音频、指令、原生候选与固定读出；全量成绩和概率质量来自同一遍预测。七类话题可报 Top-1 macro-F1、准确率、多类 NLL/Brier；逐题不同选项的 MMSU 报准确率、多类 NLL/Brier，不对 A/B/C/D 位置计算语义 macro-F1。不依赖目标测试标签改提示、阈值、温度。名单按研究问题区分，不将跨语言或同域新语料误称跨任务。

Speech-MASSIVE 为 2024 年工作，且 MASSIVE 由 SLURP 本地化得到；ATIS、Sonos SmartSpeaker 和 SLUE Phase 1 也早于本次时间窗，不作新增首选。没有把旧语料在 Hugging Face 的新版上传日期当首发日期。

## 本地 Jev 数据发布

可移植视图的目录由本地宏 SHARED_DATASETS 绑定。SIB 为 sib-fleurs-jev/eng-topic-v1（320 份录音、177 个原句、7 个话题）；MMSU 为 mmsu-jev/polysemy-v1（扫描全部 5,000 行，按精确 task_name=polysemy_reasoning 选中 98 个不同题目，4 个原始选项）；WolBanking77 为 wolbanking77-jev/audio-test10-v1（641 份录音、10 个意图、16 个说话人）。三者皆可提供 Noul 或 Choice + NONE，金标只在 scoring 子目录读取。MMSU 每题的题干和候选文本进上下文；A/B/C/D 仅标记本题的位置，不能拿位置的 macro-F1 当语义指标。SIB 保留原句组用于聚类区间，WolBanking77 按说话人分组。三者的完整候选都包含正确答案，NONE 仅能测误拒。

版本与入口：configs/portable/operations/semantic-transfer-sources.json；用 tools.reports.materialize_semantic_sources 核对固定 git revision 的普通文件及 LFS 字节。带 pyarrow、numpy、soundfile 和 scipy 的 Python 环境运行
`python -m tools.reports.build_semantic_transfer_views <config> <local-macros> <sib|mmsu|wolbanking>`。重复执行会在已发布目录全量复核。评测入口为 `./run_experiment.py resume --experiment JEV-201`，按已封存的清单和登记的端点执行。
