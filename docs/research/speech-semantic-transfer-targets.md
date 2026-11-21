# 2025 年起的语音语义判决参考集

SLURP 和 MINDS 分别用于源任务和同任务意图迁移。2026-11-21 评测范围收口：补充跨任务评测固定为 SIB-FLEURS 英语话题和 MMSU 多义词。FSC 与 WolBanking77 留在历史记录，YUE-PUB-Speech 不进入队列。已审核的分集结果见 [JEV-201](../../wiki/reports/JEV-201.md)。

| 用途 | 官方出处与已核验事实 | 采用边界 |
|---|---|---|
| 首选跨任务：FLEURS-SLU / SIB-FLEURS，2025 | [论文](https://arxiv.org/abs/2501.06117)、[分类数据](https://huggingface.co/datasets/WueNLP/sib-fleurs)、[作者代码](https://github.com/fdschmidt93/fleurs-slu)。将 FLEURS 朗读录音与 SIB-200 七类话题标注对齐，分类覆盖 102 种语言。HF 的 eng_Latn/test 为 177 个条目；同一文本条目可含多份录音，不能称作 177 条独立音频。 | 先英语，逐录音输入、七类完整面板、源域冻结，按原句聚类报告不确定性；监督字段用 category，不用网页 topic 或 URL。单句主题识别独立于 SLURP 命令意图。 |
| 细粒度跨任务：MMSU，2025 首发 / ICLR 2026 | [论文](https://arxiv.org/abs/2506.04779)、[作者数据](https://huggingface.co/datasets/ddwang2000/MMSU)。基准共 5,000 个问题、47 个子任务；已核对发布字段 audio、question、choice_a–d、answer_gt、task_name。论文含 Polysemy Reasoning。 | 优先预登记多义词语境判别，按官方任务名取完整子集、四选一；统计子集实际规模、答案能否唯一映射选项，不能拿原基准 5,000 作分母。数据托管 split 名 train，这是外部研究基准，不称作官方 TEST。排除纯声学识别、多步推理；题目依赖外部知识时单独说明。 |

协议：新旧端点使用同一音频、指令、原生候选与固定读出；全量成绩和概率质量来自同一遍预测。七类话题可报 Top-1 macro-F1、准确率、多类 NLL/Brier；逐题不同选项的 MMSU 报准确率、多类 NLL/Brier，不对 A/B/C/D 位置计算语义 macro-F1。不依赖目标测试标签改提示、阈值、温度。名单按研究问题区分，不将跨语言或同域新语料误称跨任务。

历史诊断：WolBanking77 [官方音频 TEST](https://huggingface.co/datasets/karim155/WolBanking77) 有 641 条录音、10 个意图、16 个说话人。它仍是银行意图判决，同时改变了语种、语料和录音条件，无法单独检验本轮想看的新判决能力。保留已审核的结果、封存面板和复现入口。

未纳入：YUE-PUB-Speech 的[作者音频仓](https://huggingface.co/datasets/Multilingual-NLP/YUE-PUB-Speech/tree/1d29fa1903aaee627ee946c2cfdf4b879f0347e1)与[作者代码仓](https://github.com/swaggy66/Yue-PUB-Speech/tree/b880751fc0e81896f28df1b50a55e4986a5c7283)分别公开了 1,680 个录音 ID 和 1,680 行含候选与金标的文本。14 个任务逐一核对，文本 ID 与录音 ID 无交集，未找到可核验的逐条对应表。

## 本地 Jev 数据发布

可移植视图的目录由本地宏 SHARED_DATASETS 绑定。SIB 为 sib-fleurs-jev/eng-topic-v1（320 份录音、177 个原句、7 个话题）；MMSU 为 mmsu-jev/polysemy-v1（扫描全部 5,000 行，按精确 task_name=polysemy_reasoning 选中 98 个不同题目，4 个原始选项）；WolBanking77 为 wolbanking77-jev/audio-test10-v1（641 份录音、10 个意图、16 个说话人）。三者皆可提供 Noul 或 Choice + NONE，金标只在 scoring 子目录读取。MMSU 每题的题干和候选文本进上下文；A/B/C/D 仅标记本题的位置，不能拿位置的 macro-F1 当语义指标。SIB 保留原句组用于聚类区间，WolBanking77 按说话人分组。三者的完整候选都包含正确答案，NONE 仅能测误拒。

版本与入口：configs/portable/operations/semantic-transfer-sources.json；用 tools.reports.materialize_semantic_sources 核对固定 git revision 的普通文件及 LFS 字节。带 pyarrow、numpy、soundfile 和 scipy 的 Python 环境运行
`python -m tools.reports.build_semantic_transfer_views <config> <local-macros> <sib|mmsu>`。重复执行会在已发布目录全量复核。WolBanking77 的构建参数 `wolbanking` 仅用于历史复现。评测入口是根目录 `run_experiment.py`，按台账登记的清单和端点执行。
