# 2025 年起的语音语义判决参考集

正式队列移除 FSC，已冻结的历史实验保留。当前全量评测只含 SLURP 源任务和 MINDS 意图迁移；以下是新增候选，不代表已经本地验收或跑出效果。选择标准为基准首次公开于 2025 年起、语音输入、意义判决、可固定的有限答案和独立监督。原始录音或文本的年份与新任务发布年份分别看待。

| 用途 | 官方出处与已核验事实 | 采用边界 |
|---|---|---|
| 首选跨任务：FLEURS-SLU / SIB-FLEURS，2025 | [论文](https://arxiv.org/abs/2501.06117)、[分类数据](https://huggingface.co/datasets/WueNLP/sib-fleurs)、[作者代码](https://github.com/fdschmidt93/fleurs-slu)。将 FLEURS 朗读录音与 SIB-200 七类话题标注对齐，分类覆盖 102 种语言。HF 的 eng_Latn/test 为 177 个条目；同一文本条目可含多份录音，不能称作 177 条独立音频。 | 先英语，逐录音输入、七类完整面板、源域冻结，按原句聚类报告不确定性；监督字段用 category，不用网页 topic 或 URL。单句主题识别独立于 SLURP 命令意图。 |
| 新意图域：WolBanking77，2025 | [论文](https://arxiv.org/abs/2509.19271)、[作者仓库](https://github.com/abdoukarim/wolbanking77)、[音频与文本发布](https://huggingface.co/datasets/karim155/WolBanking77)。银行意图、沃洛夫语；9,791 是文本句数，论文报告语音超过四小时。音频目录有 train/test parquet。 | 使用前核验 audio/test 音频与意图标签能否按 ID 一对一绑定、实际覆盖类和说话人；不能用文本集的 77 类覆盖或分母代替。与 MINDS 同属银行域，且更换语言，不能把结果单独解释为语义能力。 |
| 细粒度跨任务：MMSU，2025 首发 / ICLR 2026 | [论文](https://arxiv.org/abs/2506.04779)、[作者数据](https://huggingface.co/datasets/ddwang2000/MMSU)。基准共 5,000 个问题、47 个子任务；已核对发布字段 audio、question、choice_a–d、answer_gt、task_name。论文含 Polysemy Reasoning。 | 优先预登记多义词语境判别，按官方任务名取完整子集、四选一；统计子集实际规模、答案能否唯一映射选项，不能拿原基准 5,000 作分母。数据托管 split 名 train，这是外部研究基准，不称作官方 TEST。排除纯声学识别、多步推理；题目依赖外部知识时单独说明。 |
| 后备：YUE-PUB-Speech，2026 | [作者数据](https://huggingface.co/datasets/Multilingual-NLP/YUE-PUB-Speech)、[论文](https://doi.org/10.21437/interspeech.2026-289)。粤语语用理解，1,680 个实例、14 个任务、4 位母语录音者。 | 直接/间接回应具有明确的语义决策意义，但依赖对话上下文、换语言且说话者少。延后，不把音频对应台词当附加文字输入。 |

协议：新旧端点使用同一音频、指令、原生候选与固定读出；全量成绩和概率质量来自同一遍预测。七类话题可报 Top-1 macro-F1、准确率、多类 NLL/Brier；逐题不同选项的 MMSU 报准确率、多类 NLL/Brier，不对 A/B/C/D 位置计算语义 macro-F1。不依赖目标测试标签改提示、阈值、温度。名单按研究问题区分，不将跨语言或同域新语料误称跨任务。

Speech-MASSIVE 为 2024 年工作，且 MASSIVE 由 SLURP 本地化得到；ATIS、Sonos SmartSpeaker 和 SLUE Phase 1 也早于本次时间窗，不作新增首选。没有把旧语料在 Hugging Face 的新版上传日期当首发日期。

## 本地 Jev 数据发布

两套可移植视图的目录由本地宏 SHARED_DATASETS 绑定。SIB 为 sib-fleurs-jev/eng-topic-v1（320 份录音、177 个原句、7 个话题）；MMSU 为 mmsu-jev/polysemy-v1（扫描全部 5,000 行，按精确 task_name=polysemy_reasoning 选中 98 个不同题目，4 个原始选项）。两者皆可提供 Noul 或 Choice + NONE，金标只在 scoring 子目录读取。MMSU 每题的题干和候选文本进上下文；A/B/C/D 仅标记本题的位置，不能拿位置的 macro-F1 当语义指标。SIB 保留原句组用于聚类区间。两者的监督都包含正确答案，NONE 仅能测误拒。

版本与入口：configs/portable/operations/semantic-transfer-sources.json；用 tools.reports.materialize_semantic_sources 核对固定 git revision 的普通文件及 LFS 字节。带 pyarrow、numpy、soundfile 和 scipy 的 Python 环境运行
`python -m tools.reports.build_semantic_transfer_views <config> <local-macros> sib` 或 `mmsu`。重复执行会在已发布目录全量复核。tools.reports.prepare_native_seal_cache.plan 以 MANIFEST.json 的 SHA256 和数据集 ID 加载相对路径、任务定义及逐题候选。当前发布是数据与 CPU 输入验收，未派发 NPU 评测，未报告模型效果。
