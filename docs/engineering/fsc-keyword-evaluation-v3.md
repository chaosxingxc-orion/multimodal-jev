# FSC 词级线索评测 v3

词表和整句示例都由 `tools.reports.fsc_annotation_cues` 从官方 TRAIN 文本标注统计得出。官方数据说明将 CSV 中的 `transcription` 定义为采集时呈现给说话人的书面提示。脚本使用 23,132 条录音对应的 248 种文本标注、77 位说话人及原生三元组标签。另见[数据集原始说明](/mnt/cephfs/xingchao/datasets/fsc/extracted/fluent_speech_commands_dataset/readme.md)。

词级线索按归一化后的不同文本计数，按说话人数过滤。对每个字段，优先比较只差该字段的合法类别。没有这种近邻时，先找共享动作或对象的对照，再按字段全局比较。候选线索按正负出现率差排序，每个字段最多保留三个单词。同一词只归到对照区分度最高的字段，`none` 不分配关键词。完整句从该类已有的短文本标注中确定性选择。词频、说话人数及对照分母见词表审计。[词表审计](/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-keyword-evaluation-v3/annotation-cues.json)推理时 Jev 直接听音频。四组分别使用原定义、原定义加词、原定义加句、原定义加词和句。样本、31 类语义与顺序、读出标记和 step 160 策略相同。

| 阶段 | 执行与产物 |
| --- | --- |
| 数据封存 | 官方 VALID 全量 3,118 条、10 位说话人、31 类。音频引用共享原 WAV，四组共用同一音频清单和计分文件。FSC TEST 保留作已有冻结迁移对照。 |
| CPU 准备 | 分组使用 `tools.reports.prepare_native_seal_cache` 与 `configs/portable/operations/native-seal-cache.json` 生成有限及全量缓存，再用同入口独立 `--verify`。按既有入口将复核收据保存为 `VERIFIED.json`。 |
| 目标机准入 | 181 上每组先对同一批 256 条无标签 VALID 样本运行 HF 和对应的两实例服务对照。每组分别使用自身的提示缓存。原生运行器按请求、候选顺序、logit、设备和运行时身份验收。若失败，保留证据，暂停四组正式比较。 |
| 全量推理 | 每组两名单卡服务副本，四组共八卡。核对目标机设备、镜像和租约后统一派发。独立 CPU 审核和评分不持卡租约。每条录音只使用所属组的一个副本，主比较共 12,472 次请求。 |
| 配对比较 | 主对照为关键词组减原定义组。主报 Top‑1 macro‑F1、只差动作的正确候选胜率、只差地点的正确候选胜率。并报准确率、平均 Top‑1 原始概率及只差对象的胜率。对全部合法单字段邻居计分，平分按半次胜出。按说话人配对 bootstrap，三项主指标的区间下界均大于零才判为有信号。 |

准备入口为 `python -m tools.reports.fsc_keyword_evaluation --config configs/portable/operations/fsc-keyword-evaluation.json --local <本地宏>`。本地宏绑定 `FSC_DATASET_ROOT`、`FSC_KEYWORD_SOURCE_PLAN`、`FSC_KEYWORD_EVALUATION_ROOT`。缓存宏 `NATIVE_SEAL_CACHE_CONFIG` 每次绑定一个 `MANIFEST.json` 中的 dataset key、seal 根及 SHA-256、step 160 的处理器及其 SHA-256、输出根、编码配置和并行参数。面板另填 `panel_size=256`。正式收集使用 `--results-local <本地宏>`，以 `FSC_KEYWORD_RESULTS` 绑定 `seal: {root, sha256}`、四组 `arms: {组名: {audit, audit_sha256, predictions}}` 及 `report` 路径。

封存 SHA-256 为 `5f5c2f9d4f7f80c07057bdfc891840a1abe0a52f5af33401a1aab02c5fb17153`。 [共享评测集清单](/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-keyword-evaluation-v3/MANIFEST.json)

CPU 复用了 step 160 的真实处理器。最长 VALID 录音在四组分别编码为 2529、3178、2881、3530 tokens，均保留 31 个读出标记，音频特征逐元素一致。v3 是当前发布入口。v1 手写词表和 v2 初版统计产物均未派发。当前证据覆盖数据和 CPU 输入编译，NPU 推理和效果待运行。
