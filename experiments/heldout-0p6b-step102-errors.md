# 0.6B：SLURP 与 FSC 终点评分、校准与错误归因

对象为 Qwen3-ASR-0.6B 的 Stage1 rc7 step 318 → 同轮 Stage2 step 102。对每份原始预测核对 SHA-256、策略身份、候选顺序、封存标签及逐录音 ID，复算整张混淆矩阵。评测以录音为单位；并列按冻结候选顺序。此处比较的是同一意图成员判决任务的跨数据集迁移。没有用测试标签拟合任何参数，也没有调用模型审核或重新推理。

| 同一人口的两个端点 | Top-1 macro-F1 | Top-1 正确率 | 最高概率均值 | 两端都错 / 改对 / 新错 |
| --- | ---: | ---: | ---: | ---: |
| SLURP clean TEST60，12,991 | 76.63% → 78.25% | 80.92% → 82.17% | 70.45% → 89.15% | 2,008 / 471 / 308 |
| FSC native31 TEST，3,793 | 0.56% → 2.78% | 8.38% → 8.91% | 37.94% → 46.80% | 3,358 / 117 / 97 |

**FSC 读数如何刷新。** v159 是 Stage1 step 0 / 318 的原始 Top-1 重算；不含 Stage2，不会重新拟合校准器。v158 的 step 102 对全部 3,793 条各取 31 个独立 sigmoid 中最高的一项。正确 338、预测为空 0；按 31 个类别等权平均 F1 得 2.7831%。最高分均值 46.8045% 既不是准确率也不是通过率。另按 0.5 选集合时才有 2,056 条为空。两次检查点的预测 SHA 不同，不得按目录版本号视为同一次刷分。step 102 的 seal、曲线、评分配置与收据中未见额外的按类校准器。

在仓库根目录、同一台控制机可用如下只读命令刷新原始 Top-1（终端输出）；不占用 NPU。脚本检查输入哈希、人口、音频身份，并与原审核的准确率、macro-F1、空集合计数核对。若要复现**另一个可重排候选的校准器**，还需要其参数、拟合数据划分、适用 checkpoint 和 31 类顺序；复用同份预测，另出收据，不能覆盖原始报告。

```bash
ops=/srv/multimodal-jev-xingchao/operations
.venv/bin/python tools/reports/summarize_audited_top1.py \
  --predictions "$ops/heldout-error-attribution-0p6b-step102/fsc102.json" \
  --audit "$ops/stage2-heldout-host181-v0.0.158/fsc/run/audit-full-102-collected.json" \
  --seal-root "$ops/native-evaluation-seal-v0.0.129"
```

**源域 TRAIN 校准能解释多少。** 有完整同策略 TRAIN 分数的是 Stage1：40,513 条；其中严格排名错误 6,108 条，另 7,317 条排名正确但固定 0.5 集合错误。本次追加了事后敏感性分析：只用源 TRAIN 的 gold 分和最强负分，在 0.01 到 0.99 的等距概率刻度中选“只选中 gold”条数最多的全局阈值，并列取最小。这里的刻度是固定概率网格，不是按测试分布取分位数。阈值为 0.45。这是本次诊断，不是当初发布的 harness 参数。只有源 TRAIN 分支可拟合；目标 TEST 不参与。TRAIN 也是 Stage1 的学习集，其拟合分数不能当独立验证成绩。

| 只选中唯一正确项（条数），0.50 → 0.45 | Stage1 | Stage2：仅迁移旧阈值作诊断 |
| --- | ---: | ---: |
| SLURP clean TRAIN | 27,088 → 27,460 | 无该终点完整 TRAIN 评分收据 |
| SLURP clean TEST | 8,367 → 8,482 | 9,548 → 9,521 |
| FSC native31 TEST | 3 → 17 | 42 → 52 |

全局阈值改变候选集合和拒答率，不改变强制 Top‑1。若要评价 Stage2 校准本身，需要同一 step 102 的源 TRAIN 输出，优先使用预留的说话者/话语分组校准折，固定目标和阈值后外推；不能拿 Stage1 reference 或随训练变化的 rollout 冒充同一策略。任何使用 FSC 带标训练样本的纠偏必须另立“目标域有监督适配”协议，与未见目标标签的结果分列。

| 错在哪（step 102） | 从证据能解释到哪里 | 下一步如何区分或改善 |
| --- | --- | --- |
| SLURP：2,316 条排错。512 条错误首选与 gold 的差小于 0.1，1,804 条至少相差 0.1；1,007 条错选概率 ≥0.9。另有 444 条排对但集合为空、683 条排对但多选。 | 排错、单边漏选、负类过线是不同机制。general_quirky → qa_factoid 53 条、反向 38 条；play_audiobook → play_music 40 条；calendar_query → calendar_set 32 条。只能确认标签混淆，不能据此归因于听错或标错。 | 在可信 TRAIN 独立重现混淆，再决定 hardest-negative 排序损失或双侧概率约束。保留回放，分开记录纠正和新增错误。留出类别样本少时同时给分子/分母。 |
| SLURP 的少数类与宽泛类 | 开灯仅 2/20 正确，music_settings 5/21。TRAIN 的 music_settings 155 条已经全部进入 Stage2，其中 152 条在 Stage1 被分到错误桶；general_quirky 的 1,550 条已有 1,431 条入选。 | 优先核查 TRAIN 上类定义、声音证据、负类配对及有效多样性；只增加重复曝光未必有帮助。遵守逐标签准入，不从 TEST 反向造奖励。 |
| FSC：3,455 条排错，2,914/3,793 首选“调高音量”，真值仅 396 条；31 类仅 5 类有命中。 | “调低音量”→“调高音量”245 条；2,484/3,455 同时错动作和对象（含三项都错），仅 9 条只错地点。2,950 个错例的 gold 连前三都没进。少数候选吸走概率，丢失的往往是整条指令。 | 在源 TRAIN 验证定义释义、同一音频交换候选顺序/符号并按语义映回、替换音频但固定候选；分别检验位置偏好、模板绑定和音频敏感性。尚不能认定末位偏置或声学编码是根因。不得用 FSC TEST 选干预方案。 |
| FSC：step 102 最高分 ≥0.9 的 618 条，只有 1 条正确；另有 292 条首选正确却没有任何分数过 0.5。 | 两种错误共存：错误候选高置信、部分正确候选低置信。按最高分等宽 10 桶的 ECE：Stage1 29.69% → Stage2 42.66%；候选 Brier 0.04058 → 0.04185，NLL 0.18412 → 0.19425。排名略升，概率可信度反而下降。 | 用源域独立校准折冻结拒答规则，分开报告覆盖率和被接受样本的准确率；再报告跨域 ECE、候选分布、原始 Top‑1。统一阈值/正温度不能修正调高与调低的顺序。 |

**真实录音的分数例子。** SLURP 的 `audio-1490105767-headset.flac`：gold 为 news_query，0.00023；错误首选 qa_currency，0.99427。FSC 的 `e4ccb7d0-452c-11e9-b1e4-e5985dca719e.wav`：gold 为开灯（无指定房间），0.00018；首选调高音量，0.97324。这些是封存任务标签和概率，不是重听后的声学判断。

[本次独立重算收据](/srv/multimodal-jev-xingchao/operations/heldout-error-attribution-0p6b-step102/fsc-step102-rescore.json) · [完整配对统计和各概率桶](/srv/multimodal-jev-xingchao/operations/heldout-error-attribution-0p6b-step102/report.json) · [Stage1 历史重算](/srv/multimodal-jev-xingchao/operations/fsc-top1-refresh-host181-v0.0.159/top1-rescore.json)

完整分析由 `tools.reports.heldout_error_attribution` 生成；使用 `configs/portable/operations/heldout-error-attribution.json` 及操作目录下的 `local.json`。仅消费已有预测、封存标签与获审的源 TRAIN 视图；输出包含两阶段配对转移、集合错误分解、可靠性分桶和 FSC 原生三字段的错项组合。
