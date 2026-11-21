# 下一代语音 Jev：用一套训练方案学习可迁移的判决

**r64 八卡执行记录。** 181 使用[冻结的本轮配置](../configs/experiments/next-simple-jev-r64-dp8-v1.json)。固定镜像 CPU 90 项通过，全量 TRAIN 两个布局共享声学特征；[八卡前缀、恢复与尾批收据](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/pipeline)确认 512 条常规批和 65 条真实尾批。正式 Stage1 [前三步记录](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/observation/first-updates.json)含 24 份 rank 输出，loss 4.34492、4.33864、4.31171；只证明启动和更新。Stage2 r64 的双实例数值一致性与源域、迁移效果仍待验收。

**给合作团队的配置入口。** 复制[简单任务实验配置](../configs/experiments/simple-two-stage-recipe.json)，分别填写两阶段 rank/alpha、是否复用、总卡数、推理实例数、每卡有效 batch、microbatch、轮数、warmup 比例及评测间隔。例：`PYTHONPATH=src python -m tools.reports.compile_simple_experiment --recipe configs/experiments/simple-two-stage-recipe.json --output configs/experiments/simple-two-stage-generated-v3`。输出目录是新版本；相同内容可重复校验，已有不同内容拒绝覆盖。生成的 experiment.json 绑定训练配置、采样模板及奖励设计；本地宏提供主机、模型/数据实际路径、卡号与租约。Stage1 用全 TRAIN，Stage2 的正式人口等新 reference 采样后再绑定。变更 rank 或设备必须用新初始策略及对应验收；生成配置本身不派发训练。v2 同时输出 stage2-runtime.json：将其中 serving 绑定给 endpoint runtime，panel_config 绑定给 TRAIN parity inputs。面板取配置下限与完整 macro 的较大者。GRPO 的标准差下限在配方中；clip、group size、学习率从 objective 引用的 execution 配置读取；loss 权重从 objective.options 读取。[跨配置与交接回归](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/handoff-cpu-final.xml)覆盖不同 rank、设备数、动态终点及旧证据拒绝。

**研究问题。** 同一条音频换一套等义的候选规约，模型能否仍把概率给到正确语义；换一条决定性证据不同的音频，概率能否随之翻转？候选排序与每个候选的独立概率都要可用。交付是一个端到端的 0.6B 决策模型：音频和完整候选定义输入，共享 Transformer head 输出各候选的 sigmoid 概率。参数提取属于 harness 的其他模型。此文件是本轮唯一执行 proposal；先前的布局分项、rank 扩容、局部词损失、few-shot 和联合动作设计仅作研究记录，不组成训练矩阵。

| 问题类别 | 数据能证明什么 | 本轮统一处理 |
|---|---|---|
| 证据不足或需要外部状态 | 原生互斥意图只有一个 gold，没有“信息不足”的独立真值。低概率、错序和标注缺失都不能推出 UNKNOWN。 | 原生完整可信的 TRAIN 标签才用于训练。缺乏判决依据的派生题留空，不合成为负类；不启用 UNKNOWN 头。 |
| 信息充分但混淆：声学线索与候选定义未正确对应 | 旧 Stage2 step102 在 53 对方向录音上，原布局有 49 对双边胜过反类、45 对双边完整首选；给边界说明后同为 45 对双边首选，一起逆序后降为 24 对。这是同一 TRAIN 探针的组合干预，无法归因到某层。 | Stage1 全量音频和可信原生标签学习区分；两轮让同一录音各见一次原生布局和一次**定义与 marker 同步置换**的布局，以 ID 绑定监督。翻转/保持录音作成对诊断，不把几对样本反复喂给模型。 |
| 候选间比较负荷高，或分数在换任务后失真 | 已审的 102 步 rollout 内，48,522 个错序比较中，12,059 个 16 次全错，无法得到该对的组内排序信号。旧 FSC TEST 的 macro-F1 从 0.0056 到 0.0278，而 NLL 从 0.18412 到 0.19425，Brier 从 0.04058 到 0.04185。前者说明部分组无梯度；后者只说明排名与概率质量可以分离。两者都不能证明参数不够。 | Stage2 只强化完整面板排序和存在性，以连续监督为全错组保留梯度。源域与迁移分别报告两项目标，模型容量固定 r64，不再同时寻找最优 rank。 |

**一次正式训练的定义。** Stage1 使用全量 40,513 条 clean SLURP TRAIN，181 八卡 DP8、每卡 64、microbatch 8、全局 512、warmup 8 步、2 轮 160 步。固定声学前端；语言 q/v LoRA r64/alpha128 与浅层决策 head 一起训练。沿用原配置的 InfoNCE（温度 1）、gold 对最强负类概率间隔 0.10 和 0.50 附近的双侧锚点；不加新的损失项。给两轮而非每条每次双前向：全程 81,026 次真实曝光。全任务复用两个固定前缀，调度可按布局聚集但不能漏样本、改 DP 分母或重复补曝光；padding 无梯度。归一化与报告始终用语义 ID。取固定末点，用原生部署布局重算全 TRAIN 原始概率、校验 LoRA/head 身份，再冻结 reference、难例与简单回放视图。测试或目标域不参与。

Stage2 在 181 上 DP6 + 两个各一卡的推理实例，全局 384、G=16。每个 G 固定同一音频和同一候选比较，保留全部正负对；对不同录音不共用 GRPO 均值和方差。保留排序/存在 GRPO（系数各 0.25），集合动作只记录、不贡献 loss；保留最强负类 softplus（0.10）、概率 margin（0.10）、全候选 Bernoulli NLL 之和（0.05）与相对 Stage1 的平均 Bernoulli KL（0.01）。一般错序对仍可能没有直接的排序 RL 梯度；NLL 可以更新其两个候选，不能声称解决了全部排序稀疏性。复用同配置 Stage1 adapter，重置优化器。全难例加按既有规则分层简单回放（目标 20%，各类下限沿用冻结配置），以**实际**视图大小 N 确定两轮共 2×ceil(N/384) 步。旧 reference、旧 19,379 条视图及旧 102 步都不可直接沿用。仅存适配器与 head 等最小复现材料。

**对抗式验证及失败判决。** 对照只用上一代冻结基线；不同扰动是对同一策略的检查，不各训练一份模型。

1. **反驳“几个例子足以训练通用能力”。** 对全量 clean TRAIN 执行监督准入与录音去重，按来源、说话者、句式、录音隔离已有诊断面板。可用局部编辑对在前次扩展规则下仅 6 对翻转、15 对保持，不能作为训练分布。用全量原生目标；局部对只描述诊断结果，同时给出样本与句架数。
2. **反驳“改了布局就学会语义”。** 先检查完整录音曝光、输入不依赖 gold、gold 与 marker 的 ID 双射、原生输入逐字兼容、缓存键分离。再用同音频等义换序以及证据不同的双边录音考核原生正确率、双边都正确、按 ID 对齐的概率偏移。常数预测不能靠低偏移过关。有限 NPU 验证需覆盖真实 processor、actor、推理与重载后的同策略数值；不得用 CPU 结果代替。
3. **反驳“排名升了概率就可靠”。** 检查全组动作相同、共同平移所有 logit、把相同答案置于不同位置、DP 尾批 padding 等反例。最终以同一来源、相同人口源域 selection 的 Top‑1 macro‑F1 为主要指标，准确率、NLL、Brier 与正负类分项共同解释；按说话者/源组配对 bootstrap 给差值 95% 区间。冻结前约定：macro‑F1 差值区间下界大于 0，NLL/Brier 点估计均不变差，才称源域主线成功；不满足如实报失败或不确定。过程指标不能替代正式端点。
4. **反驳“源域有效即可迁移”。** 模型与 TRAIN 校准阈值冻结后，仅一次报告 SLURP clean TEST、FSC native31、MINDS native14；同一模型、不用目标域调阈值或模板。每个目标域分别给分母、macro‑F1、NLL/Brier、预测类覆盖及与冻结同域基线的差值。有排序收益但概率退化，不能称为整体迁移收益；某域缺同口径基线或区间不确定，单列未证实。无跨任务独立监督，不声称跨任务成功。历史 TEST 数字用于说明风险，不参与设定权重。

资源与判决按“准备/CPU → 有限 NPU → 一次完整训练 → 冻结评测”推进。仅技术故障修工程并重新封包；效果未过保留负结果，不临时改目标、换阈值或加实验臂。把每轮实验证据和证明边界记在下方。

**采样与评价口径复核（2026-11-20）。** 本轮 r64 的冻结视图仍用既有规则：非严格首选、固定 `0.5` 阈值失败、概率间隔或双侧锚点未达标，任一满足就进同一个训练池。[TRAIN 发布收据](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/view-prepare.json)中的 10,427 条是这个规则筛出的人口，不能当作已确认的语义难例。固定脚本复算：3,918 条 gold 并列或未排第一，4,113 条首选正确但没有候选过线，1,206 条首选正确但多候选过线，另外 1,190 条未通过间隔或锚点。即使音频和候选次序完全不变，所有 logit 加上同一个常数也可能改变池成员。反过来，原生模板下高置信正确的录音也可能在等义换序后失效。下列是下一版评价与采样设计边界，本轮冻结配置保持原口径。

| 要分清的现象 | 如何验证和使用 |
| --- | --- |
| 证据充分，但混淆语义或绑定到错误的候选 | 可信 clean TRAIN 中按完整候选排序、同一对类的双向混淆找候选，不设固定 Top‑K。固定规约，用决定性音频线索改变的翻转对、等义保持对验证两边都正确；固定同条音频，比较原定义、补充类边界的规约及等义换序，按语义 ID 对齐。混淆统计可发现候选，只有对照能支持错因。若要改变训练模板，先验证各方向净收益和训推一致。 |
| 证据不足或需要请求时的外部事实 | 对同一判决 Q 比较条件完备、只去掉决定性证据、去掉等量无关证据；需要外部事实时固定音频，只改可信情境。没有独立真值的样本只留作诊断，不从低置信自动标 UNKNOWN 或当普通负例。 |
| 证据充分，但组合信息负荷高 | 用可信且可判的简单/组合对照固定候选含义，逐项改变所需证据，并记录哪项已能正确处理。需要同条件的能力或容量对照才可归因为参数不足。 |
| 排序正确，但给 harness 的分数不可靠 | 保留独立的概率质量诊断：报告正负候选分项 NLL、Brier；由源 TRAIN 校准运行阈值、冻结后再核对覆盖率和风险。固定 `0.5` 的空集合与多选只描述该工作点，不能替代对音频、规约和证据的验证。 |

下一版应复用现有原生 reference 做候选筛查，在独立来源组先补成对证据和类别覆盖，再冻结带证据标签的语义难例、单列的概率问题和简单回放。同时保留未分类样本的分母，不把未验证的错因写成全量监督。正式评测分别报告原生排序、成对翻转/保持和概率质量；`UNKNOWN` 有独立标注时才能评价。TEST 与迁移数据继续只作冻结评测。详见[原研究协议](04-jev-logic-capability-scaling.md)和[对比录音设计](../experiments/0p6b-contrast-groups-and-capacity.md)。

**新版难例集的已验证子集（2026-11-20）。** 用同一个 r64 Stage1 step 160 的原生规约和全量 clean TRAIN reference，排除旧机制探针及其同源项后，在 4,809 条可配对的录音中发现 96 条同词不同说话者的一错一对案例（94 条错序、2 条并列），另按 20% 配入 24 条两边都对的简单回放，合计 120 条、48 类。[独立复算](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/paired-train-evidence-v1/independent-audit.json)与训练入口的 CPU 读取一致；冻结 [manifest](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/paired-train-evidence-v1/view/MANIFEST.json)。这是音频变化下的保持性失败，不能进一步归因为漏听关键词；严格的同说话者局部翻转及保持对未增加训练成员。全 TRAIN 的其他 3,822 条错序或并列保留为未确认候选。没有当前终点的等义换序配对结果，也无独立的证据不足或组合负荷标注；这些类别不填造。按现有 DP6 配置只有两次更新，将其视为机制子集而非正式大规模训练替代品。操作见[数据发布入口](../docs/engineering/stage2-dataset-release-v2.md)。

**全 TRAIN 分层采样。** r64 step 160 的 40,513 条 clean TRAIN 进入本轮采样。采样视图不读取历史探针。按录音去重后选出 6,248 条，覆盖 60 类、3,327 个来源组。其中 2,228 条排序失败有真实音频对照，1,690 条保留原生可信标签与已观察到的错序或并列。排序正确的人口中再各选 384 条 gold 低概率和负候选高概率的样本，以平均分项 NLL 排序，两个名单互斥。最后按类别铺开 1,562 条简单回放，占最终视图 25%。两轮、global batch 384，共 34 次更新、12,496 次真实曝光。[采样配置](../configs/portable/operations/stage2-full-train-mining-dp6-v1.json) · [分层计数与独立审核](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/full-train-stratified-v1/independent-audit.json)

本轮仍使用原生音频和原生 gold。错因未确认的录音使用原生排序监督，局部词变化、同词不同说话者和近远麦克风的重叠证据均可追溯。同音频的等义规约、信息可判性和组合负荷需要单独的对照证据，当前训练集没有这三类监督。旧诊断面板历史结果可追溯，新模型在该面板上的结果属于 TRAIN 诊断。正式源域和迁移仍用冻结评测集。

**历史 r4 验证记录与复现。**

| 轮次 | 已执行与证据 | 结论边界 |
|---|---|---|
| 数据覆盖 | [真实 TRAIN 局部对](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/stage1-cue-pairs-v2.json)；[单一方案的全量分配与编译面板](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/next-simple-jev-plan-v1.json)。第一轮 20,257 原生/20,256 换序，第二轮对调；逐 ID 验证，无额外曝光。 | 证明方案有人口承载；不能声称已学到局部词识别。 |
| 语义绑定与工程 | [7 项 ID、目标及缓存检查](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/next-simple-binding-cpu.xml)，故意用错位 gold 的反例确实产生更高 loss；[20 项采样、缺失和污染拒绝检查](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/next-simple-sampling-cpu.xml)。另有 9 项 Stage2 视图封包检查通过。 | 证明组件在 CPU 上行为正确，不能替代实际训练器接线和 NPU 数值验收。 |
| 目标函数 | [24 项梯度、锚点、尾批和集合奖励检查](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/next-simple-loss-cpu.xml)；[181 正式镜像 CPU 回放](/srv/multimodal-jev-xingchao/operations/stage2-no-set-credit-v0.0.170/cpu/report.json)。回放旧正式 TRAIN 首组的前 8 条真实曝光，原生 verl 结果显示减去的 logit 梯度等于单独集合项梯度，排序/存在/辅助项及 KL 数值不变。 | 只支持“准确移除了指定信用”；不代表 LoRA/head 参数梯度冲突已解决，不能外推效果。 |
| 冻结策略反证 | [step318 诊断包收据](/srv/multimodal-jev-xingchao/operations/next-simple-318-probe-v0.0.171/paired/receipt.json)：298 条相同源 TRAIN 音频、60 个语义 ID，两种布局。 | [独立收集的配对评分](/srv/multimodal-jev-xingchao/operations/next-simple-318-probe-v0.0.171/paired/hf-queued/paired-score.json)已完成；见下表。这是旧 Stage1 的 TRAIN 机制证据。 |

[统一配置](../configs/experiments/next-simple-jev-v1.json)绑定 Stage1、Stage2 及采样配置身份；全量布局通过 `tools.reports.prepare_stage1_layout_plan` 与 `configs/portable/operations/stage1-layout-plan.json` 生成。固定镜像回放使用 `configs/portable/operations/next-simple-real-credit-cpu.json`，现场路径由本地宏绑定。仅共享配置与结论，不入库模型、轨迹或数据。Stage1 的调度、预取、消费端监督、恢复和独立审核均已接入布局计划；有限、恢复、尾批和正式包从各自的源码目录读取同一份内容固定的计划。旧 endpoint 制作的 Stage2 预备包只能验证工程。

**独立反证：把范围从词对扩大到任务类别。** 旧 Stage1 step318，同一批 298 条 TRAIN 录音、同一检查点、同一组 60 个定义，只同步置换定义和读出。分母为录音，指标为 Top‑1 正确数；“修复/退化”均按同一条录音配对。这里没有增加日历说明，也没有向模型提供真实日期。

| TRAIN 分层 | 原生顺序 | 同步置换 | 修复 / 退化 |
|---|---:|---:|---:|
| 背景 | 81/96 | 63/96 | 2 / 20 |
| 方向 | 92/106 | 93/106 | 3 / 2 |
| 日历 | 87/96 | 21/96 | 0 / 66 |
| 合计 | 260/298 | 177/298 | 5 / 88 |

方向对的双边正确数在两种布局下均为 40/53。另一个历史终点 step102（[原始评分](/srv/multimodal-jev-xingchao/operations/jev-boundary-source-panel-v1/runtime-v2/paired-score.json)），固定补充说明、改为逆序后，背景、方向、日历分别退化 22、22、35 条，三组均无修复。两个终点的置换不同，不用二者差值判断训练增益。共同证据支持“候选位置改变会破坏一部分本可做对的决策”，不足以证明模型不会听某个词、缺少日历知识，或容量不足。训练覆盖全 TRAIN，不对日历另设奖励。两种训练布局也可能被背下来：增加一个**仅评测**的未见顺序，在采样前固定（种子取训练置换种子 +1、同一编译器），同时报告原生和未见顺序；若仅已见顺序改善，只称已见布局鲁棒，不称学到了普遍的候选绑定。与旧基线做同顺序配对，不增加训练臂。

**工程反证：防止把旧输入和新标签配在一起。** 使用 [缓存配置](../configs/portable/operations/stage1-layout-cache.json) 与 `tools.reports.prepare_stage1_layout_cache`，在逐文件校验的原生缓存上完整验证 40,513 条 clean TRAIN；训练只需要两个文本模板和原有共享声学张量。298 条有独立真实 processor 编码的录音，两种布局均逐项比对 input IDs、mask、决策位置、BF16 声学特征及声学 mask；检查通过，换序后的全量长度 2,818–3,115，不超过 4,096。[可加载视图与证据](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/VIEW.json) 约 107 KiB，无模型权重和逐样本声学副本。其余录音核验的是已有特征及模板，不声称每条都重新从波形编码。训练端按实际读入的模板检查，无法匹配就停止。审核端按本地冻结计划重建每个槽位的语义真值。

[本地 CPU 验证](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/consumer-tests.xml) 60 项通过，涵盖缓存、目标绑定、尾批、恢复、训练审核；固定镜像 CPU 验证使用 [独立配置](../configs/portable/operations/next-simple-stage1-cpu-tests.json)，其收据与 BF16 有限 NPU 更新的收据分别判定。全量缓存核对、旧模型 NPU 诊断都不能代替新方案的有限更新或正式收益。

181 执行交接：[Stage1 调度配置](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173-local.json) 绑定共享缓存视图、原始 step0 策略、四卡与本轮计划。首次固定镜像检查 88/89 通过，失败原因为两轮模拟调度缺少真实调度已有的字段 `real_queries`；已补正测试数据，[消费与独立审核复验](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/consumer-audit-cpu.xml) 13 项通过。继任包的调度按确切容器退出事件收集 CPU 结果，通过后执行有限 NPU 验收；收据分阶段写入[本轮运行目录](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173)。排队不计作完成。

评测入口：[未见布局配置](../configs/portable/operations/next-simple-unseen-probe.json)生成原生与 seed=318 的新顺序，拒绝与训练顺序重合；[已封存输入](/srv/multimodal-jev-xingchao/operations/next-simple-unseen-probe-v1/seal/MANIFEST.json)固定同一 298 条 TRAIN 音频及 60 个候选。[配对评分](../tools/reports/score_next_simple_pair.py)读取各自经过收集验证的预测，按原生语义 ID 对齐，解决等概率时读出位置影响胜负的问题。比较支持同策略不同布局、不同策略相同布局两个方向。按完整 60 类计算 macro‑F1，无分母的类计零。对现有旧 step318 原生 / 训练计划的同步置换预测做[实际回放](/srv/multimodal-jev-xingchao/operations/next-simple-unseen-probe-v1/step318-native-trained-order.json)：macro‑F1 0.53537 → 0.36133，差 −0.17404（91 位说话者、2,000 次配对 bootstrap，95% 区间 [−0.18589, −0.10381]）；候选平均 NLL 0.01968 → 0.05361，Brier 0.00489 → 0.01440。这是 TRAIN 面板上的布局效应，不能当作新模型收益；区间不包含更换训练种子的变异。

固定镜像继任候选已[独立收集通过 89/89](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/jev-simple-stage1-layout-cpu-173/report.json)。[181 四卡 BF16 有限训练审核](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173/finite-audit.json)确认两次更新各 256 条真实请求；LoRA 和 head 均发生更新，四个 rank 重建的 BF16 导出与 master 一致。此收据不含初始训练收益、跨布局泛化或 Stage2 准入。

[恢复审核](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173/replay-audit.json)通过（四 rank 的 batch、master、Adam、BF16 输出与连续执行一致）；[尾批审核](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173/tail-audit.json)通过（第一轮末批 65 条、第二轮首批 256 条）。181 四卡正式 Stage1 已产生[前三组 loss](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage1-173-observation/loss-summary.json)：4.31467、4.32259、4.31452，各 256 条，均为全 rank 的真实样本归约。前三步平均 gold–最强负类概率差为负，margin 达标率均为 0；这些 warmup 批次不能用作最终能力判断。

交接时，除终点步数、策略、数据和提交内容外，Stage2 的 source 与数据发布入口都要求本轮 design、plan、cache 三个独立预置的哈希；旧 318 步策略不能借用新协议。模型推理仍使用原生部署布局。通过 [44 项 CPU 交接回归](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/stage2-handoff-tests.xml)。[终点收集任务](/srv/multimodal-jev-xingchao/operations/next-simple-layout-cache-v1/source-finalization-173/config.json)按本轮训练进程退出事件触发；源域曲线主指标为完整候选表的 Top‑1 macro‑F1，旧阈值集合分数仅作诊断。只有终点通过审核、原生布局的全 TRAIN reference 完整后才可发布新视图，不能用有限窗口模拟正式交接。

r64 八卡配置复现与 Stage1 → Stage2 的合作方命令见[工程交接](../docs/engineering/simple-experiment-admission-v1.md#r64-八卡实验的配置与接续)。仓库内的接续模板加本轮本地宏，已逐字段重建本轮冻结的 Stage2 输入。此验证覆盖配置与来源，设备数值和训练效果按各自收据报告。

**本轮 0.6B 端点比较的预登记对照。** 旧 Stage2 固定末点 step102，策略 `bc9dac8f16574c8554c1b8b95bc009b4820a30afb9cf17d5eb98d3007883e1d2`。按下表锁定四个人口：先从原始概率重算 Top‑1、NLL 和 Brier，再与各自独立审核核对。[本轮冻结收据](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/evaluation-baseline/baseline-fixed.json) SHA‑256：`e32097307257f2648bed191b037e209f1e82d6a36ff3adb680f6a976810710d6`。这些数字只属于上一代，不是 r64 结果。

| 数据与分母 | Top‑1 macro‑F1 | NLL | Brier | 分组单位 |
|---|---:|---:|---:|---|
| SLURP clean selection，9,226 | 79.37% | 0.02246 | 0.00517 | 2,146 个源组 |
| SLURP clean TEST，12,991 | 78.25% | 0.02251 | 0.00528 | 2,947 个源组 |
| FSC native31 官方 TEST，3,793 | 2.78% | 0.19425 | 0.04185 | 10 位说话者 |
| MINDS native14 外域，8,168 | 62.01% | 0.23628 | 0.04775 | 14 个语言/地区组，上游 split 为 train |

[固定比较入口](../tools/reports/compare_simple_endpoints.py)和[可移植配置](../configs/portable/operations/simple-endpoint-comparison-v1.json)只消费已收集的原生概率与独立审核。每个模型在四个数据集上必须有同一训练终点。本轮只允许固定末点与上一代固定末点比较；另行保留整个 selection 曲线。绑定的旧入口为 [local-fixed.json](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/evaluation-baseline/local-fixed.json)。合作方将其机器绑定迁到本地宏，用 `ENDPOINT_MODE=compare`，在 `ENDPOINT_BASELINE_RECEIPT` 填上述收据的 path 与 sha256，`ENDPOINT_CANDIDATE` 和 `ENDPOINT_BASELINE` 均填 policy_id 及按数据集命名的 cells。每个 cell 绑定 plan、audit（path/sha256）、seal（root/sha256）和 predictions 路径；文件内容须与审核哈希一致。使用新的 `ENDPOINT_REPORT` 输出路径。

```bash
PYTHONPATH=.:src python3 -Bm tools.reports.compare_simple_endpoints \
  --config configs/portable/operations/simple-endpoint-comparison-v1.json --local "$EVAL_LOCAL"
```

源域成功判据按同一批录音分组配对重采样；FSC 与 MINDS 各自报告。MINDS 旧结果是独立 HF 完整评测，服务数值一致性不合格的尝试不作为对照。MINDS 的区间表达语言/地区组的重采样变异，不解释为说话者间变异；FSC 只有 10 位测试说话者。全组评分耗时不等于 NPU 推理或完整评测耗时。校准后的 harness 阈值另行绑定，只能从源 TRAIN 获得。
