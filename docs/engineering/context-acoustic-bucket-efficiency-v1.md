# 声学批处理与训练长度分桶效率实验

## 选定配置验收进展（2026-11-17 UTC）

三臂统一重审已成功，状态`complete_efficiency_evidence_reconstructed`，
完整100步及剔除前2步均以`acoustic_kv16`最快；相对service32实测吞吐提升
12.5783%，节省940.4027秒。汇总位于
`/srv/multimodal-jev-xingchao/operations/training-efficiency-recommendation-final-v1/report.json`，
SHA256 `4b0054ea3444af8a79de073d31a3728f8265b7aee115f299b770e8349eb4feae`。

声学KV16真实尾批65条及随后128条的独立审核通过，结果
`operations/acoustic-kv-tail-v1/audit.json`为`finite_dp2_tail_outputs_audited`。
两次更新仅保存LoRA/head及优化器范围，FP32 master到BF16导出精确。
这不是新进程恢复证据（`fresh_process_replay_verified=false`）。

原service32的FSC有限面板exact通过，但MINDS259条中10条出现差异：
最大prob误差0.02438146、logit误差0.21875，top1变化0、0.5阈值翻转0。
队列按失败停止，未执行SLURP TEST；不能通过放宽exact条件宣布验收。
原输出和SHA256已保留在`operations/minds-macro-parity-v1/failure-evidence`。
差异样本均为2500–2900 padded mel帧，对应25–29个100帧声学块；
CNN批形状可能相关，尚未证实根因。01:16UTC优先启动所选声学KV16的MINDS面板，
沿用已封包实现及独立HF参考，尚无结果。正式全量封包重审正在运行，未训练。

## SLURP 源域TEST parity exact 通过（2026-11-17 UTC）

`acoustic_kv16` 的 **SLURP 源域** `SLURP_clean_TEST60` 有限面板隔离 parity 已通过：
面板 269 query × 60 decision，独立 HF 参考 vs 双 TP1 macro 输出逐条一致。

| 指标 | 值 |
| --- | ---: |
| query / decision | 269 / 60 |
| `probability_max_abs` | **0.0** |
| `logit_max_abs` | **0.0** |
| `threshold_flips` | 0 |
| `top1_changes` | 0 |
| `exact` | true |

- 结果状态 `target_width_macro_parity_outputs_audited`，收集器 `audit_outputs` 在
  `exact=false` 时直接拒绝，故 exit0 即等价于逐条一致。
- 操作入口：`tools/reports/context_transfer_parity_operations.py`
  （`--local .local/config/slurp-test-acoustic-kv-parity-v1-run.json`，推理卡 6、7），
  日志 `.local/decision-objective-research/slurp-acoustic-parity-v1-{run,collect}.log`。
- 证据边界：`full_population_parity=false`、`target_scored=false`——有限面板、只验 parity，
  不是任务分或迁移结论。
- **意义**：效率基线所在的源域（SLURP）上，批化声学路径与 HF 参考**逐条数值一致**；
  此前 MINDS 的 10/259 小差异属**迁移域**，不等于源域效率基线不可用。MINDS 差异仍需在
  迁移评测阶段定位，但不阻塞把 `acoustic_kv16`（74.76 s/step）作为 DP2 效率基线。

## KV16完整100步通过（2026-11-17 UTC）

`kv-training-study-v1/terminal.json` 已记录 exit0、`kv_training100_compared`，
驱动1974358已退出，不能再报告为运行中。客户端00:22:46UTC退出0、nonOOM；
完整收集及独立审核通过。两臂实际样本成员/顺序和reference misses一致，
均100步、12800次曝光；保持声学实现，仅将service KV从1GiB改为16GiB。

| 指标 | 声学+KV1GiB | 声学+KV16GiB |
| --- | ---: | ---: |
| 100步更新耗时（秒） | 7764.1680 | 7476.4063 |
| 平均每步（秒） | 77.6417 | 74.7641 |
| 样本吞吐（条/秒） | 1.64860 | 1.71205 |
| 平均rollout（秒） | 28.2916 | 24.4427 |
| rank0平均actor（秒） | 39.6153 | 40.4011 |
| rank0平均准备（秒） | 33.2473 | 29.4462 |
| actor峰值allocated（bytes） | 40302224896 | 40302224896 |

吞吐再提升3.8489%，去前两步后提升3.8488%，节省287.7617秒。
两臂均为`finite_dp2_efficiency_audited`；`fresh_process_replay_verified=false`。
单次顺序运行，更新计时不含启动/收集/评测，不能声称任务或迁移收益；
actor峰值一致不表示推理显存一致。训练总耗时应直接比较，不能叠乘推理探针提速。

报告：`/srv/multimodal-jev-xingchao/operations/kv-training-study-v1/comparison.json`，
SHA256 `e119f0e89d2b308de2021cc8561f9a7b5c24ff797b11a728a71d6f36cd9ae4b3`。
已启动固定汇总脚本独立重审三臂，使用本地宏
`.local/config/training-efficiency-recommendation-final-v1.json`，输出目标为
`/srv/multimodal-jev-xingchao/operations/training-efficiency-recommendation-final-v1/report.json`。
该报告生成前不视为汇总完成。原service32尾批/parity队列继续依赖原有驱动，
不能代替所选声学runtime的尾批、目标域parity及恢复验收；正式训练尚未派发。

## 声学批处理完整100步通过（2026-11-16 22:04 UTC）

`acoustic-training-study-v1/terminal.json` 已记录 exit0、
`acoustic_training100_compared`。客户端于21:49:16UTC退出0、nonOOM；结果收集后
完成独立审核。基线及候选均为 `finite_dp2_efficiency_audited`，实际100步样本成员、
顺序和逐步reference misses一致，各12,800次曝光/12,800次misses。

| 指标 | 原service32 | 声学批处理 |
| --- | ---: | ---: |
| 100步更新总耗时（秒） | 8416.8090 | 7764.1680 |
| 平均每步（秒） | 84.1681 | 77.6417 |
| 样本吞吐（条/秒） | 1.52077 | 1.64860 |
| 平均rollout（秒） | 33.4117 | 28.2916 |
| rank0平均actor（秒） | 40.5971 | 39.6153 |
| rank0平均准备（秒） | 38.4757 | 33.2473 |
| rank0导出/保存/同步残差（秒） | 5.0945 | 4.7781 |
| actor峰值allocated（bytes） | 40302224896 | 40302224896 |

100步吞吐提升 **8.4058%**，总更新耗时降低 **7.7540%**，节省652.6410秒。
剔除前两步后吞吐提升8.5121%。主要观测差异来自rollout；actor也有小幅波动，
单次顺序对照不能将每个计时分项变化都确定归因于声学改动。启动、结束后的收集审核
及正式评测耗时不计入更新计时，actor显存不代表推理worker显存。

两臂保持DP2/global128/micro8、双service32、KV1GiB、无新分桶、同初始策略与
算法奖励配置，仅引入已通过固定策略数值一致性验证的声学实现。这是完整训练更新
效率证据，不是任务准确率或跨数据集迁移收益，也不证明两条训练轨迹逐位相同。
候选审核的 `fresh_process_replay_verified=false`，不能据此声称新进程恢复已验收。
相比此前分桶单次+1.05%且padding未变，声学批处理应优先作为新runtime候选；
最终容量选择继续等待KV16的独立100步结果，不混合固定策略推理与训练提速比例。

报告：`/srv/multimodal-jev-xingchao/operations/acoustic-training-study-v1/comparison.json`，
SHA256 `46f1a949c9274c504e4a8bf8fdf70cdba6d7d45eafe078b69f19a125d0225543`。
基线manifest `6d6c8a6eccae8e27625f0aea6bf5bf73c9a7204fdbd75e3904546f7b4632e2e9`，
候选manifest `c5f1c2c634827a1f49b1a7bc52d5bb1bcc270d35d376e9f2e8e3f8a46b0e118d`。

22:04核验KV后继1974358准确存活，已生成 `kv-admission.json`，正在执行前置
审核，尚无KV训练attempt。KV监听及原tail/parity队列存活；未重复派发或提前
晋升正式训练。目标域一致性、尾批/恢复、正式学习及跨域评测仍待完成。

## 完整效率证据接入正式封包（2026-11-16 21:22 UTC）

新增 `tools/reports/prepare_context_acoustic_formal.py`，复用原正式封包器，
不修改运行中的冻结依赖。入口先调用完整训练效率汇总器，独立重审 service32、
acoustic、acoustic_kv16 三个100步运行，再按配置中的 `selected_arm` 和
`selection_reason` 选择父包。描述性最快排名不自动成为正式运行配置。
所有候选保留在路径重叠检查中，选择后重新核对配置hash和审核manifest；
声学两臂使用声学正式验证器，原service32使用原验证器。

Portable入口为 `configs/portable/operations/prepare-context-acoustic-formal.json`，
整个配置读取本地宏 `ACOUSTIC_FORMAL_PREPARATION`。宏需要
`recommendation_config`、`recommendation_local`、`selected_arm`、`selection_reason`，
以及共享封包器的 `project`、`output`、`remote_root`、`operations_root`、`tag`。
命令：`python -m tools.reports.prepare_context_acoustic_formal --config
configs/portable/operations/prepare-context-acoustic-formal.json --local <本地宏文件>`。

新入口12项及共享封包/声学正式操作回归共36项测试通过（2.67秒）。编排测试使用
合成证据，不代表真实完成；对实际未完成的两组study调用选择入口，正确拒绝，
没有创建正式包或派发。完整效率、所选runtime尾批/parity、稀疏保存恢复及正式
launch验收仍须实际完成。此前“正式选择器仍未接声学证据”的工程缺口由此入口
补齐，但尚未获得真实完整结果来封包。

21:20固定监听：声学双rank76/100、76提交、末步最大77.33秒、策略身份一致。
21:22四个执行/后继驱动准确存活，配置pins均无变化；未新增远端查询。

## 声学/KV目标域一致性包已齐备（2026-11-16 20:35 UTC）

新增`context_acoustic_transfer_parity.py`复用原HF-first、目标宽度面板、双service和完整
收集审核，保持HF参考路径，仅替换声学服务入口。显式绑定声学/KV效率父包，验证父
资产和源码逐字节继承、额外文件集合、目标服务完整配置差分；候选宽度由原验证器
绑定目标定义，KV容量保持父实验设置。无需读取目标任务标签或计算任务成绩。
18项测试通过（2.25秒），包括四角色命令/设备继承与重新封包后的篡改拒绝；模型及
输入准入在测试中用替身，实际三个prepare/validate另行成功。

以下三个候选均由同一KV16初始策略父包制作，未排队或派发NPU：

| 本地operations子目录 | manifest SHA256 |
| --- | --- |
| fsc-acoustic-kv-parity-v1-prepared | b7b8927b4544d8043aff22393ae5fd3e69b6f190d67f42b0eb7ede54ddd92af0 |
| minds-acoustic-kv-parity-v1-prepared | b0a8b36e9833c90a7e4f7a0eb47f9aa2f0f0554a828d2ebc5587f86e25d9b858 |
| slurp-test-acoustic-kv-parity-v1-prepared | a1bd1fe86da3669273866e89006b78ec84844e6f8c6f6c725c31d1f41d741547 |

根目录`/srv/multimodal-jev-xingchao/operations`。portable入口
`configs/portable/operations/context-acoustic-transfer-parity.json`；路径宏在
`.local/config/{fsc,minds,slurp-test}-acoustic-kv-parity-v1-prepare.json`。
三个prepared的inference许可均false。它们仍是有限面板和共同初始策略的待运行验证，
不是全量迁移分数，也不替代最终选定checkpoint的对应验收。现有旧service32队列不变。

20:35固定监听采集声学双rank42/100、42提交，最近一步76.83秒，policy一致；
六现有进程准确live、pins全部保持。本轮没有额外远端查询。

## 完整训练效率汇总入口（2026-11-16 20:28 UTC）

20:57补充：新增5项完整报告路径测试，验证三种运行分别使用对应validator/command、
比较器收到重新审核的证据、原报告篡改/成功终态缺报告/错误study绑定/原始审核失败
均拒绝。模块共18tests1.99s通过（3c0637），原始大型审核和比较器在这些编排测试中
使用替身，不能宣称真实100步结果已完成。运行代码及冻结pins未改；六进程准确live。

新增`context_training_efficiency_recommendation.py`读取service32、声学及声学KV16三个
冻结配置和两个study配置。两组完整100步成功后，独立重审三个运行、重建两份比较并
与原报告逐项相等；验证共享声学运行身份，报告全100步与去前2步的耗时排名、直接
计算训练吞吐比。它不将固定策略推理收益相乘，不自动晋升或派发正式训练；单次顺序
运行的重复性、对应新runtime尾批/parity及任务迁移仍是独立要求。

13项测试通过（2.00秒），覆盖不完整/不匹配/非法计时拒绝、排名分歧、并列偏向简单
配置及未完成时不触发全量审核。真实运行输出
`/srv/multimodal-jev-xingchao/operations/training-efficiency-recommendation-v1/pending.json`，
状态`awaiting_complete_evidence`，声学与KV均尚无完整结果；未产生候选排名。
portable为`configs/portable/operations/context-training-efficiency-recommendation.json`，
本地宏`.local/config/training-efficiency-recommendation-v1.json`。该次输出仅为当前快照，
最终报告需使用新的输出路径执行，不能覆盖原快照。当前无新增自动驱动。
20:28六个既有进程准确存活、pins全保持，无新增远端查询。

## 声学/KV配置尾批候选已封包（2026-11-16 20:14 UTC）

新增`context_acoustic_tail_operations.py`复用原尾批准备、双rank安全CLI及完整
生命周期/输出审核。它显式接受acoustic或kv16父包，仅增加尾批诊断入口和全日程
校验器；服务继续使用声学模块，完整保留KV、输入、算法和模型设置。校验器检查
父manifest、同设备租约、文件集合/逐字节继承、完整训练及双service配置差分、
新增源码和运行源码身份。prepare默认不释放诊断或正式训练许可。

真实KV16父包准备成功并通过validate，输出
`/srv/multimodal-jev-xingchao/operations/acoustic-kv-tail-v1-prepared`，manifest
`7d0cbe1f3ae773a800d28eafc75b391a489300d9640ab3d6f061456d8913cd8c`。
本地宏`.local/config/acoustic-kv-tail-v1-prepare.json`，portable
`configs/portable/operations/context-acoustic-tail.json`。真实日程indices316/317，
65+128条真实样本、两次fresh诊断更新；没有完整正式历史或恢复含义。
目前仅封包，**未排队、未NPU派发**，不改变既有原service32尾批/目标parity队列。

入口与既有CLI12项测试通过（2.12秒）；新增6项完整差分/篡改检查后，此模块12项
测试通过（1.99秒），合计18项不同测试覆盖。父校验/远端在测试中使用替身，实际
prepare另外通过b494b6。20:14现有六进程准确存活、pins全不变，无新远端观察。
声学100及KV100完成后，再按选定runtime开展对应尾批/目标域一致性验证。

## 声学正式训练入口适配（2026-11-16 20:09 UTC）

原`context_macro_formal_operations.command`沿用旧service入口，不能直接把声学封包
当正式声学运行。新增`context_acoustic_formal_operations.py`复用正式actor/preflight
命令与完整输出审核，仅将双service模块换成已测声学入口，并核验四个声学源码文件。
该模块只提供validate/collect/audit，无派发；旧正式入口及全部运行依赖保持不变。
相关17项测试通过（2.19秒），覆盖四角色命令差分、原绑定不受影响及源码变化拒绝。

在两个真实已封存声学/KV候选包上执行正式fresh/resume源码生成与AST解析，均成功且
生成源码hash分别相同：fresh
`983548c80df7cc81172346c98fd86da7cb05df4e829fa0861e0fb4d30f01010d`，resume
`c65a758767ecf2381d065081e118730965705e33f949244f971d5c119cbf953a`。
两个候选的完整真实schedule均通过现有全量校验：40513条×2轮=81026次真实曝光、
634步、epoch尾批65条；release
`2418e2a974b529fe95a4fd41c0c99673e10dc0b698e91e1136eef127a7dd72f2`，schedule
`d9cd5832aa160df46a4c10a2051ff12c352d2a2609e2cffd6901267b28a814f6`。
这只证明CPU生成/日程兼容，不证明NPU执行或恢复。正式封包选择器仍仅支持原micro/
service对照，声学/KV完整100证据选择、匹配新runtime的尾批/迁移parity仍待接入。

20:05固定监听记录声学双rank18/100、最近一步77.60秒，两端policy一致；20:09六个
现有驱动/监听准确存活、全部pins一致。本轮无新增远端查询或正式训练派发。

## 三个目标域一致性验证已接续（2026-11-16 19:56 UTC）

新增`run_context_transfer_parity_sequence_v2.py`仅适配修复后tail生命周期与成功终态，
复用已封存的三个目标域封包、HF-first/双service生命周期及完整输出审核。
队列准确等待tail v2成功，独立重审尾批后，顺序执行FSC_native31、MINDS_native14、
SLURP_clean_TEST60有限面板。三者共用来源策略及runtime，禁止重复attempt；失败
停止，不自动重试、不评分目标任务、不启动正式训练。它验证原service32目标宽度
一致性，不能作为声学/KV新runtime或最终训练checkpoint的完整迁移验收。

真实三包及队列`--validate-only`通过；v2适配与既有序列15项测试通过（2.24秒）。
本地宏`.local/config/transfer-parity-sequence-v2.json`，复用portable
`configs/portable/operations/context-transfer-parity-sequence.json`。
输出`/srv/multimodal-jev-xingchao/operations/transfer-parity-sequence-v2`；驱动身份
记录在相邻`transfer-parity-sequence-v2-driver/driver.json`：PID1976169，start_ticks
343562318，cmdline SHA256
`be0922ce4c6124593008218553effe3bfd2b40067bade44cdec8b9943e012acb`，143pins。
19:56准确/proc存活，pidfd等待tail1975303；尚无目标NPU结果。原声学、KV、tail及
监听共六进程均准确存活且pins一致。本轮无新增普通远端查询。

## 全量训练尾批诊断已排队（2026-11-16 19:53 UTC）

固定监听器19:50记录声学only训练双rank完成7/100步，最近一步77.44秒，
7个checkpoint提交；这不是完整效率结果。本轮只读取本地监听产物，无额外远端查询。

全量SLURP有40513条训练样本，global128的epoch尾批含65条真实样本。此前tail v1
在双rank共享父目录创建时失败；v2封包已修复CLI并保留原训练逻辑，但尚无NPU验证。
现新增固定后继`tools/reports/run_context_tail_admission_v2.py`，等待KV100成功后，
运行已封存v2的尾批及下一批两步诊断并完整收集/审核。只释放诊断许可，正式训练
许可仍为false；失败保留现场、无自动重试。比较队列和诊断共用同一设备租约。

此诊断使用原service32实现，**不证明声学/KV配置的尾批通过**，也不是执行至第317步
的正式历史或恢复验收。新基线的尾批、目标域训推一致性及正式学习/迁移评测仍须完成。

portable：`configs/portable/operations/context-tail-admission-v2.json`；本地宏：
`.local/config/tail-admission-v2-queue.json`。真实`--validate-only`通过，后继和CLI
18项测试通过（2.16秒，远端使用替身）。旧封包manifest仍为
`e5b4499aea3bc48857e3418d83ec1b987014946e000c8dff2a7006889d6b94f1`。
输出`/srv/multimodal-jev-xingchao/operations/tail-admission-v2-queue`，PID1975303、
start_ticks343543946、cmdline SHA256
`60501aebae03e47b8a8453e1e581b465b794f7e95be30602fa7fb7b96feb4533`，140pins。
19:53按/proc身份核验存活，准确pidfd等待KV驱动1974358，尚无tail attempt；原四进程
仍准确存活且全部依赖未变。没有增加远端轮询或模型驱动分析频率。

## KV训练后继已排队（2026-11-16 19:49 UTC）

固定驱动`tools/reports/run_context_kv_training_study.py`已启动，输出目录
`/srv/multimodal-jev-xingchao/operations/kv-training-study-v1`。它通过准确PID句柄
等待声学only100驱动成功退出，不做远端轮询；当前尚未派发KV训练。
流程为：重验冻结依赖及实际测量策略绑定 → 独立重审KV推理数值证据 → 完整重审
声学训练父基线 → KV16GiB训练100 → 完整归档审核 → 同实际样本/顺序/reference
misses效率比较。训练与推理源码、策略和服务参数均绑定，除物理卡映射及本地路径外，
推理面板策略需与训练初始策略一致。任何失败保留现场、停止后继，不自动重试或晋升。

训练生命周期和比较器复用现有实现，输出schema为`jev-kv-training-efficiency-v1`。
portable入口为`configs/portable/operations/context-kv-training-study.json`，本机宏为
`.local/config/kv-training-study-v1.json`。已真实执行`--validate-only`通过；
KV配置、后继顺序、失败停止、完整100步比较及观察器绑定合计30项本地测试通过
（2.43秒）。远端训练在单元测试中使用替身，测试不代表训练成功。

后继PID1974358，start_ticks343518000，cmdline SHA256
`2a0398f71f5ca11c2dfcf2084eaa765e3f94e812de466aae6316cee6d5a86caf`，132pins。
共享900秒监听绑定`watch_context_kv_training.py`已启动，PID1974405，start_ticks
343520205，cmdline SHA256
`30c24cd452bca892a38cdb8a01687c06cec196be0ec25df27e67628e7f796a55`，137pins；
宏`.local/config/kv-training-v1-watcher.json`。尚无attempt时仅本地记录未启动。
19:49四个声学/KV驱动与监听均按/proc准确身份核验存活，冻结依赖全部一致。
本轮没有新增手动远端观察；声学100仍无终态。最终学习收益和跨数据集迁移待验证。

## 当前训练状态与KV训练候选（2026-11-16 19:42 UTC）

声学only100后继已完成父基线审核并派发两个推理服务和DP2训练客户端；本地
`services-ready.json`及`client-launch.json`已存在，客户端容器为
`85e75f5c3412a4621f3788060452a6c777e23ed92282a344fc777b0d6aafa6f9`。
尚无完整训练结果，不将已派发等同于100步成功。驱动1943462与900秒监听1943531
按PID/start_ticks/cmdline核验仍存活，125/132个依赖hash全部匹配；本轮没有新增
远端状态查询，最新监听记录为尚未到期。

KV16GiB完整训练候选已通过真实prepare/validate，尚未派发或排队。固定入口
`tools/reports/context_kv_training_operations.py`复用现有声学训练生命周期；portable
配置为`configs/portable/operations/context-kv-training.json`，本机路径集中在
`.local/config/kv-training-v1-prepare.json`。从声学only训练封包克隆后，除运行目录
重定位及派生manifest外，仅修改两个服务的`policy.kv_cache_memory_bytes`为
17179869184；模型、源码、训练配置及输入身份保持一致。验证器检查文件集合、
逐文件hash、完整配置差分和父封包身份。

候选：`/srv/multimodal-jev-xingchao/operations/kv-training-v1-prepared`；manifest
`4bff0fcd10c19bc2cfbf04a79883973e3649f2c10deea8a7f531bc111198aa38`。
准备产物的`finite_training_dispatch_allowed=false`，不抢占现有实验。
KV与声学训练契约合计14项测试通过（系统Python，2.26秒）；真实prepare退出0。
测试包含拒绝非KV配置变化及禁止隐式派发，但不代表KV16GiB已完成NPU训练。
下一步先收齐声学only100的完整对照，再单独开展KV16GiB训练100验证。

## KV容量实测完成（2026-11-16 19:25 UTC）

KV-only对照容器于19:23:31 UTC退出0、nonOOM；后继驱动终态为
`kv_capacity_independently_compared`。两臂使用相同声学批处理、固定初始策略、相同
TRAIN128输入、service32及131072 token预算，仅KV容量由1GiB改为16GiB。
每臂独立进程，预热、两次计时和逆序四pass均强制重新编码音频。

| 指标 | KV1GiB | KV16GiB |
| --- | ---: | ---: |
| 两次计时均值（秒/128条） | 17.7500 | 14.5718 |
| 每pass实际language forward | 40×batch3 + 8×batch1 | 4×batch31 + 4×batch1 |
| 每pass encoder调用 | 48 | 8 |
| 每pass conv调用 | 48 | 19 |
| 每pass conv chunks | 433 | 433 |
| 计时pass最大allocated（GiB） | 3.1571 | 21.1406 |

吞吐提升21.8104%，耗时下降17.9052%。完整logits/prob最大差均为0，固定种子
Bernoulli采样完全一致。实际日志容量149,760 tokens，4096长度理论并发36.56；
提交仍为4×32，实际是31+1，不能声称已实现forward32，也尚未确定剩余拆分原因。
容量扩展使实际调度批量明显增大，支持原1GiB是批量调度瓶颈的判断。

范围仍为单固定策略、TRAIN128和两次计时；这不是完整训练吞吐、全量数值准入或
任务/迁移效果证明。不要将本次21.8%和前次声学19.4%直接相乘作为训练收益。
后续应在声学only100完成后，以同一声学实现对照KV容量的完整训练效率，核验
策略切换、实际调度、内存及训推一致性，再决定是否晋升默认配置。

报告：`/srv/multimodal-jev-xingchao/operations/kv-capacity-successor-v1/comparison.json`，
SHA256 `0e69dfc9c6a2fa979e93bcccff5c0c0ad2e02ec4b03694ce23e3cf3275f99c9c`。
声学训练后继1943462仍准确live，已生成`acoustic-admission.json`，进入父基线重新
审核；此时尚未启动训练。它保持原KV1GiB、无分桶，以独立测量声学收益。

## 完整分桶100步结果（2026-11-16 19:17 UTC）

`efficiency-study-v3` 已退出0，终态为 `acoustic_and_bucket_efficiency_compared`。
父基线与分桶候选均通过 `finite_dp2_efficiency_audited` 完整审核；比较器检查了
相同镜像、封包身份、实际100步样本成员及顺序、逐步reference misses。
两臂均100步、12,800次样本曝光、12,800次reference misses；候选仅改变训练分桶，
仍使用原声学编码、KV1GiB、DP2/global128/micro8和两个batch32推理服务。

| 指标 | 原service32基线 | 分桶候选 |
| --- | ---: | ---: |
| 100步累计耗时（秒，含每步导出/同步） | 8416.8090 | 8329.1332 |
| 平均每步（秒） | 84.1681 | 83.2913 |
| 样本/秒 | 1.52077 | 1.53677 |
| rank0 actor均值（秒） | 40.5971 | 40.3423 |
| rollout均值（秒） | 33.4117 | 33.0150 |
| actor allocated峰值（bytes） | 40,302,224,896 | 40,290,749,952 |

观测吞吐提升 **1.0526%**，排除前两步后为 **1.0961%**；100步节约87.6759秒。
actor耗时仅下降约0.63%，而未改动的rollout和导出/同步也有下降，单次配对运行
没有重复试验或置信区间，不能将全部差异确定归因于分桶。
比较器基于实际processor长度重建的padding总量未变：tokens为36,361,736、frames为
4,604,800；最慢rank线性成本代理仅下降0.1933%。当前global batch内重排空间有限。

决策：保留分桶候选与证据，不将这一微小收益作为默认基线晋升依据；优先完成已经
排队的KV容量对照和声学only100训练，集中验证更大的效率改动。上述结果不证明任务
效果或跨数据集迁移收益，正式训练的尾批和目标域一致性验收仍待完成。

完整报告：`/srv/multimodal-jev-xingchao/operations/efficiency-study-v3/bucket-comparison.json`，
SHA256 `f9d2b81f1155235ce896d259015d6af35091915853ff7fa744243f4e13abb129`。
候选最终policy为 `b3fb13c133c74991c0b15604040891f0dfa05f352808e933fd7c8d0e61ec4a3b`。
KV后继已生成独立released配置并成功启动容器
`d705f6614da1403cc5f6aea6f1e771a8df4a546af0efdc7af5eaac76e318bc6a`；
此处仅有launch/start-response证据，尚无容量收益结论。

以下为按时间保留的实施记录；当前状态以本节为准。

2026-11-16 16:39UTC：声学v4已完成实际NPU对照与独立审核；bucket-only100已启动。
服务batch32基线已完成100步，吞吐相对batch4提升2.6994%；这不是声学或分桶收益。

## KV容量对照已封包并排队（2026-11-16 16:49UTC）

新增固定脚本 `run_context_kv_probe.py`、`context_kv_probe_operations.py` 和
`run_context_kv_successor.py`，共用既有声学测量、容器生命周期及完整数值审核。
两臂都使用已通过实际数值对照的声学批处理；仅改变
`kv_cache_memory_bytes`：1073741824 → 17179869184，其他策略/输入/服务32与
131072 token预算保持一致。每臂独立模型进程，预热、两次计时及逆序四pass均强制
重新编码音频。报告保留实际language forward批量分布、提交批量、allocator峰值、
完整logits/prob及固定Bernoulli采样；不会因更快就自动准入训练。

机器路径集中在 `.local/config/kv-capacity-v1-prepare.json` 和
`.local/config/kv-capacity-successor-v1.json`，公共入口为 portable 配置。
实际封包 `/srv/multimodal-jev-xingchao/operations/kv-capacity-v1-prepared`，manifest
`6b11dab2b6b1ab22de674e03666271cabbb015e0a734dfea4a554b95889d7bb7`。
继承声学v4的源码和资产逐字节保持，prepared默认禁止派发。
实际prepare与validate通过（ba8948、46029f）。测量及已有契约33项测试通过
（a7f5e0，2.92秒）；新增队列和测量16项通过（188a7a，2.35秒，含前述测量重跑）。
队列测试用远端替身，不代表16GiB已通过NPU准入。

后继进程 PID1940314/start342442482/cmdhash
`59a344028916dd11a7002a12caa8a4fe0243b015c627ca8d3c73298d8b4702b4`，119pins。
16:49已核验准确 `/proc` 身份，`waiting.json` 通过pidfd等待当前efficiency-study-v3
PID1933847；不作远端轮询。只有完整bucket100队列成功并重新审核声学的完整数值
一致性后，才生成独立released operations，获取同一资源锁并检查实际空闲后运行。
失败保留原始证据、不重试、不自动晋升训练。当前没有16GiB容量实测或bucket100
收益结论。bucket客户端已于16:43启动，当前配置保持冻结，监听间隔仍900秒。


声学批处理接入训练服务的独立候选已准备（2026-11-16 17:01UTC附近）。
新增 `context_acoustic_service_candidate.py` 和 `context_acoustic_training_operations.py`，
复用原服务认证、LoRA/head切换、清理和原生verl actor，仅更换两推理服务的owner。
固定DP2/global128/micro8、原始100步顺序、原KV1GiB；不混入分桶或KV扩容。
封包 `/srv/multimodal-jev-xingchao/operations/acoustic-training-v1-prepared`，manifest
`c5f1c2c634827a1f49b1a7bc52d5bb1bcc270d35d376e9f2e8e3f8a46b0e118d`；配置宏 `.local/config/acoustic-training-v1-prepare.json`。
原训练源码、模型资产保持逐字节一致；client/preflight命令相同，仅每个service的
模块入口替换一个参数。22项服务绑定、设备隔离和声学测试通过（06b41d，2.47秒），
真实prepare通过d1ae5b。候选默认禁止派发，尚未接后继队列，不影响当前bucket100及
已排队KV实验；后续需完整100步审核，不能用固定策略推理收益替代训练吞吐证据。

## 声学训练100步后继（2026-11-16 17:09UTC）

已通过固定脚本接通：bucket100 → KV容量对照 → 声学only100训练。
新入口 `run_context_acoustic_training_study.py`，portable配置
`context-acoustic-training-study.json`，本地绑定集中在
`.local/config/acoustic-training-study-v1.json`。后台driver PID1943462，准确身份见
`/srv/multimodal-jev-xingchao/operations/acoustic-training-study-v1/driver.json`，125pins。
当前通过pidfd等待KV后继1940314；未远端启动声学训练，不改变现有实验。

前置成功后再次审核声学v4实际容器、transport及完整数值一致性，并重审原service32
完整100步基线，再释放已封包的声学only100。该训练仍KV1GiB、不采用KV16GiB，
因此独立测量声学因素。完成后完整收集并审核两rank更新、LoRA/head和激活轨迹，
校验实际100步数据成员、顺序和reference misses相同，再比较包含激活的总耗时及
去掉前两步的吞吐。任何步骤失败保留原始证据，不自动重试、晋升正式训练或声称迁移收益。

测试18项通过（3bcb1b，2.29秒），覆盖队列成功/失败停机、输入顺序/缓存条件/镜像/
不完整审核拒绝、吞吐计算以及现有服务/观察绑定；远端在队列测试中用替身。
真实125pins、训练封包和声学准入封包验证通过e98de9。
新增900秒固定监听 `watch_context_acoustic_training.py`（PID1943531，132pins），
准确绑定新driver；没有attempt时只本地记录未启动，训练开始后观察准确句柄。
17:09核验两个新进程准确live。原bucket监听17:08观察双rank17/100、17提交，
末步约84.26秒；不以单步时长推断完整效率收益。

## 声学v4实测结果

准确容器`96ff33ddd455c15628529a3ef0712cdcc1e20c36eff1a5c9f620b5c4a9329e4e`
exit0、nonOOM，父基线、完整输出与transport独立审核通过。报告
`/srv/multimodal-jev-xingchao/operations/efficiency-study-v3/acoustic-comparison.json`，
SHA256 `9f93c8409a51c0b266eef910adb2d3087d7be85a891cce741f532160e94d5b11`。

| 指标 | 原声学 | 批处理声学 |
| --- | ---: | ---: |
| 两次计时均值（128条） | 21.0835秒 | 17.6562秒 |
| 每轮conv调用 | 128 | 48 |
| 每轮encoder调用 | 128 | 48 |
| 每轮处理的conv chunks | 433 | 433 |
| 计时轮allocated峰值最大值 | 3,389,939,200 bytes | 3,389,939,200 bytes |

吞吐比1.19411，即+19.4112%；耗时减少3.4273秒，即-16.2557%。所有四pass
（预热、两次计时、逆序）实际声学调用均非零，完整logits/prob相对原预热输出
最大差均为0，固定种子Bernoulli G16采样完全一致。计时均值排除初始化/预热。
范围仅单个固定策略、TRAIN128、一对进程和两次计时，无置信区间；不证明全量训练
吞吐、梯度或最终任务指标收益，尚不自动准入训练。

真实调度限制也已观测：每轮4次encode提交，每次32条（token总量90241/90553/
90830/91592），但实际48次language forward为40次batch3和8次batch1；两臂完全
相同。配置`kv_cache_memory_bytes=1073741824`，实际日志KV容量9344 tokens，
4096长度并发2.28。batch32是提交上限，不能写成实际forward32。KV容量很可能
限制了调度批量，下一独立容量实验应验证扩大KV后的实际覆盖、显存及数值；本轮
不原位改动已经开始的bucket100。该事实不改变同容量声学配对比较的结论。

bucket-training-v1已生成attempt及preflight-launch，仍使用旧声学以隔离分桶因素。
efficiency-study-v3 PID1933847继续运行；watcher-v2固定900秒，下一本地到期约
16:50:25UTC。尚无bucket100结果，不提前宣称分桶提速。

### KV容量规划（16:43UTC）

固定脚本`plan_context_kv_capacity.py`和portable模板
`context-kv-capacity-plan.json`读取固定模型配置、已测请求长度与真实encode计划，
机器绑定在`.local/config/kv-capacity-plan-v1.json`。实际报告
`/srv/multimodal-jev-xingchao/operations/kv-capacity-plan-v1/report.json`已生成。
28层×8 KV heads×128 head_dim×K/V两份×BF16两字节=114688 bytes/token。
每请求按128-token块向上取整后，1GiB给出9344 tokens，与NPU日志准确一致。
真实32请求的最大KV需求10804527104 bytes（10.0625GiB）；32×4096的上界为14GiB。
下一容量候选16GiB预计给出149760 token容量，可覆盖这两个KV下界。

这不是显存准入：模型权重、声学缓存、分支和主干激活、workspace、allocator余量
都要另外实测。固定shape公式及无效模型/块/请求测试6项通过（ca0213，0.18秒），
实际规划执行af0ab1通过。当前仅规划、未封包或派发16GiB；保持正在执行的bucket100
配置。后续只改变KV容量，检查实际forward覆盖/数值/峰值显存/耗时后再做训练对照。

## 当前执行状态（替代下方原队列记录）

最新：声学v3测量已失败并保留证据，原因是预热后的音频编码缓存命中。两组子进程
都成功完成四pass，但计时/逆序pass中conv与encoder调用均为0，最终compare以
`actual acoustic telemetry required`拒绝，容器exit1/nonOOM。baseline计时约14.50秒、
acoustic约14.18秒属于热缓存路径，不能解释成声学编码收益；warmup含首次编译等
开销，也不作为正式声学对照。原efficiency-study-v2及watcher已终态退出。

修复测量脚本：每pass通过独立音频MM UUID命名空间强制重新编码，输入张量、
策略、engine请求身份、输出对齐逻辑保持不变，继续要求实际声学调用非零。
23项相关测试通过（d392e4，2.58秒），包含缓存模拟、张量身份保持、异常恢复和
声学边界。新声学v4封包manifest
`df10065af5635f135bf288d0ca15ec4ef6fd4faab9a32255add992eb815602ca`，实际prepare/
validate通过517354，输出`operations/acoustic-batch-v4-prepared`。

新固定队列`operations/efficiency-study-v3`（PID1933847，113pins），宏
`.local/config/efficiency-study-v3.json`，重新审核完整baseline后执行声学v4、原
bucket100；`bucket-training-v1-watcher-v2`监听900秒。旧终态和容器不修改，
没有放宽准入或宣称实际声学提速。下方v2队列记录为历史。

原tail-admission-v1在15:54UTC返回失败：两个torchrun进程在CLI中执行
`a.output.mkdir(parents=True,exist_ok=False)`，竞争同一个run目录，触发
`FileExistsError`，发生在训练开始前。实际client退出1、非OOM；双服务由固定
失败清理流程停止，退出137、非OOM。原tail/parity/efficiency三驱动均终态失败且
本地进程已退出，没有声学或分桶的失败结果。日志保留在
`/srv/multimodal-jev-xingchao/operations/tail-admission-v1/client.log`。

尾批用于后续全量训练准入，目标域parity用于迁移评测准入；两者不是源域完整batch
效率测量的必要前置。本次新增`run_context_efficiency_study.py`及portable模板
`context-efficiency-study.json`，直接绑定已成功的service-batch32-study-v3：检查
完整100步/12800曝光、实际成员和reference misses匹配，并重新执行完整独立审核，
再依次运行原封包的声学测量与bucket-only100步。尾批与目标parity仍未通过，
不会据此派发正式训练或迁移评测，也不会自动提升声学候选。

实际新队列`/srv/multimodal-jev-xingchao/operations/efficiency-study-v2`，
PID1929357/start342139861/cmdhash
`2b449c2a4eeeb3e49ba2fabe0aec34bc3643ffe95cc16f46e733280dcd1bf52d`，112pins。
本地宏`.local/config/efficiency-study-v2.json`。旧108pins及两个封包保持不变。
原队列失败终态不覆盖、不伪造成功、不自动重试。

分桶监听使用`watch_context_bucket_training.py`复用原900秒观察器，仅替换bucket
校验器和准确启动命令。输出`operations/bucket-training-v1-watcher`，owner绑定
新效率队列；尚无训练attempt时只记录本地等待，训练启动后每900秒只读准确容器，
owner或client终态即停，不调用模型。原batch32监听已正常终止。

新增检查：checkpoint契约/真实CPU Adam恢复12项通过（81828，3.98秒）；
新队列准入、执行顺序及原候选契约组合24项通过（36a0be，4.04秒）；
bucket监听/共享900秒间隔11项通过（dd1ef4，2.33秒）。远端任务在队列测试中
使用替身；这些仍不等于NPU速度或精度验收。

16:05UTC尾批故障修复准备完成：新增`context_macro_tail_entry_v2.py`，共享输出
父目录使用`exist_ok=True`，原生trainer的独立rank目录仍`exist_ok=False`，
既支持双rank并发启动，也拒绝复用已有rank结果。新增
`context_macro_tail_operations_v2.py`复用原封包/生命周期，仅替换client CLI。
真实封包`operations/tail-admission-v2-prepared`，manifest
`e5b4499aea3bc48857e3418d83ec1b987014946e000c8dff2a7006889d6b94f1`，
1098个继承Python文件逐字节不变，preflight和双service命令不变；新封包仍禁止
派发，未抢占效率队列。36项相关测试通过（90766，34.87秒），包含实际并发目录
创建和旧rank目录拒绝；封包与validate实际通过（fb5216/2fc251）。
本地宏`.local/config/tail-admission-v2.json`，后续仍需实际NPU尾批及独立输出审核。

16:10UTC本地队列证据确认：`efficiency-study-v2/baseline-audit.json`已写入，完整
父基线重审通过；声学容器
`4df31501699b671b645c31e7df40306e232dae96b9aae8454ba58b8c416ad5f3`
已有准确launch及start-response，远端根`/data/xingchao/jev-acoustic-v3`。尚无完整
测量结果，bucket尚未派发。未新增普通远程查询或修改112pins。

另补独立长音频参考测试：按已保存原生Qwen3OmniMoeAudioEncoder源码的
pad_sequence→布尔mask提取→录音内attention窗口路径，比较100/101/199/200/201/
799帧录音在chunk batch1/4/32的输出及输入/参数梯度，参考路径不调用候选layout
或forward。声学9项测试通过（848513，2.14秒）；使用CPU小层数值替身，不能代替
实际BF16 NPU验收。原有短音频独立参考、逆序及边界检查保留。

## 两项实现

### 实际输入的离线空间估计（16:09UTC）

固定脚本`analyze_context_bucket_padding.py`读取batch32完整100步的真实rollout
processor shapes，核对transport文件hash、12800个slot顺序、引擎请求hash及policy，
并逐步重建实际DP2 rank分配。配置来自portable模板
`context-bucket-padding-analysis.json`和`.local/config/bucket-padding-analysis-v1.json`。
证据：`/srv/multimodal-jev-xingchao/operations/bucket-padding-analysis-v1/report.json`。

| 量 | 原长度排序 | 新分桶候选 |
| --- | ---: | ---: |
| padded token总量 | 36,361,736 | 36,361,736 |
| padded frame总量 | 4,604,800 | 4,604,800 |
| token padding占比 | 0.1198% | 0.1198% |
| frame padding占比 | 7.0991% | 7.0991% |
| 逐步最慢rank成本代理之和 | 20,628,792 | 20,588,920 |

99步选择相同token/frame排序后重分配microbatch到两卡，1步保留原方案；没有选择
二维桶重排。因此本候选的实际离线变化主要是卡间负载分配，最慢rank代理仅降低
0.1933%，不能宣称已减少padding或有明显训练提速。原基线已充分排序，这个结果
说明仅在现有global128内部重排的空间有限。保持预先冻结100步候选，用实测判定
是否保留；不把这个代理换算成NPU速度，不据此调整奖励或跨batch样本成员。

13项测试通过（993364，1.99秒），包括合成100步日志、slot/engine不同ID空间、
文件篡改和rank分配篡改拒绝。首次真实分析因混同两种request ID被拒绝，修正为
重算engine identity hash后成功（47293f）；没有修改原始日志或运行中的候选。

声学入口`tools/reports/context_acoustic_batch_candidate.py`将多个录音拆成完整
100-frame卷积窗口，按最多32个chunk批量执行三层卷积；每个chunk重置位置编码，
仅保留有效输出，再以累计长度隔离各录音的注意力窗口，批量执行声学Transformer。
短录音仍先补齐完整窗口，避免卷积边界改变。没有合并录音、截断输入或复制基座权重。
通过新worker/owner显式安装，原逐录音声学profile保留。

训练入口`tools/reports/context_bucket_trainer_candidate.py`沿用原生verl优化器循环，
仅替换global batch内的rank/micro分配。原基线已经按token长度排序，因此本轮新增
token/frame二维桶、按估计负载分配完整microbatch到DP rank；保持global128、DP2、
micro8、每卡64条、8次累积以及每个slot的标签/权重。算法候选不读取标签选择桶。
token桶宽128、frame桶宽200、两项代理权重均1，通过本地配置设置并绑定checkpoint。
桶规划保留旧方案为候选，代理成本不会高于旧方案；代理成本不是实际吞吐承诺。

## 实验与判定

1. 声学：相同TRAIN128输入、初始policy、推理service32/token131072，两个独立进程
   分别执行旧声学、新声学。每组warmup、两次计时、逆序输入一遍；记录真实logits、
   prob、G16采样、卷积batch shape、encoder边界、worker峰值显存和完整推理时间。
   固定脚本重核输入/策略/容器/传输及完整输出。数值差异保留，不自动放宽准入标准。
2. 训练：bucket-only独立100步与已完成的batch32基线比较；推理仍使用旧声学。
   相同初始模型、全量SLURP日程前100步、global/micro、GRPO奖励与loss、冷reference
   缓存和每步LoRA/head/optimizer保存频率。逐步重核输入、权重、双rank状态、
   rollout、梯度记录及训推概率门槛；统计全部100步及去前2步窗口的总时长、actor、
   rollout、同步残差、峰值显存以及padding/DP成本代理。
3. 声学数值不一致时不接入训练，但bucket-only可以继续。运行失败保留现场并停止，
   不自动重试。两项独立结果出来后再决定联合100步；当前无自动正式训练提升。

这些是效率实验，不能证明任务准确率或迁移提升；后续仍需要全量源域训练、
selection选点、冻结策略后FSC/MINDS迁移和SLURP TEST评测。

## 配置与实际队列

Portable模板：`context-acoustic-probe.json`、`context-bucket-training.json`、
`context-efficiency-successor.json`，均在`configs/portable/operations`，机器路径
集中在`.local/config`宏中。复用已有镜像、独占设备租约、固定生命周期和收集器。

实际声学封包：`/srv/multimodal-jev-xingchao/operations/acoustic-batch-v3-prepared`，
manifest `28516c4dc351d92c969efb8e5369d16c862598bd2363c57d5a0c8c1bd728873f`。
v1仅本地封包时runner命令绑定错误，未派发；v2因新增allocator验证而被v3取代，未派发。
实际训练封包：`/srv/multimodal-jev-xingchao/operations/bucket-training-v1-prepared`，
manifest `b27b58c2ad2a148d22f19098d78438446895163715287b075ee4c90d973c3425`。

新队列`efficiency-successor-v1` PID1924302/start341978963/cmdhash
`7a1a3a526f67038b5e59d7d69cc4ef686ff378d562fcff492d9fd96bb48afe4a`，108pins。
准确/proc身份与全部依赖已核验，等待原parity队列1900510成功；通过后依次执行
声学测量和bucket100，并输出`acoustic-comparison.json`及`bucket-comparison.json`。
原100步、尾批和parity驱动73/81/95pins全未改。原formal/source-curve/full-transfer
三个仅等待的本地驱动已用pidfd准确终止；各目录保留efficiency-priority-supersession.json。
没有终止远端训练或服务，也未取消最终任务/迁移评测目标；后续需要按新证据重新接队列。

## 验证范围

43项测试通过（40394，2.99秒）：实际CPU声学forward、独立短录音参考、混合长短
音频边界、输入逆序、梯度比较、分桶成员/零权重保护、冻结日程校验、原生训练入口
生成、命令隔离边界、测量篡改拒绝和固定队列顺序。远端模型及队列任务在相关测试中
使用替身，不是NPU精度/性能证据。两份真实封包validate已通过（41194）。

本轮另补固定监听`watch_context_macro_training.py`，复用900秒原观察器，10项测试
通过；后台PID1920149/start341887847，每到期才查询准确容器句柄。15:32:17UTC
实际读到双rank98/100，98提交，client/双服务running；下一普通>=15:47:17UTC。
终态收集由原驱动阻塞等待触发，无模型分析轮询。
