# 四卡缓存 GRPO 效率验收协议

本协议只覆盖简单 SLURP 原生意图成员判决，接续本轮 DP4 InfoNCE 终点318。
以 `configs/experiments/context-two-stage-v0.0.11.json` 为算法、数据、LoRA和预算依据。
不改模板、候选列表、奖励、采样种子、精度或选择规则来获得吞吐收益。
历史八卡及旧初始化的效率值仅作容量参考；当前尚未产生本协议的NPU效率结果。

## 准入与固定配置

必须先独立审核同一模型端点、runtime的TRAIN256数值一致性、全量TRAIN reference、
连续两步、跨轮尾批及新进程恢复。之后另封效率包，不能延长有限验收包的stop字段。
每配置固定100更新，达到上限即结束，不进入第101更新。

固定配置：2卡DP训练、2个单卡推理实例，BF16执行，FP32主参数/Adam/loss；
每DP有效batch64、micro8、global128、macro128、rollout32、service32。
训练使用全量40,513条clean TRAIN的原冻结日程前100步，共12,800次真实样本暴露；
这是一项受限效率实验，不等于完成一轮训练或证明全量任务收益。
排序/存在性双信用GRPO目标及所有系数保持冻结；reference固定为本轮318终点完整缓存。
启动前核验两推理实例、actor、缓存的策略/源码/processor/输入/候选身份；缓存缺失拒绝。

工程候选入口：`tools.reports.context_cached_efficiency`，模板
`configs/portable/operations/context-cached-efficiency.json`读取`${CACHED_EFFICIENCY_CONFIG}`。
prepare配置包含`parent`、`reference_job`、`acceptance_kind="efficiency"`、
`acceptance={prefix,tail,recovery}`（三项完整独立audit配置）、新`output/remote_root`。
复用现有只读reference封包及已封存训练器，源代码逐字不变；manifest单独声明100更新。
派发必须显式`efficiency_dispatch_allowed=true`，重新审核三项真实有限结果，
核对相同source/runtime/reference及prefix恢复来源，保留同一host/lease。

当前效率候选保持`checkpoint_interval=1`，逐步保存可训练参数与optimizer，
对应审核器要求完整100步分片、BF16导出、两实例激活、训练侧行为一致性及reference命中。
该成本与正式稀疏checkpoint不相同，不直接外推正式634步耗时。没有复制主模型。
独立结果审核输出`cached_100step_efficiency_outputs_audited`后才可使用吞吐统计；
封包、CPU测试或进程exit0均不替代这一结果。

2026-11-17：效率封包/门禁/输出、成本统计及提交格式组合54项CPU回归通过。
测试覆盖真实封包重构、实际生成runner及readonly reference绑定，NPU前驱为显式替身；
当前三项真实有限验收尚未完成，因此未生成或派发本轮100步效率包。

固定后继入口为`tools.reports.context_cached_efficiency_successor`，模板
`configs/portable/operations/context-cached-efficiency-successor.json`读取
`${CACHED_EFFICIENCY_SUCCESSOR_CONFIG}`。配置只声明绑定SHA256的有限验收驱动/config/terminal、
新输出目录以及效率实验package/remote_root/operations_root/tag和显式派发开关。
它通过pidfd等待准确前驱，随后核对三项实际audit配置、相同host/lease/reference，
复用效率封包的独立重审，执行100更新、准确wait/collect、独立审核和成本统计。
失败保留句柄并停止，无自动重试，不派发正式训练，不修改排队中的前驱或runtime。
重入拒绝覆盖既有目录；状态字符串不代替数值、恢复与完整传输证据。
普通状态检查仍为900秒；准确进程等待不产生轮询或模型调用。

2026-11-17：后继流程、效率封包及成本统计共71项CPU测试通过，包括错误前驱、
配置变更、重叠路径、异host、失败收集/审核/成本统计和重复执行拒绝。
测试中的NPU执行及前驱审核使用显式替身；不把测试通过称为真实效率验收。

## 统计与对照

记录100步完整曲线。主报告保留全部100步；同时单列预登记前5步warmup之后的95步，
不能事后挑选更快区间。只计真实query，padding不计吞吐。
使用现有 `context_macro_efficiency_summary.summarize` 统计两rank完整step时间的最大值，
以及preparation、rollout、reference读取、behavior与credit、actor、导出/checkpoint/sync。
嵌套耗时不相加，完整step与容器墙钟分别报告。

另列冷启动、模型加载、服务准备、reference建库、独立审核、封包/传输/收集成本。
以准确容器start/finish和完整产物计量墙钟；分清分配卡时与实际活跃设备耗时。
reference建库单独列一次性成本，不能把它移出训练循环就声称端到端节省同等成本。
若只能获得容器墙钟而不能得到细分计时，明确标记不可得，不推算成实测值。

成本汇总固定入口为`tools.reports.context_cached_efficiency_cost`，模板
`configs/portable/operations/context-cached-efficiency-cost.json`读取
`${CACHED_EFFICIENCY_COST_CONFIG}`。本地宏声明`run_root`、新`output`及以下
`{path,sha256}`证据：`training_config`、`reference_build_config`、`reference_result`、
`training_inspect`、`training_handle`、`reference_inspect`、`reference_handle`、
`training_transport`、`reference_transport`。
它复用既有100步/5步warmup统计，将计时文件、配置和reference结果逐项绑定到准确容器的
收集清单。只允许同镜像、同四卡映射、先建库后训练、完整40,513条固定reference及零miss。
一次完整reference容器成本计入本次100步实验，不假设未来复用次数来摊薄。

输出分别列训练loop、reference builder与两容器时间；四卡分配卡时为两容器时间之和乘4，
不是实际NPU活跃卡时。容器之间的等待间隔单列。`container_non_step_seconds`和
`container_non_build_seconds`仅表示余量，不将其冒称冷启动时间；控制器审核、传输/收集、
实际设备活跃卡时及正式634步成本明确标不可得。工具不作数值/训练正确性准入或加速结论。

2026-11-17：本地36项CPU回归通过，包含完整100步合成记录、95步稳态窗口、
reference成本只计一次、错容器/错策略/错设备/计时越界/篡改文件拒绝。
当前尚无本轮100步真实产物，因此未生成实测成本报告，测试数字不作为实验结果。

历史74.7641秒/步等结果来自不同初始化和runtime；与当前结果并列时标为非匹配参考，
不计算为本轮优化的因果收益。宣称某个工程改动加速时，需匹配起始策略、输入日程、
目标函数、卡数/型号、镜像、精度、checkpoint频率及同一100步窗口，仅改变该项工程策略。
冻结缓存方案是当前训练契约，本轮不为制造对照而恢复在线reference补算。
缺少可信匹配对照时，先报告绝对吞吐及成本，不宣布加速百分比。

## 正式训练衔接

效率包与正式包分开；100步效率终点不续接成正式效果曲线。
正式634更新从同一个Stage1终点重新开始，使用预登记的11个评测点，
统计全量selection上的set-F1、top1、macro-F1、NLL、Brier与退化，最终冻结后评测TEST和FSC/MINDS。
效率报告不能替代源任务或迁移收益；FSC/MINDS只称跨数据集迁移。

正式运行的稀疏恢复checkpoint与每步behavior导出分开计量；只保存LoRA/head及必要optimizer，
主模型仅路径/hash引用。固定脚本负责采集、统计与审核，不触发逐步模型分析。

正式入口为`tools.reports.context_cached_formal`，模板
`configs/portable/operations/context-cached-formal.json`读取`${CACHED_FORMAL_CONFIG}`。
封包配置包含同一`parent/reference_job`、完整`efficiency_audit`、
`acceptance_kind="formal_source"`、`evaluation_steps=[0,64,128,192,256,320,384,448,512,576,634]`
和新`output/remote_root`。prepare/audit-package重审100步效率证据，逐字复用原runtime，
仍从Stage1终点初始化；不以效率终点续训。正式配置不含checkpoint_interval，
仅预登记10个非零评测点保存optimizer；所有634步导出并激活LoRA/head策略。

launch/run要求`formal_training_dispatch_allowed=true`、相同host/lease和reference审计输入，
复用准确容器等待与收集；run进一步独立审核634更新、每轮完整覆盖、81,026真实暴露、
81,152个含padding的缓存消费slot、零在线reference调用及完整策略发布。
结果写入operations_root/finalization，失败保留句柄且不自动重试。只有训练审核成功
才输出`formal_cached_training_collected_and_audited`，selection和迁移仍单独标为待评测。
运行须使用后台生命周期。新增[正式训练后继](../docs/engineering/context-cached-formal-successor-v1.md)
等待完整Stage1曲线结束并重审同一100步效率证据，再封包派发；登记后继不代表正式NPU已执行。

该入口修正正式审核的两个集成缺口：给日程校验传入derived plan，并将正式实验契约
纳入checkpoint契约。修改仅在新调用方生效，排队的有限/效率审核器及封存runtime不改。
2026-11-17：正式封包/准入门禁/生命周期、实际生成runner、生成的正式checkpoint发布、
完整两轮日程及契约回归53项CPU测试通过。NPU计算/已完成效率前驱为显式替身，
不代表正式路径已经NPU执行或获得任务收益。
