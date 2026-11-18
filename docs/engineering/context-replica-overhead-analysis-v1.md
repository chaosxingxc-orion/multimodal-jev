# 推理实例数量与 overhead：当前证据及锁定条件

2026-11-16。结论：以尽快完成训练为目标，当前暂留 actor DP2 + 两个 TP1
推理实例；尚未完成同条件单/双实例完整训练对照，不能宣称已锁定最优拓扑。
这里的单实例指一张卡 TP1，不指两卡 TP2。

最新配置优先级：用户随后指定推理内部batch32/48、每DP副本有效batch64。
因此下文固定global128及DP3常规补齐方案仅保留为修订历史；当前执行设计见末节。

## 固定脚本与证据

统计由 `tools/reports/analyze_context_replica_overhead.py` 离线生成，配置为
`configs/portable/operations/context-replica-overhead-analysis.json`，本机绑定为
`.local/config/replica-overhead-v1.json`。运行：

```bash
PYTHONPATH=src python tools/reports/analyze_context_replica_overhead.py \
  --config configs/portable/operations/context-replica-overhead-analysis.json \
  --local .local/config/replica-overhead-v1.json
```

输出由本地宏指定；本次实际报告为
`/srv/multimodal-jev-xingchao/operations/replica-overhead-v1.json`。
脚本复核读取文件与归档 transport 清单的 hash、历史两组输出和 G16 采样一致性。
这是归档计时分析，不是重新执行完整运行时审计，也未改变运行中的冻结包。

## 推理本身

历史同策略、相同 128 条输入，服务内部 batch4：

| 部署 | 两次 timed pass 平均耗时 |
| --- | ---: |
| 单 TP1 实例 | 17.232 秒 |
| 双 TP1 实例 | 8.768 秒 |

描述性比值 1.965 倍，记录、logits、概率与 G16 采样一致。不过两组来自不同运行，
调用及计时边界不同，不能作为同条件完整训练加速比。

已审核 v6 两步训练中，四个 macro 分别为 reference/behavior/reference/behavior。
每个 macro128 分成四个 rollout32，每实例处理 64 条：

| macro | 完成耗时（秒） | 外层剩余耗时（秒） | 两实例任务时长均衡度 |
| --- | ---: | ---: | ---: |
| step1 reference | 16.839 | 0.099 | 99.48% |
| step1 behavior | 8.761 | 0.094 | 97.09% |
| step2 reference | 12.167 | 0.093 | 96.80% |
| step2 behavior | 12.222 | 0.097 | 96.53% |

外层剩余耗时 = macro wall time − max(各实例 job 耗时之和)，占 macro 的
0.59%–1.07%。job 内已包含 IPC、模型执行和校验，因此这不是全部 RPC overhead。
均衡度 = sum(load)/(2×max(load))，不是 NPU 利用率。

## 完整训练步的成本

| 成分 | step1 | step2 |
| --- | ---: | ---: |
| 完整步 | 82.045 秒 | 80.068 秒 |
| actor | 41.095 秒 | 38.187 秒 |
| preparation（包含 reference、behavior、credit 等） | 36.325 秒 | 37.263 秒 |
| 导出、checkpoint、同步等剩余部分 | 4.625 秒 | 4.618 秒 |
| reference 阶段 | 17.391 秒 | 14.075 秒 |
| behavior + credit 阶段 | 12.669 秒 | 16.207 秒 |

后两行属于 preparation，不能再次相加。behavior + credit 扣除推理 macro 后
还有 3.91/3.98 秒，包含 CPU credit 和校验，不能全归为通信。
step2 两次阶段切换（两实例各自从 behavior 切 reference、再切回），更新后还要
激活新策略。当前没有独立 activation 计时，4.62 秒也不能称作纯同步开销。

actor 与同一步推理串行执行，两个推理实例在约 40 秒 actor 更新期间没有该步推理
工作。这是占卡成本的重要来源。reference 固定且有缓存，第二轮命中后减少推理与
切换，双实例相对收益可能下降。当前两个训练步均为 128 条 reference miss，不能
外推成全训练平均。

当前 macro128/rollout32 是交付粒度，服务 max_batch_queries=4、token budget16384
仍决定内部计算批次；不能把完整 macro 交付误认为一次 forward128。扩大服务内部
batch 是候选优化，需要显存、变长输入与训推一致性验证。actor microbatch8/16/32
的现有实验是另一变量，不构成推理实例数量对照。

固定 actor DP2 时，单实例共 3 卡、双实例共 4 卡。按分配设备秒计，双实例更省
资源的必要比较条件为 T_single/T_dual > 4/3；但用户当前优先 wall time。
将当前两个实例的 job 时间简单串行相加，估算单实例约 102–107 秒/步，双实例实际
80–82 秒/步，即约 1.28–1.30 倍。这仅是保持其它成本不变的反事实估算，不能替代
真实单实例测量，亦未计单实例 CPU 竞争、缓存和批处理变化。

## 工程基线锁定前的有限对照

1. 现有 microbatch 对照完成后，固定其优胜配置及相同 BF16、模型/LoRA/head 起点、
   数据顺序、global128、GRPO/奖励、服务 batch/token budget；仅改变实例数量。
2. DP2+1TP1 与 DP2+2TP1 各最多 100 步，使用同一套可配置脚本。
   预声明相同 warmup 排除规则，同时报告包含启动成本的总耗时。
3. 冷 reference 与预先填充的命中 reference 使用同样输入分别比较；预计算成本
   单独计入，不能向某一组免费提供缓存。完整比较成本用预期训练轮数加权。
4. 主指标为完整 update 耗时、有效 query/s、100 步总耗时；同时报告 p50/p95、
   分配卡秒、显存峰值、缓存命中数、CPU credit、RPC 序列化、激活和 actor 计时。
5. 同输入同策略推理输出、LoRA/head 加载与更新一致性验收通过后，按总训练 wall
   time 选择；若两者差异处于重复测量波动内，优先单实例减少占卡与部署复杂度。

本次只完成分析及归档，未启动上述拓扑实验，未停止现有 microbatch 串行实验。

## 用户修订：固定四卡预算，优先验证 DP3 + TP1

用户指出减少推理实例应把释放卡用于训练。这修订上文固定 DP2 的主比较：
目标是 **DP3 + 单 TP1** 对 **DP2 + 双 TP1**；DP2 + 单 TP1 仅可作为拆分因果的
辅助测量，不能代替固定四卡资源下的最终选择。

actor 从两卡到三卡的理想计算时间为原来的 2/3（约 40→26.7 秒），这是忽略
通信、负载不均及固定成本的估算，不是已经证明的大幅加速。仅把原推理串行化可能
抵消该收益，因此必须同时提高单实例的实际吞吐，再比较完整 update。

源代码核实：batch4/16384 来自 `prepare_context_batch_probe_candidate.py` 的
前期验证设置，并非已测容量极限。已归档 TRAIN128 输入展开 token 总数为363216，
在既有长度排序下，32/64/128 条分组的最大总 token 分别91592/182422/363216。
这些是输入预算，非显存预测。旧 HF eager 路径的 OOM 不能证明当前 vLLM 分段
attention 路径的容量上限。

当前有三层约束：macro128、每次发给实例的 rollout32，以及服务内部 batch4。
若要单实例一次接受完整 macro，rollout_queries 应为128；服务应允许完整128请求
交付，vLLM内部根据实际 token、音频长度、激活内存预算调度。当前自定义 owner
把 encode 总 token 与调度 token 上限绑定，且关闭 chunked prefill，因此不能
声称现状已支持大请求队列/小计算批次独立调度。只改 max_batch_queries=128 无效。
声学塔当前仍按录音逐条 forward；增大语言侧 batch 不会自动把声学侧并行化。

固定策略容量对照应覆盖16/32/64/128，batch4保留作历史锚点，记录实际 encode/
worker forward 数、峰值显存和完整宏吞吐；先通过数值一致性，再投入最多100步
训练对照。未测满128容量前不保证128一次 forward，也不预设4为最优。

DP3需要解决128不能整除3及microbatch的问题。保持128条真实样本与统一目标分母，
补零权重槽位并让三个rank执行相同数量collective/backward；例如micro16需要144
槽（每rank48），micro32需要192槽（每rank64）。零权重padding也消耗计算，需计入
效率；不得直接求各rank局部均值或悄悄将global改为192。使用跨rank真实权重和
对应DDP/FSDP平均的缩放，并以不等长/尾批和单进程参考梯度验证后才能派发。

候选 `context_replica_macro_rollout.py` 与 `context_replica_macro_client.py` 复用原
流程扩为显式1/2实例，不修改正在执行的70个源码/配置pins。新协议schema为
jev-replica-macro-client-v1；目前仅候选调度/连接组件，DP3训练、launcher、容量
验收及完整审计尚未接入，不能称已完成拓扑优化或部署。

## 最新规则：每副本64，推理batch32/48

不再为DP3保持global128。定义清晰区分总训练卡数、每副本TP/PP占卡及DP副本数：

```
DP副本数 = 训练卡数 / (TP × PP)
global_batch = 每DP副本有效batch64 × DP副本数
每副本有效batch64 = microbatch × 梯度累积次数
```

当前TP=PP=1，DP2为128、DP3为192；micro16累积4次、micro32累积2次。
DP不是要再除一次的数据并行副本数。推理服务内部batch仅比较32/48，macro随
global batch派生为128/192。DP3单实例服务batch32对应6个内部批次，batch48对应
4个（仅在token/音频容量允许且无额外scheduler拆分时）；交付仍须完整macro。

这属于每卡工作量固定的扩展：DP3每步处理更多样本，actor每步不再能直接用此前
40×2/3估算。效率主指标为有效样本/秒、等量样本成本及整轮耗时；100步DP2/DP3
分别最多12800/19200曝光，不能只比较100步总时间或把相同步数当作等训练预算。
正式源训练按同两轮完整人口对齐，目标集仍不参与选择。

固定规划入口`plan_context_topology_batches.py`和portable模板
`configs/portable/operations/context-topology-batch-plan.json`，本地宏
`.local/config/topology-batch-plan-v1.json`。报告位于
`/srv/multimodal-jev-xingchao/operations/topology-batch-plan-v1.json`。
已生成DP2/DP3×服务32/48四组候选；micro16在当前计划中只是可配置占位，待已有
效率实验选定。现有40513条源数据按逐轮保留尾批计算，DP2两轮634步，DP3两轮424步；
DP3每轮最后1条真实样本，故尾批处理仍需验证，不能直接套旧634步恢复/评测计划。

规划器与真实调度函数的14项CPU测试通过，包含跨拓扑相同完整epoch真实样本顺序、
每rank64槽位、尾批权重及TP/PP除数检查。不是实际DP3/NPU训练验收。
既有运行冻结包及串行microbatch实验不修改，新规则需重新封存训练日程、macro192
传输及审计、DP3执行、32/48服务容量和评测检查点后才可用于正式运行。

## Macro192 传输组件进展

新增`context_replica_bulk_rpc.py`将候选单次请求上限扩为192，
`context_replica_batched_service.py`复用原完整预校验、去重、内部token/数量分组、
原顺序恢复与原子完整交付；`context_replica_runtime.py`复用原模型owner和runtime
source/processor校验，增加显式macro192握手（旧128服务不能被误接）。
`context_replica_macro_client.py`现已连接新协议，容许完整192 macro。
候选服务有独立CLI入口，未封包、未派发。

42项本地测试7.16秒通过：真实Unix socket/payload写入/读取/hash/删除/原子journal，
覆盖1/2实例、32/48内部批次、完整192输入身份及顺序、策略切换确认、错误容量握手、
最后一个输入非法、中途模型失败后禁止partial commit和重试，以及旧bulk/service回归。
模型和策略加载为替身；存储挂载类型校验在临时测试目录放宽，原子提交/fsync真实执行。
单实例192→6×32或4×48的测试验证的是调用分组，不是NPU forward数或速度。
70个运行中sequence pins复核未变；DP3 actor、真实服务显存/精度/吞吐和部署审计仍待。

2026-11-16 12:03:33 UTC固定observe确认micro8-attempt2双rank100/100，100checkpoint，
client与两service均exit0/nonOOM；末步87.632秒（不是100步均值）。原finalizer和
sequence PID仍live并负责收集审核，未并行collect或手动派发后继。
普通下一观察不早于12:18:33 UTC，准确终态等待和既有驱动不受此限制。

## 全量 DP3 日程与 native verl 验证

新增`context_replica_schedule.py`和`prepare_context_replica_schedule.py`，先验证原
完整日程hash及发布人口，再逐epoch删除原padding，保持真实样本顺序，按64×DP
重新分批。rank分配在执行时基于实际encoded长度计算，不沿用旧DP4 rank索引。
规划/调度15项测试0.49秒通过，覆盖人口遗漏、重复、顺序与权重篡改拒绝。

真实本地宏`.local/config/replica-dp3-schedule-v1.json`已生成
`/srv/multimodal-jev-xingchao/operations/replica-dp3-schedule-v1`。完整40513×2人口
独立对照原epoch逐条顺序相同，424批/81026真实曝光；每轮211×192+1。
`batches.json` SHA256为cf878142804a6ca95f0e7f4fb057a828711e5e98d74043dea3e6def1ad3a0d2e。
这是候选日程，当前micro16/推理32仅为配置绑定，不代表已选定赢家。

`test_context_replica_actor_dp3_resume.py`复用原native verl/DDP真实进程测试，world3、
global192/local64、micro16/32，仅joint GRPO；分别测试192真实样本与1真实样本尾批，
梯度对照单进程全局目标，连续两步与新进程恢复逐张量比较，并注入单rank分母/
behavior错误。固定镜像CPU封包372文件，容器
2119b2dc13ff71cc4a3168418f9ff2e789f7f3feb4baf69dd29c0045b61cc7ad已启动且12:10UTC
inspect确认running/nonOOM。准确docker wait session28849仍待，尚未收集或通过。
配置`configs/portable/operations/context-replica-dp3-cpu-tests.json`，本地宏
`.local/config/replica-dp3-cpu-v1.json`；预期5tests零skip，收集沿用collect_fixed_cpu_tests。
测试为合成FP32 logits，无NPU设备，不代替BF16 FSDP、音频及实际服务容量验收。

上述CPU运行现已终态：准确wait28849返回0，独立collect_fixed_cpu_tests确认镜像、
命令、无NPU设备、来源372文件及归档hash；5tests/291.56秒，零failure/error/skip，
exit0/nonOOM。报告`context-replica-dp3-fixed-image-cpu-v1.json`。两种microbatch、
全量192/尾批1的真实DDP梯度对全局目标，以及新进程恢复参数/Adam/采样/梯度逐张量
检查已通过；这是CPU分布式语义证据，不是NPU性能或音频数值证明。

新增`context_replica_trainer_candidate.py`，从原macro trainer派生有限训练入口，保留
原native verl update、优化器、LoRA/head导出、失败与恢复语义。配置同时绑定实际
训练卡集合、DP2/3、每副本64、新macro client及新日程；原engineering两步/replay及
efficiency最多100步限制保留，尚未提供正式424步入口或部署启动器。
新的schedule loader每次重建经hash验证的父日程，比较实际完整新批次、顺序、
padding身份/权重及总人口，不仅核对hash。已用真实发布和父日程复算424批81026
曝光成功（13881终态）。24项新loader/候选接线/旧macro回归0.77秒通过，涵盖篡改后
重封hash仍拒绝、设备不匹配与候选生成器。70sequence pins保持不变。
