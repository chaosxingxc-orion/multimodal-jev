# GRPO DP2 microbatch 效率对照

本阶段只开展简单任务：全量SLURP单音频意图判决的效率与收益，以及FSC/MINDS
冻结权重跨数据集迁移。复杂任务及其工程准入留到下一阶段，不构成本阶段门槛。
SLURP的困难样本仍保留；任务结构简单与样本难度是两个维度。

前置证据已具备：v6 两步独立审计及 replay-v1 新进程恢复比较均通过。
恢复审计逐张量比较第二步 groups、execution、两rank参数和Adam状态，策略身份相同。
证据 `context-macro-replay-v1-observed.json`；冷缓存恢复单步89.96秒，仅作恢复验收。

统一入口 `prepare_context_macro_efficiency.py` 重新审计实际replay后才封包。
同一源码、镜像、Stage1策略、GRPO两路信用、CPU FP64排序协议、global128、
TRAIN前100批、双TP1、冷reference缓存、每步checkpoint。三组仅microbatch8/16/32、
相应DP2长度分组及独立输出路径不同；训练端累积次数分别8/4/2。
没有重采样、主模型复制或目标数据反馈，不将历史FP32实验作隔离提速对照。

效率模式需显式experiment_kind=efficiency，起点0，预算3..100个完整optimizer更新，
checkpoint_interval=1。原两步工程模式及新进程恢复保持原预算，不扩大历史包。
训练循环、operations验证器及独立审计读取同一预算；审计仍覆盖每一步的完整macro、
目标/奖励/优势、所有microbatch对齐、两实例激活、参数及FP32→BF16导出。
源端发布权限修复进入所有新包，冻结replay仍通过独立有哈希收据的权限修复收集。

按micro8、micro16、micro32顺序运行；通过前一组审计且显存允许后再推进下一组。
大microbatch若OOM或未通过对齐，记录实际失败，不静默减小batch冒充原配置。
相同全局batch在BF16下改变累积顺序可能有数值差异，不能声称跨配置参数逐位一致；
各组内部仍必须通过实际behavior/actor一致性门槛。任务效果需后续完整选模/迁移评测。

固定脚本报告完整step耗时、actor/rollout/reference/保存同步、吞吐及actor峰值显存。
前两步另列warmup，全部100步同时报告；reference冷/热单列，不将嵌套时间重复相加。
启动和评测不含在step计时中，不以当前表替代端到端总成本。
配置模板 `configs/portable/operations/context-macro-efficiency-prepare.json`，
机器绑定仅在 `.local/config/macro-efficiency-v1.json`。

三组已完成实际封包与validate，micro8开始派发。首次启动因嵌套父目录不存在在mkdir
阶段失败，尚未创建容器；保留原attempt，修复统一启动器创建父目录后，使用同一冻结包
派发micro8-attempt2。preflight已创建，尚无100步完成或效率对照结论。
micro16/32保持prepared，不与micro8争卡。完整收集/审计由准确等待driver的finalizer负责。

结果比较入口 `compare_context_macro_efficiency.py` 和对应portable模板已完成，
逐组重新执行完整审计，再核对相同源码、除microbatch/输出路径外的完整训练和服务配置、
实际batch记录顺序及每步reference misses。报告全窗口及排除前两步的速度比，
不要求不同BF16累积顺序下的参数逐位相同。10项比较测试通过；实际三组准备配置
规范化后一致，证据 `operations/macro-efficiency-v1/prepared-controls-audit.json`。
这仅证明封包控制条件一致，目前没有已完成的效率结果比较。

收集器改为流式hash和解包，归档/传输超时默认1800秒，可通过本地配置
`collection_transfer_timeout_seconds`调整（55..7200），容器inspect仍55秒。
7项实际归档程序的本地替身测试覆盖哈希/完整文件集/符号链接拒绝/时限；
连原审计和比较共25项通过。micro8运行中的等待finalizer已在核验准确PID及尚未开始
收集后替换为新版，原等待记录归档；训练driver及冻结包保持不变。

日常观察使用 `observe_context_macro_training.py`，沿用同一operations宏配置。
900秒间隔从attempt启动或最近观察尝试计算；SSH失败也消耗间隔，未到时不发远端请求。
仅核验准确容器并读取双rank小型进度，不收集中间权重、不自动重试训练。2项测试通过。

## 固定串行驱动

`run_context_macro_efficiency_sequence.py`与portable模板
`configs/portable/operations/context-macro-efficiency-sequence.json`连接既有三组实验。
通过PID/start_ticks/cmdline哈希准确等待micro8原finalizer，而非另起收集器；终态成功
并重新审计后，依次调用共享训练/设备lease/容器等待/收集审核入口运行micro16、
micro32，每组100步。任一启动、训练、对齐、收集或审计失败就停止并保留准确句柄，
不自动重试，不减小microbatch，不派发正式训练。未通过micro32时不得声称三组
对比完成，之后按实际失败证据决定是否采用两组结果另行修订容量选择协议。

三组完成后执行已有compare脚本，报告所有100步和去前两步统计。配置和驱动已
加载的本项目源码共70文件pin，在等待后及每次下一组派发前核对，避免等待期间
源码改变静默混入实验。运行中的冻结训练包保持不变。普通观察器仍900秒；该
驱动只使用准确进程/容器等待，不调用模型做循环分析。

20项串行顺序、失败停止、重复启动拒绝、准确进程等待及旧比较回归通过2.82秒。
真实三组prepare、规范化控制条件及原finalizer PID1816873/start339823834均核验。
本地宏`.local/config/macro-efficiency-v1-sequence.json`；驱动PID1849133/start340484633
已确认存活，位于operations/macro-efficiency-v1-sequence-driver，实际等待记录在
operations/macro-efficiency-v1-sequence/waiting.json。此时尚未派发micro16/32，无新
效率或任务收益成绩。请勿另行手动启动后两组或并行收集micro8。

## 已收集运行的固定瓶颈分析

`analyze_context_macro_efficiency.py`使用portable模板
`configs/portable/operations/context-macro-efficiency-analysis.json`，机器路径集中在本地宏。
先重新审计已收集运行，再按互不重叠的阶段拆分完整更新耗时，避免把rollout及其
内部reference、behavior重复累计；输出总窗口、去warmup窗口和各阶段成本。
该脚本不访问正在训练的进度，不改变训练、采样或目标集。

2026-11-16实际v6两步（256次query曝光）重审成功，共162.114秒、1.579 query/s。
本次warmup=0，两步均纳入；不能视作稳定100步吞吐，亦不包含启动、收集与评测。

| 阶段 | 两步秒数 | 更新窗口占比 |
| --- | ---: | ---: |
| actor更新 | 79.282 | 48.91% |
| reference | 31.466 | 19.41% |
| behavior与credit | 28.876 | 17.81% |
| rollout外准备 | 11.327 | 6.99% |
| 导出、checkpoint与同步 | 9.243 | 5.70% |
| rollout其余 | 1.920 | 1.18% |

双rank完成时间差另计0.000815秒。actor峰值allocated为39,804,589,568字节。
首先完成micro8/16/32的100步隔离对照以验证actor收益，再依据正式报告判断是否值得
优化reference与behavior的大块成本。若仅将actor耗时减半，其他时间保持不变，
这两步成本模型对应1.324倍吞吐；这是反事实估算，不是已测提速。

独立信用审核中，排序非零分量527/15104（3.49%），存在性843/15360（5.49%）；
分量加权reward均值分别0.9910与0.9082。这提示后续需检查信号利用率，但分量不是
query，不等于准确率、难样本占比或梯度有效率。效率对照期间不改变reward、采样
及loss；后续仅基于源TRAIN/selection评估信号改进，同时验收排序与概率校准。

实际报告由本地宏`.local/config/macro-v6-efficiency-analysis.json`指定输出，文件为
`/srv/multimodal-jev-xingchao/operations/macro-training-v6/efficiency-analysis-v1.json`。
14项分析与既有summary测试通过；真实报告来自已收集v6产物重审，非合成运行。
当前仍无新版全量源评测或迁移收益结论；FSC/MINDS不能代替不同判决语义的跨任务证据。
