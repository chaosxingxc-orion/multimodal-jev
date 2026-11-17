# 连续 context RL：设备内存与常驻交接候选

最新运行状态：2026-11-14旧selection cd263613正常exit0，租约桥释放后原队列
启动actor a0198f4b2ec3efba67f7d860e1b43a7b3824e3acdd3168966e0383d2a61f552c
及service b685080bff7a8ef44f40decbef751ab776705d32abccacb6fa3623a20e142c3f。
remote1794670284.0934343实际inspect两者running/nonOOM；固定镜像和设备隔离
actor4–7/service7核验正确，证据`.local/decision-objective-research/context-online-npu-v1-start-observed.json`。
driver362663接续原验收链，尚无更新/恢复完成证据；普通下次>=1794670884.0934343。
Stage1全量driver仍等同锁，旧评测CPU归档原链继续，不新增或重复派发。

2026-11-14。本候选只补既定连续RL发布缺口，不修改context-prob-v1冻结源码/配置。
前一轮完成冻结身份复核及活跃缓存观察，本轮推进实际actor/engine交接。

## 设备约束的实际证据

已收集的 `context-resident-switch-npu-v2-collected.tar` 中phase-2.log报告：
engine权重1.53GiB、峰值activation 0.17GiB、non-torch 0.2GiB、KV cache22.48GiB，
warmup后allocated24.0GiB、reserved24.02GiB。现有actor短测峰值约38GiB。
这不足以证明二者能在64GiB卡安全共驻；不能把0.6B权重体积当作训练总内存。

候选仅显式设置 `kv_cache_memory_bytes=1073741824`，其余输入4096、60决策、BF16、
TP1及数值runtime不变。新容器a3c811c7897b802a99fd92129301acc3eaab0db4508fe8c430f2ce957c6844cd
已在175物理7完成，exit0/nonOOM，准确wait工具session44072和独立collector PID339685均完成。
对照仍为已保存的两个策略、16真实学习音频、A/B/A与独立HF参考，不重新训练参考策略。
独立收集 `context-resident-small-cache-npu-v1.json` 证实48×60概率对HF全零差、原始策略
张量精确；两次切换0.801/0.793秒。实际日志KV容量9344 tokens，可承载4096单请求。
这项通过只证明低KV预算的切换/概率，不证明DP4 actor共驻峰值或连续RL收益。

## 常驻接口及候选训练器

`tools/jev/context_rollout_rpc.py` / `serve_context_rollout.py`提供单owner Unix socket服务：
控制消息带严格序号，输入文件以SHA绑定、weights_only读取，只准入音频张量/marker位置；
标签、奖励与动作不发给engine。score绑定当前策略，switch验证joint资产和worker收据。
观察/执行失败使连接不可继续，不自动重试不确定更新。

训练侧 `ContextRolloutSession`先按整批补固定reference缓存，再恢复当前行为策略；
当前行为每批重算，G16采样、固定reference、辅助衰减及完整组优势归约保持原定义。
`train_context_online_candidate.py`复用已冻结组件连接完整40513人口/634日程，
rank0通过IPC组批、全局组优势计算后分发四rank，调用原生verl actor更新；
每个候选更新保存master/Adam、组动作/奖励/优势、逐micro概率检查、联合策略导出，
激活下一策略后再进入下一批。支持固定拓扑checkpoint恢复，但尚未真实执行验收。
目前只允许最多两更新的engineering prefix，明确拒绝正式Stage2派发。
探针起点须标注诊断性质，不能用未完成Stage1策略冒充正式RL起点。

新增IPC的8项测试使用真实Unix socket及张量文件、替代engine；包含错序、过期policy、
标签泄漏、hash及路径拒绝。与session、online batches、reference cache共33项CPU测试通过。
训练器/服务编译通过；本地无verl/NPU，编译及替代测试不构成真实循环准入。
context-prob-v1原364项身份仍通过。

## 下一项执行验收

先独立收集低KV预算证据；通过后在统一租约下，隔离TP1服务仅见物理7、DP4 actor见4–7，
先启动服务再启动actor，训练与rollout分阶段执行，测实际共驻内存。
沿全量日程做两次更新，必须由新策略生成第二批行为；再新进程从step1恢复重放step2，
独立比对全部rank的master/Adam、联合BF16策略和采样/奖励。通过后才讨论Stage2发布。
现有Stage1/cache/selection自动链保持，不抢占运行中的训练或新增第五张卡依赖。

## 2026-11-14 连续候选审计补齐

物理6占用已确认属于本项目旧`class-spread-batch-dp4-selection-v1`，容器
`cd263613020955be94f688ab3544d6244cd00bf5008867390a2b615bf6bdf9bf`。
14:07:58 UTC核验仍running/nonOOM，HF进程CPU累计约43分钟；未停止或重启。
旧评测驱动PID229436、Stage1接续307877和选择集接续310979均仍live。
新的两更新候选未创建远程容器，本轮只重新prepare-only封装候选。

补存每query的原始behavior/reference record，供独立collector重新计算FP32
log1p(-p)/log(p)、绑定完整输入身份，并按317/update/request-slot独立重放G16采样。
新增`tools/jev/context_rollout_audit.py`及破坏性测试：即使同时篡改动作和动作hash，
采样重放也必须拒绝；固定参考、输入、随机seed、两侧logprob均单独检查。
与online/session/reference/RPC合计40项CPUtests过，候选trainer/collector编译通过。
初次使用项目.venv因无torch在收集阶段失败，改用既有/usr/bin/python、CPU torch2.10.0
执行通过，未安装或更新生产依赖。证据`context-online-sampling-audit-cpu-v1.json`。
初始化异常资源释放也扩至outer finally；尚未声称NPU异常恢复已验证。

本轮到期观察remote epoch1794665456：原输入cache10240/40513、逆序assist25600/40513，
均running/nonOOM、各自hits0；不是唯一人口完成清单，仍由原writer最终发布与独立审核。
下一次普通缓存观察不得早于1794666056。原364项冻结校验仍通过，Stage2未发布。

## 恢复入口与已有评测的设备租约

本轮发现旧selection启动器只在创建/启动阶段持有共享设备锁，随后退出，
其容器持续运行；这解释了新的DP4候选拿到锁仍因物理6繁忙而被预检拒绝。
未修改旧任务，新增`hold_existing_selection_lease_v1.py`，PID357467已持共同锁并
对明确的cd263613容器执行准确`docker wait`。收到退出码才释放；SSH观察失败保留锁，
600秒后等待同一handle，绝不重启/终止容器。原Stage1链可等锁，避免完成缓存后因
已知占卡直接失败。状态`context-existing-selection-lease-v1-status.json`。

连续候选launcher/collector新增`--resume-replay`：只有前两步已独立收集通过才允许
从checkpoint1配置新进程，初始化engine为step1策略，固定reference仍为原起点，
日程从update1继续，源runtime必须与连续运行封装hash一致。恢复来源只读绑定，
master/Adam分片、commit、joint policy和native adapter逐文件绑定。
此入口已编译，尚未因前置实际连续运行未完成而执行。

`tools/jev/context_replay_audit.py`逐项比较第二步的batch、采样组、四rank完整
master/Adam状态、BF16执行参数与含marker的joint策略，并要求每rank的恢复收据。
独立入口`verify_context_online_replay_v1.py`先复核两次收集的全部文件hash；
运行时间/cache hit等非数值字段不用于声称重放一致。47项相关CPUtests通过，之后
新增磁盘artifact集成测试，重放模块共8项通过（覆盖rank3 Adam偏差及恢复收据错误）。
证据`context-online-replay-preparation-v1.json`，不等同真实NPU重放验收。

## 已排队的验收链与选择集CPU输入

连续更新→独立收集→新进程恢复→独立收集→逐张量重放比较链已启动，
PID362663，状态`context-online-acceptance-v1-status.json`为
`launch_waiting_shared_device_lease`。当前尚无新NPU容器；正在等待旧selection的租约桥。
候选源码、launcher及collector由`context-online-acceptance-v1-pins.json`绑定；
队列执行期间不得修改这些文件。失败保留handle，不重启训练，SSH观察失败只重等同handle。
这条链只派发两步工程探针和恢复重放，不派发正式Stage2或研究曲线。

新增独立选择集CPU预构建`tools/jev/prepare_context_selection_cache.py`，
复用已验收cache格式/processor identity与只读词表加速，全9226输入按既有preflight
核对tokens、positions、音频receipt；worker仅接收id/audio/hash，无标签或模型推理。
容器77affe32d09d02433133e3c03077f25d1715b0647b487b00e29b5e18d1c3952e
14:20:48 UTC启动，2CPU、8GiB、无NPU，固定镜像与/data ext4热路径预检通过。
driver365727准确wait后独立收集，状态`context-selection-cache-v1-status.json`；
普通观察不早于remote启动+600秒。输入缓存输出到既有evaluation使用的
`/data/xingchao/multimodal-jev/simple-final-v1/context-selection-input-cache-v1`，
未改训练/评测实现，不冒称完成cache或已获得NPU吞吐收益。

到期remote epoch1794666079：原learning writer39168/40513、hits27839、仍live；
辅助writer已exit0/nonOOM，终态manifest40513条，独立核对receipt文件hash。
辅助终态只证明其记录完整，原writer最终发布及逐cache文件校验仍由原自动链承担。
证据`context-learning-cache-assist-v3-terminal.json`；下次普通learning观察>=1794666679。

收到原链进入collection后核验：主cache9822c31d已14:21:33 UTC exit0/nonOOM，
manifest40513、existing_hits29184，原collector PID366350仍live，已输出21477行
independent.jsonl，尚未最终逐文件验收。本次终态观察
`.local/decision-objective-research/context-learning-cache-v1-terminal-observation.json`；
后续以`docs/engineering/context-learning-cache-v1.json`实际发布为准，不重复派发cache。
新增完整RL曲线汇总独立工具见`context-selection-curve-v1.md`，不改已排队验收的pins。

## 全学习缓存独立验收与接续恢复

原collector完成40513条逐文件/张量核验并输出manifest后，Ascend退出阶段向stdout
追加`DrvMngGetConsoleLogLevel failed. (ret=4)`，导致本地尾部JSON解析失败。
原始independent.jsonl与失败driver日志完整保留；缓存容器exit0，未重建缓存。
新增`context_cache_receipts.py`只允许这一条精确尾日志，未知日志、内部日志、重复尾日志、
人口不全或非零terminal均拒绝，6CPUtests过；选择集collector也使用该解析器。
独立finalize读取原完整stream，重新核验全40513人口/请求身份，最终发布
`docs/engineering/context-learning-cache-v1.json`，manifest SHA4e2dfebb，
原始stream SHAe95689c8，尾日志另列driver_diagnostics，没有丢弃原始证据。

已确认原Stage1和selection驱动因该收集异常终止，且未创建full训练容器；
保留失败status后恢复两个本地驱动，PID373142与373145。前者已到
`launch_admitted_full_schedule`（step2→634）、后者`wait_full_training_chain`。
不重启任何缓存或训练容器，仍共享设备租约；冻结364项通过，连续验收队列pins不变。
旧selection最新remote1794666538仍running，HF进度9226/9226；不能仅凭该阶段完成
释放整个容器的租约。普通下次该任务>=1794667138。

选择集cache77affe32已14:29:26 UTC exit0/nonOOM，9226全部无命中冷构建475.94秒，
独立collector完成并发布`context-selection-cache-v1.json`（manifest893f3766）。
完整逐文件/张量/请求/人口核验通过，准确等待驱动正常收尾；这是CPU输入准备，
不是模型预测或NPU吞吐提升证据。至此learning40513/selection9226纯输入cache均通过。
Stage1全量和连续RL探针仍在原共享锁队列，未改已排队源码或参数。

到期remote epoch1794667171核验旧selection cd263613仍running/nonOOM：
HF9226/9226，vLLM1687/9226且progress当时更新，属于正常计算等待。
租约桥357467继续准确wait；连续验收362663、全量Stage1接续373142与后续评测373145
保持原队列，不强占、不重复启动。下一次普通该任务观察>=1794667771。
`context-full-input-readiness-v1.json`复核排队Stage1 step2→634确实绑定验收后的learning
cache，356训练文件未漂移；learning40513、selection9226、FSC3793均共享同processor身份。
这证明输入依赖就绪，不证明权重训练或任务收益完成。
