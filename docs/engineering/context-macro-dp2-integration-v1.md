# DP2 训练与完整 rollout 双实例接入

目标布局为175物理4/5卡FSDP数据并行训练，6/7卡各一个TP1推理实例。
本轮增加 `tools/reports/context_macro_trainer_candidate.py`，通过受校验的原训练器
生成新的有限工程入口。原DP4训练器、历史实验、日程和封存包保持不变。

新入口明确选择 `dual_credit_grpo_floor_clipped_v1`、joint排序/存在性目标；
global128，每rank64，micro8/16/32分别对应累积8/4/2。分片数量、collective接收
列表、每rank DataProto切分、checkpoint世界大小均从world/layout推导。
rank0通过`connect_pool`连接两个已启动实例，并复用DualContextRolloutSession。
服务启动器和实际两卡梯度/恢复验收尚未完成，不能作为正式训练发布。

原已认证SLURP日程继续负责完整40513学习人口、两轮634次更新、样本成员与顺序。
每批先按实际processor长度重新核对原DP4 rank_indices，再仅对该128条批内做
DP2长度分组；不改变样本、零权重padding或全局分母。批次收据记录实际DP2分组。
该入口暂未启用动态难度采样，以免在首次拓扑验收时同时改变学习分布。

恢复契约加入macro_layout及world_size=2；旧DP4 Adam分片不能直接装入DP2。
同一DP2拓扑的恢复仍需实际fresh-process验证。保持LoRA/head训练范围检查、
FP32优化器、BF16执行与导出、跨后端概率检查以及双副本策略确认。
保留最多两次更新的工程限制；不自动放开全量训练。

验证命令：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools/jev python -m pytest -q tests/test_context_macro_trainer_candidate.py tests/test_context_dual_trainer_candidate.py`
结果21 passed（2.14秒）。包含3档DP2分组、完整曝光/分母、旧分组错配拒绝、
拓扑/算法/runtime拒绝、生成代码与既有runtime连接测试。没有执行actor optimizer，
不构成实际DP2梯度一致性、NPU吞吐或恢复证据。

## 已启动的真实音频前置探针

先执行固定TRAIN128的单实例serial/batch4对照，验证实际logits、prob、G16与请求
顺序不变性，并记录语言attention及audio tower真实调用次数。两个timed pass，
另含warmup和反序pass；计时不包含actor、导出与策略同步，不能用作全训练提速。

首次远端预检失败原因：Python导入校验模块生成未列入清单的pyc，完整包校验拒绝。
远端独立清点显示仅新增`tools/reports/__pycache__/run_context_batch_probe_candidate.cpython-310.pyc`，
无丢失或改动清单文件；当时没有同名容器。保留首次目录及失败结论：
`/srv/multimodal-jev-xingchao/operations/dual-probe-grpo-pivot-v1/failure-conclusion.json`。
启动器两个远端python校验命令改为`python3 -B`，相关21项测试通过。

新版本重新封包1095文件，manifest SHA256
`3785aee7b52820b64aea9ddde85d87608be8d0a81a36775f727cc5e0494e9897`。
本地宏`.local/config/dual-probe-grpo-pivot-v2.json`，可复用模板
`configs/portable/operations/context-dual-probe-idle-window.json`，只开启有限推理。
仍复用原exclusive lease及空闲卡检查；未恢复旧RLOO或原后继任务。

实际容器 `d178640382da9c9b9c8ed9460198339c501a5b37cf7a453db0f75e44f4ac8760`，
固定镜像`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
2026-11-16 07:28:33 UTC开始，仅挂载davinci7，inspect确认running，未OOM。
本地driver1783421/start_ticks339083057等待准确容器终态；自动收集及独立审计。
证据根`/srv/multimodal-jev-xingchao/operations/dual-probe-grpo-pivot-v2`。
该v2已终态：07:31:53 UTC退出1，OOMKilled=false，serial模型实际加载完成后
测量`apply_model`函数RPC被vLLM拒绝序列化，尚未测量。完整失败输出已自动收集到
上述operations目录`context-dual-probe-npu-v1-collected`。审计正确拒绝非零退出。
准确wait session39658已消费退出1，driver1783421已终态，不再等待或重启v2。

修复为worker具名`jev_dual_measurement` RPC，仅传serial_wire/install/reset/read字符串；
不打开pickle回退。serial owner只替换worker注册，保持原单序列backbone及activation；
不安装segmented profile。measurement/probe相关18项CPU tests通过（2.25秒），
worker方法测试仅抽取实际方法体执行，本地没有vLLM，实际注册由NPU验收。

新v3已封包并启动，manifest
`cdcba45d18d8a1a035d6f346d0b2616fecb5435d75fb9ec7fbfe11cd5353e227`；
容器`b841b13a9242d0813092510e2c9ad8af88e9da28164b5cb29231e0330679c114`。
本地driver1784927/start_ticks339122509；宏`.local/config/dual-probe-grpo-pivot-v3.json`，
证据`/srv/multimodal-jev-xingchao/operations/dual-probe-grpo-pivot-v3`。
tag现从本地宏读取，旧容器/包保留。v3尚未终态；不把启动当作一致性或提速成功。

## DP2 实际 native verl CPU 更新与恢复

`tests/test_context_macro_actor_dp_resume.py`复用参数化的原四rank验证实现，明确
world2/global128、GRPO、micro8/16/32。两rank分别64与1个有效样本，共65个real、
63个零权重padding，检验不均衡分布下的实际DDP梯度平均和native verl累积。
每档连续更新两次，再用新解释器/进程组/模型/Adam读取第一步状态重放第二步，
逐张量比较采样、梯度、masters和Adam。另检验单rank概率饱和、分母篡改、
行为不一致时所有rank共同拒绝，优化器不被修改。

固定镜像CPU容器`d2ee78f76d4a24038992bea965aa8d295447b405859e9817ba15a7c12e5fc5f1`
已于2026-11-16 07:40:01 UTC退出0、非OOM、无NPU设备，1个完整集成测试通过，
零skip/error/failure；源码368文件。准确wait及独立收集均完成。
报告`docs/engineering/context-macro-dp2-fixed-image-cpu-v1.json`。
该测试使用合成FP32 logits，仍不证明BF16 FSDP、真实音频、双服务激活或NPU性能。

## 下一步双实例真实音频验收工具

`tools/reports/context_macro_measurement.py`：在两服务已启动后提交4个完整macro128
（warmup、两次timed、反序），逐条核对实际logits/prob/G16和完整收据；测量包含
IPC/journal/验证，不包括服务启动、actor、导出、同步、checkpoint。与先前单服务
内部测量的计时边界不同，因此不自动报告训练或双实例加速比。

`tools/reports/prepare_context_macro_probe.py`：根据宏模板
`configs/portable/operations/context-macro-probe-prepare.json`封装两个服务配置，
物理6/7独立TP1、独立socket/exchange、共享只读policy/adapter及runtime源码清单。
必须先绑定成功父探针audit、transport和serial measurement的hash；原始输入仍为
固定TRAIN第一批128条，无新抽样或目标集介入。没有父验收证据时拒绝封包。

measurement及DP2生成器测试17项通过（3.87秒），封包6项fixture测试通过（1.90秒）。
这些是本地CPU fixture；尚未真实生成双实例包、启动双服务或获得其NPU结果。
下一步先消费原v3准确容器wait，再封包并完善隔离容器启动/失败收尾/独立审计。

## 实际单实例验收结果与双实例派发

v3现已exit0并完成独立审计，原wait34580已消费终态。观察归档：
`docs/engineering/context-dual-probe-npu-v3-observed.json`。
同一TRAIN128，两次timed串行28.8857/29.7961秒，batch4为17.2073/17.2567秒，
平均29.3409→17.2320秒，耗时减少41.27%，serial/batched比1.7027。
实际语言前向128→64次，batch4容量不等于每次实际处理4条；调度器实际拆分。
worker峰值allocated约2.94→3.13GB，reserved约2.98→3.30GB（十进制）。
实际logits、概率和原Bernoulli G16在warmup、两timed及反序pass均完全一致。
该结论不包括actor/export/switch/checkpoint，不是整轮训练提速或dual-ranking收益。

基于该成功audit/transport，真实双实例包已生成：
`/srv/multimodal-jev-xingchao/operations/macro-probe-v1-prepared`，
manifest `4bd0ad0801f34f73856779a5c4502d7496627b80e7c2a602d78d44b19efdd721`。
宏`.local/config/macro-probe-v1.json`集中绑定路径，operations模板
`configs/portable/operations/context-macro-probe-operations.json`。
新增`context_macro_probe_operations.py`持exclusive lease，核对6/7空闲、固定镜像、
只读源包和独立设备挂载，先记录create intent与准确ID，再启动两个服务。
固定脚本每5秒等待startup ready（非模型普通训练监听），最多900秒；随后启动无NPU
client完成4批macro128。失败只收尾本次创建且scope检查通过的容器，不重试、不删除。
服务关闭后分别记录inspect/log/terminal；独立collect/audit另行运行。

实际service0/卡6容器
`b61fc3423db4f9cdb7c596fb5dcf299b396eef75aff53af97c0cecb5621dcf65`，
service1/卡7容器
`4e9fe22fdd0a6ed3f80692ac3515b5ea86ccff6bb86d7b2c5ee4ab37301da889`，
07:48:09/10 UTC启动，inspect均running非OOM。
driver1787953/start339200668，根
`/srv/multimodal-jev-xingchao/operations/macro-probe-v1`，远端`/data/xingchao/jev-macro-v1`。
当前准确pidfd wait session93086，尚未完成；不能重复派发或把启动当成功。

新增`audit_context_macro_probe.py`，只收集measurement/journal/service结果，
不收集基座模型或cache。重核3容器命令/挂载/设备/终态、传输hash、输入张量身份、
完整job规划及归属、每rollout内容hash、宏COMPLETE与最终全量结果、原始G16采样和
计时聚合。运行方式为`python -m tools.reports.audit_context_macro_probe --config
configs/portable/operations/context-macro-probe-operations.json --local .local/config/macro-probe-v1.json`。
仅成功driver可收集审计；已有collection拒绝覆盖。失败结果保留在远端与operations logs。
operations/prepare18 tests通过；独立audit/operations20 tests通过（14.40秒）；
diff-check/py_compile通过。这些fixture不替代当前双实例真实验收。

## 效率对比预算与最新验收状态

用户明确：效率对比不必跑完整数据集，最多100个训练step即可停止。
`configs/experiments/grpo-transfer-macro-batch-v1.json`已记录每个比较配置100次全局
optimizer更新上限，step含双副本激活确认；microbatch和rollout子步骤不计为独立step。
同一对比使用全量SLURP TRAIN固定日程的相同前缀，不另外挑小数据集。
冷reference填充、热缓存、启动、评测及导出/同步成本分开报告；不为测效率要求完整epoch。
短工程验收仍保持4推理pass/2训练更新。此上限仅用于效率对比，效果与迁移实验独立定义。
当前只是配置预算，100-step执行器尚未发布，不应声称已启动100步训练。

双实例macro-v1已终态失败：client导入缺少封包中的`multimodal_jev.local_config`，
exit1，没有rollout。两服务ready后由启动器按准确ID收尾，exit137、OOMfalse；不是训练
或推理成功，也不是OOM/core dump。原pidfd wait93086已消费，失败审计正确拒绝。
日志和inspect保留于macro-probe-v1 operations目录，未删除历史包。

封包显式加入local_config.py，operations增加无NPU preflight容器：先在固定镜像导入
实际client/measurement及核验两个runtime/source/processor，成功才创建NPU服务。
这样避免仅CPU fixture通过而遗漏完整实际部署依赖。prepare/operations19tests通过。
新macro-v2 manifest `f307bbcc3900a288fb54630f7ac1edbdbda390b76356df6db869a31e590a741e`，
本地宏`.local/config/macro-probe-v2.json`，prepared和operations分别位于
`/srv/multimodal-jev-xingchao/operations/macro-probe-v2-prepared`及`macro-probe-v2`，
远端`/data/xingchao/jev-macro-v2`。新attempt正在执行，旧v1不可重启。

此外已编写`prepare_context_macro_training.py`及宏模板，等待成功双实例审计后才允许
封装两步GRPO DP2包，保持完整日程、Stage1 policy、只读基座和独立reference cache。
该新封包工具尚未真实执行/固定镜像验收，DP2训练启动及恢复仍未完成。

## 双实例验收通过与两步 DP2 派发

macro-v2已正常完成，准确driver wait57656已消费终态。独立收集与审计通过，
归档`docs/engineering/context-macro-probe-v2-observed.json`。
固定128条TRAIN完整macro，两次timed为8.793255/8.742091秒，平均8.767673秒，
14.5991 query/s。四次pass（warmup/timed/timed/reversed）均完整覆盖128条，
每批四个rollout的实例归属为[0,1,1,0]，没有query或候选/G16组跨实例拆分。
actual logits/prob/G16与已审计serial参照完全一致。该计时包含IPC、journal、
验证和整批交付；之前serial/batch4计时有profiling hooks但没有同样IPC边界，
不能直接据此宣称训练加速比，也尚未覆盖policy更新/恢复。
三个实际工作容器及preflight全部正常exit0，镜像/设备/挂载/输出hash独立复核通过。

已从成功macro包及冻结训练日程生成真实DP2训练包：
`/srv/multimodal-jev-xingchao/operations/macro-training-v1-prepared`，manifest
`f0f8603f57245f67c2cb743d1e82b0c0226960ffe0c7dc95e6c59eeb38c22921`。
真实prepare重新审计macro内容，保留全量40513 TRAIN/634日程以及Stage1起点，
仅前两次update，GRPO双信用joint/global128/micro8/DP2。基座未复制。
训练源码与双推理源码共同进入runtime manifest；reference cache另建以避免跨runtime复用。

新增`context_macro_training_operations.py`复用同一服务生命周期，显式注入训练
validate/command，空闲核验覆盖4–7。actor仅挂davinci4/5、torchrun nproc2；
服务各见6或7。共享数据只读、编码cache按既有接口写、音频mirror只读；全部pins
必须落在显式挂载范围内。工程启动器仍只接受两步，不扩大到效率100步或全量训练。
训练/推理operations共19tests通过（2.12秒），曾发现Docker资源参数采用等号形式，
测试暴露后修正命令构建，再通过。实际封包validate及diff-check/py_compile通过。

宏`.local/config/macro-training-v1.json`，模板
`configs/portable/operations/context-macro-training-operations.json`。
driver1790468/start339308101，证据根
`/srv/multimodal-jev-xingchao/operations/macro-training-v1`，远端
`/data/xingchao/jev-macro-train-v1`。准确pidfd wait工具session56246等待原driver。
无NPU preflight容器`2ce0deb18d0a501f723f0075846f836b72bfa935be68099df74a393ae82a54f0`
已exit0，校验实际client/actor/session imports、完整pins、runtime/processor及native adapter。
service0 `9a24d6ade9c38b1a4a74ae0ccbca716abe2ce033a48e040f168d5cc250c9623f`，
service1 `d40cf27d8cd8808b448e2fa01ba541fc6a7025d10366c70ff3f6204931e1adab`已启动。
等待就绪后才启动actor；尚无训练完成、双policy激活、恢复或效能收益结论。
下一步消费该准确driver，并补DP2完整训练collection/audit及fresh-process replay；
旧DP4 audit硬编码4rank，不能直接拿其通过与否当DP2证明。

## 2026-11-16：DP2 rendezvous 故障与修复版预检

macro-training-v1 已失败终态，准确 wait56246 已消费。actor
`f118f0159b07acf540057a77ad8bac6ea14d3658b402c912880e7e455c17cc34`
实际 exited1、OOMfalse，日志为 `torch.distributed.DistNetworkError`。
`torchrun --standalone` 的动态 rendezvous 在 network=none 下使用容器主机名
f118f0159b07；实际 /etc/hosts 只有 localhost，无该主机名，TCPStore DNS连接
300000ms超时重试。未进入actor训练，不能把此耗时算作训练step。发现后按实际ID
请求停止；actor最终自身exit1，两服务由原driver清理exit137、OOMfalse。
失败日志、停止意图及四容器inspect均保存在原operations目录，不改旧包。

`context_macro_training_operations.py` 改为单机静态 `--master-addr=127.0.0.1`
及 `--master-port=29500`，去掉standalone，保留network=none、隔离NPU与原镜像。
无NPU preflight新增相同参数的torchrun双进程Gloo all_reduce（1+2=3），
进程组30s超时、外部90s上限，通过后才创建NPU服务。此测试证明通信入口，
不证明HCCL/NPU训练。命令构造测试同时解析真实嵌套probe源码，避免字符串转义错误。

新宏 `.local/config/macro-training-v2.json`，prepared/operations分别为
`/srv/multimodal-jev-xingchao/operations/macro-training-v2-prepared`、`macro-training-v2`，
远端 `/data/xingchao/jev-macro-train-v2`。manifest
`740574a632c1fa01f22bd2edac9871b5e93bb8b1a48097b8e75326ed1d537377`。
driver1794166/start339404740，准确pidfd wait15884；preflight
`296fbf9615f0704cd60dd3b3a1e480ce5d96def5159d19dc94fbda00b236ee6d`已创建。
仍为两步GRPO joint/global128/micro8；不启动100step效率对比。

补充独立审计工具 `audit_context_macro_training.py`，沿用同一operations宏，
`--action collect`负责准确终态四容器、远程源码/pins复核、仅run/journal/service
产物传输；`--action audit`可离线核验已收集bytes/inspect，避免重复归档。
检查DP2原日程内重分组、独立GRPO两信用重算、FP32 Adam步数与分片、BF16执行与
native adapter一致、两服务激活确认、每个完整rollout归属及实际reference/behavior
在journal中的覆盖。主模型不收集。master-export及checkpoint verifier显式支持
world_size=2，默认保持4；旧冻结包不变。

实际音频mask有padding，不能用特征宽度当有效长度。journal audit根据mask
SHA256匹配二值prefix mask有效长度再复算调度；不匹配则拒绝。44相关CPU测试通过，
并用已独立通过的真实macro-v2 warmup128记录复核journal通过。这不是DP2训练通过。
后续仍需实际训练产物审计及fresh-process恢复，再进入每组最多100step效率比较。

修复版最新preflight实际exit0，双进程Gloo rendezvous/all_reduce通过。两服务已创建：
service0 `4abef23342e5af6ee2087c0071ce2a1ba965ac7ec27e71078347cc980bfc25b0`，
service1 `acbdefb657750452d3878ec505738f085294bf7b6730983c405d7bbe4fa6a814`。
原driver/准确wait15884保持，未重派；尚无NPU actor更新结论。
