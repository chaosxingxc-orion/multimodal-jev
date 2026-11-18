# 当前全量训练吞吐分析

分析对象：175物理4–7，Qwen3-ASR-0.6B，BF16 LoRA r4，DP4/FSDP，
global128、micro8×accum4，class-spread完整40513录音×2轮。当前instruction CE训练，
不是包含rollout的RL吞吐。不更改正在运行的配置。

## 实测

读取四rank共同闭合的update341–404，共64步，其中63步无checkpoint。
原始日志和长度样本hash见`slurp-current-throughput-breakdown-v1.json`。

| 阶段 | 普通step均值（四rank取最大值） | 解释 |
| --- | ---: | --- |
| 准备 | 6.553秒 | rank0逐条音频hash/解码/processor/模板和身份构建，随后object scatter；含其他rank等待 |
| actor更新 | 15.713秒 | 前向、反向、FSDP通信、optimizer和内部host工作，尚无算子级拆分 |
| 审核 | 0.092秒 | master/Adam有限性及步数检查 |
| 完整step | 22.337秒 | 约5.73录音/秒 |

各阶段分别取rank最大值，并非同rank时间线，不要求数值精确相加。包含checkpoint的
64步平均22.357秒；step384保存0.744秒、审核0.548秒、总23.605秒。保存不是主瓶颈，
没有依据删除fsync或削减持久化保证。峰值allocated29.88GiB、reserved35.28GiB是训练
allocator统计；单次npu-smi总占用约36.8–39.3GiB，与allocator口径不同。

一次瞬时npu-smi中4–7的AICore为0，docker CPU约581.8%（约5.8核），仅说明捕捉到
空闲阶段，不能外推整段平均利用率或CPU饱和。正式算子归因需短时profile。

同64批8192条长度1952–2249，均值1982.49 token。按实际rank/micro排列复算，batch
padding占0.1924%，固定128行投影tile额外位置合计3.3992%。现有长度组批已有效，继续
细分bucket不是首要收益来源。LoRA减少可训练参数，却不消除约2k token的backbone
前向和为语言LoRA传播的反向，因此不能按“仅r4参数”估算训练工作量。

## 优化顺序

1. **先解耦并重叠输入准备。** 当前每步rank0同步准备128条，其他rank等待；每epoch
   同音频重复解码/processor。优先固定原始音频和processor版本，将纯输入张量缓存到
   本地/data，以有界CPU worker预取下个batch。策略ID、采样/标签权重仍在消费时绑定；
   不能把旧policy标记一同缓存后用于RL。保持原波形/特征/位置hash和学习日程，严格
   验证失败传播、尾batch、恢复游标，不能删除长音频或改变实际batch。
   若能完全隐藏6.55秒准备时间，当前窗口理论上限约1.42倍（22.34→15.78秒），不是
   实测收益或保证；磁盘/CPU/通信干扰会降低它。
2. **再profile模型更新。** 已接受profile对420个Linear挂固定128×8 tile投影，内有
   Python循环和多个bmm；FSDP也按线性模块包装。两者可能产生小算子和通信开销，
   现有阶段计时不足以判定占比。用独立固定batch、相同checkpoint做CPU/NPU活动、
   bmm调用数、all-gather/reduce-scatter和同步时间profile，再只优化占比最大的环节。
   替换投影/包装粒度需重新验证同策略完整概率、梯度、恢复和vLLM一致性。
3. **microbatch按有效吞吐选，不按显存空余猜。** 历史DP2容量试验micro16约52GiB，
   micro32在首次前向即出现device OOM（见slurp-true-batch-dp2-capacity-v1.json）；
   该试验不是当前DP4稳定窗口，micro16尚未证明稳定快于micro8。保留global128时，
   micro16×accum2值得配对基准，但须覆盖长音频、实际更新及相同参数训推概率；不能
   直接外推当前micro8的29.88GiB峰值至micro32安全。global256会改变优化日程，应
   独立登记，增加global batch本身不会提高每张卡的并行计算。
4. **资源扩展后置。** 0–3当前有其他Omni服务，不抢占；0.6B LoRA是否值得FULL_SHARD
   而非复制基座/同步LoRA梯度，也需实测。8卡收益取决于输入管线和通信，不能保证线性。

当前全量轨迹保持不变，优化候选在独立入口验证；checkpoint收集、完整selection和
概率质量评测照常衔接。性能优化不代替RL任务收益及跨数据集泛化验证。

## 输入缓存与预取实施进度

新增`encoded_audio_cache.py`按音频内容、完整request（含version）、processor/源码
身份与preprocessing配置绑定CPU输入张量；拒绝labels/额外字段，保存重载校验逐张量
hash，原子不可覆盖发布并保留fsync。策略ID、目标、权重和采样结果不入缓存，消费时
独立绑定。调用方须在运行准入时验证源文件身份；缓存键本身不证明源文件未变。

`ordered_prefetch.py`提供有界CPU线程预取、原序消费和异常传播；只对纯输入回调使用，
worker不得调用NPU/分布式collective或绑定变化中的策略。关闭等待已执行回调结束，
不能强制取消已执行的Python回调，因此I/O需有限等待。5项CPU测试覆盖精确重载、
隔离修改、版本miss、冲突/损坏拒绝、顺序/有界读取和错误/恢复游标。

真实CPU探针`probe_encoded_audio_cache.py`在固定镜像检查中位/最长音频、原Final60
输入。每条对照新鲜processor结果、三次缓存读取及policy v1/v2绑定；不训练。容器
`0f1322850f054ddd87cc4ab977753540e94243ba358b1ddec0e816c0ed11fd11`已启动，
本地启动记录`.local/decision-objective-research/encoded-audio-cache-cpu-v1-launch.json`。
已收集exit0/nonOOM终态，331项远程版本绑定复核通过；本地独立复算缓存张量、位置和
策略身份重绑定，结果见`encoded-audio-cache-cpu-v1.json`。中位音频冷准备86.1ms，
热读取4.9–6.1ms；最长音频冷准备57.1ms，热读取6.5–6.8ms。两条音频的冷启动时间
不可用于推断长度与耗时关系，也不能把单条热读取倍率当成训练倍率。

新增`cached_span_preparation.py`封装线程独立processor、源音频hash验证以及消费时
绑定当前策略和原监督。连同缓存/预取的8项CPU测试通过，包括缓存命中跳过解码、
源文件变更拒绝与并发发布一致性；当前训练循环未接入。仍需全batch/全人口成本、
恢复游标和实际NPU吞吐对照，不能从这些组件测试宣称已实现1.42倍加速。

## 当前判断与验收边界

原始64步文件hash和四rank阶段最大值均值已独立复算。当前瓶颈证据支持优先改善
同步输入准备；模型更新内部的固定形状投影及细粒度FSDP仅为待profile的嫌疑项，
不能直接称为已确认的通信或算子瓶颈。全量两轮634步按当前速度约3.94小时，
不含启动、重载及完整selection评测；隐藏全部准备的理想下限约2.78小时。

下一候选对照保持global128、同一checkpoint、同一日程、BF16与相同输入：先测
缓存/预取，再测micro8×4与micro16×2。报告热身后稳定窗口的录音/秒、step耗时、
输入等待、峰值显存和完整验证墙钟时间。micro32已有历史前向device OOM记录；增加global batch
并不直接减少每条样本的计算。模型内核/FSDP变化须另做训推概率、梯度及恢复验证。

## 完整batch实测与候选入口

`cached-span-batches-cpu-v2.json`记录固定镜像CPU容器aa122be8成功终态，346项版本
绑定复核通过。固定学习日程batch0与316，每批128槽位，真实录音数分别128和65；
本地逐槽独立重建输入绑定并核对缓存张量、位置和receipt，256槽位全部一致。
缓存193个唯一输入，共40,475,673字节。未使用selection或TEST。

| CPU输入服务 | 普通批次 | 尾批（含63补齐槽位） |
| --- | ---: | ---: |
| 原逐条解码/processor | 4.969秒 | 5.083秒 |
| 热缓存串行（两次） | 1.076 / 1.101秒 | 1.106 / 1.091秒 |
| 热缓存双线程逐条预取（两次） | 3.307 / 3.259秒 | 3.442 / 3.489秒 |

该容器限制2个CPU，torch单线程；以上不含NPU更新、分发或模型并发干扰，不能直接
替换生产6.55秒准备项。双线程逐条方案反而慢，因此候选改为一个后台线程串行
准备整批，最多预取下一整批，与前台当前批训练重叠。首次缓存构建成本也须计入
全量训练墙钟收益。v1因缺少发布审核文件挂载在准入阶段失败，已归档，v2补齐原
路径只读依赖，没有削弱数据身份检查。

`cached_instruction_batches.py`固定日程游标、输入编译和策略/监督消费边界；
11项相关CPU测试覆盖尾批、非零恢复游标、异步重叠、异常不跳步。独立候选入口
`train_slurp_cached_instruction_candidate.py`从原入口SHA256
`d1b6d63f577374dcf74f9b0852a02dde1e304eb696a69ee0b1d58a05e1028924`派生，仅替换
输入准备及清理，保留actor、loss、checkpoint和恢复流程；限制两更新验收。
原入口与RC2原1360文件hash均未变；候选尚未NPU派发。真实CPU组合路径另以v3
探针验证，成功也不能替代NPU更新、恢复、HF/vLLM概率和稳定吞吐验收。

真实组合路径v3容器3b28efe9已exit0/nonOOM，347项pins复核、256槽位缓存独立重算
通过，真实任务的输入、候选、监督、权重及补齐身份保持一致。整批后台服务耗时分别
2.547 / 2.616秒；包含单线程processor建立及消费编译，不是NPU重叠收益。
证据`cached-span-batches-cpu-v3.json`。最近到期训练观察四rank完成474、执行475/634，
仍running/nonOOM。

## 后段耗时复查

最新534–565共32步平均准备13.24秒、更新15.67秒、总28.98秒，准备中位7.22秒；
536–540附近部分准备达到31–33秒。最近558–565的准备6.47–7.00秒、总22.28–23.30秒，
已恢复接近原水平。证据`slurp-late-throughput-breakdown-v1.json`与其绑定原始四rank
日志。模型更新未明显变慢，证据只支持输入阶段暂时停顿，不能直接归因到某CPU探针
或Ceph。全量新协议CPU预检4worker正在运行，普通观察仍至少600秒。


## 用户再次询问后的现场复查（完成579步）

本次用户请求触发一次即时读取，之后普通观察仍至少600秒。新证据
`slurp-throughput-user-review-20261114.json`绑定完整四rank日志与资源快照。
运行容器仍running/nonOOM，完成579/634，progress中的step579为零基索引，执行第580次更新。

| 窗口 | 准备秒/步 | actor更新秒/步 | 总秒/步 | 128槽位/总秒 |
| --- | ---: | ---: | ---: | ---: |
| 早期341–404普通步 | 6.55 | 15.71 | 22.34 | 5.73 |
| 最新516–579普通步（63步） | 16.77 | 15.70 | 32.53 | 3.94 |
| 最新572–579（含一次checkpoint） | 22.33 | 15.68 | 38.22 | 3.35 |

最近已完成窗口相比早期吞吐下降约31%；最近8步下降约42%。完整step各rank取最大值，
子阶段也分别取最大值，不能要求相加严格相等。以上均为Stage1 instruction训练，非RL rollout吞吐。
采集时未完成第580步的actor进度标记已约82秒未更新，未计入上述窗口；不能依据已完成步
的稳定actor均值排除正在发生的新计算/通信停顿，也不能单凭进度文件诊断死锁。

源码确认主rank逐条完成hash、音频解码、processor、模板/候选编译及元数据写入后，
通过object collective分发；准备计时合并这些操作及等待。已确认结构上的串行输入瓶颈，
尚不能将新增耗时细分归因于Ceph、CPU配额、processor或scatter。资源单次采样显示训练
约566% CPU、预检约94% CPU，四训练卡AICore瞬时0；不能当持续利用率，也不能证明
预检抢占是根因。采集环境的pressure/cgroup统计不能替代逐容器配额及NPU profiler。

效率优化顺序继续为：本地纯输入缓存+整批ahead1预取 → 细分输入和actor profile →
相同global128下micro8×4对micro16×2 → 根据profile决定投影融合和FSDP粒度/复制基座。
原固定投影每个micro8、约2k token的线性层约需16次小bmm调用；这是源码形状估算，
不是已实测的调用总数或耗时归因。不能直接撤掉训推一致性路径。
历史DP2固定64条实验micro8长度组批4.78条/秒、micro16为4.86条/秒，差异很小且仅两次；
micro16峰值约52GiB、micro32已device OOM。因此没有证据支持仅增大micro能解决当前下降。

下一次性能候选以固定checkpoint/数据/输入/全局batch比较，报告冷构建与热运行两种墙钟，
热身后的稳定窗口、真实录音吞吐、输入各阶段、计算/通信、显存；通过更新和训推一致性验收
后再提升为统一基座。缓存代码及真实CPU整批验证已完成，NPU训练加速仍未验证。
现有完整训练只剩55次更新，优先完成并取得完整selection结果，避免重启已完成轨迹。


后续到期核验：四rank已完成594步，原第580步已完成，不构成确认死锁。563–594普通步
均值准备21.67秒、actor15.71秒、总37.46秒。最新独立原始副本
`.local/simple-final-v1/throughput-updates563-594.json`；对应
`slurp-late-throughput-breakdown-v1.json`已更新，不能用该滚动receipt复核先前534–565窗口。
先前窗口原始数据仍包含在用户复查516–579的独立快照中。


缓存候选增加细分计时：后台worker服务/命中数、前台等待、绑定监督、组批、元数据写入、
broadcast与scatter单独报告。后台服务可能与前一步actor重叠，不能与前台等待相加。
本地及固定镜像1bc794d2的11项缓存/顺序/恢复游标/尾批测试通过，证据
`cached-input-timing-cpu-v1.json`。四卡候选已重新prepare-only，未NPU派发，不修改
原训练入口和原RC2。两步prefix只验收正确性，不足以报告稳定吞吐收益。

本地收尾流程`continue_class_spread_full_evaluation_v1.py`已启动，PID见对应launch。
锁防重复运行，准确docker wait等待训练终态，成功后顺序bulk收集→CPU checkpoint审核→
256条parity→独立parity审核→9226 selection→独立finalize。失败/传输异常停止并保留
原handle，不重启训练、不抢占卡。普通观察仍至少600秒。新probe不得与该收尾流程争用卡。


最终训练已exit0/nonOOM，四rank完成634步（本次执行5–634，共630步）。逐一认证2520个
rank/step文件后，全程普通619步准备均值9.05秒、actor15.62秒、总24.73秒；容器墙钟
15731.49秒，即约4小时22分。末64步总均值38.98秒。加权指令loss前64步3.053、
末64步0.548，样本混合不同且这是训练损失，不是selection或RL效果。证据
`slurp-class-spread-full-terminal-timing-v1.json`。自动收尾已启动CPU独立checkpoint审核
98629d12，完整selection仍未产生，不提前宣称指标提升。
