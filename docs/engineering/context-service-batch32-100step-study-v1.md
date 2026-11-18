# DP2 + 双 TP1：服务 batch4 与 batch32 的100步对照

## 2026-11-16 15:48UTC：完整100步审核通过

study-v3终态exit0，status=service_batch32_100_steps_compared。两组均100步、
12800真实曝光；固定比较器确认逐步样本记录和reference misses相同，两个完整运行
独立审核通过。两组reference均12800 miss；除声明的service batch/token容量及
隔离路径外，训练与推理配置一致。下表取完整更新墙钟，不含启动、初始化和收集。

| 指标 | batch4 | batch32 |
|---|---:|---:|
| 完整100步累计秒 | 8644.0089 | 8416.8090 |
| 累计分钟 | 144.0668 | 140.2802 |
| 平均每步秒 | 86.4401 | 84.1681 |
| 真实query/s | 1.480794 | 1.520766 |
| actor平均秒 | 40.4497 | 40.5971 |
| rollout平均秒 | 35.9363 | 33.4117 |
| reference平均秒（rollout内） | 16.1529 | 14.9573 |
| behavior+credit平均秒（rollout内） | 17.8837 | 16.5111 |
| 导出/checkpoint/同步残差平均秒 | 4.9799 | 5.0945 |
| actor峰值allocated bytes | 40302224896 | 40302224896 |

完整吞吐提升2.6994%，节约227.1999秒（3.7867分钟）；去前2步窗口吞吐提升
2.6680%。收益主要在rollout，actor未改善。这是一次匹配运行对照，未作重复运行
置信区间，也不证明任务效果或迁移收益。不能用之前serial→batch诊断2.142倍
替代此batch4→32训练实测。

证据：`/srv/multimodal-jev-xingchao/operations/service-batch32-study-v3/comparison.json`
及同目录`terminal.json`。原尾批后继已接续；用户优先的acoustic及bucket100队列
继续等待tail和parity成功，不直接启动正式634步。

## 2026-11-16 15:12UTC：阶段收益归因与统计边界

本地核验study及后继驱动准确进程身份和依赖hash均未变；完整100步comparison仍未
产生。沿用15:02的77/100进度及前55步同窗口报告，不增加远端查询。

前55步中，rollout每步减少2.5141秒，actor变化仅+0.0008秒；导出/同步残差增加
0.1623秒，最终每步净减少2.3938秒。新配置actor占48.26%、rollout占39.65%、
其余准备和导出/同步残差分别约6.04%。当前证据支持小幅吞吐收益；两组均DP2，
本轮没有测量增加训练卡数的收益。

核对批处理路径：macro128按音频/文本长度估计成本，拆成四个rollout32，完整
rollout分配给一个实例，双实例并发，全部返回后训练消费。服务按长度排序并受
query/token容量约束组batch。声学塔的segmented_audio_forward仍对每条录音调用
原始forward，保持录音边界和训推数值路径；提高服务batch没有将音频编码一起
批量化。相关源码已与实际封包逐字节匹配。固定策略128输入诊断仍有48次语言
模型调用，encode批量与worker实际forward数不同；这不能替代100步调用分布。

后续大块优化优先验证声学编码批处理/复用、冻结reference跨epoch缓存、actor
长度分桶的显存和吞吐收益。每项须保持输入与policy身份并验证训推一致性。
本次不改冻结配置；micro16已有设备OOM证据。上述候选没有已测提速结果。
本轮仅batch4→32受控比较，尚不能回答DP3+单实例是否优于DP2+双实例。

## 2026-11-16 15:02:17UTC：77/100

固定observe31224终态0：双rank77更新、77提交，client/双service running/nonOOM。
末步rank0总85.6135秒、actor40.8582秒、prep39.8231秒，双rank下一policy一致。
完整100步统计仍待，不用此末步替换前55步同窗口对比。下一普通观察
1794842237.2473216，即15:17:17UTC；未更改运行配置或提前选胜。

## 2026-11-16 14:46:34UTC：65/100

固定observe16604终态0确认双rank65更新、65提交，client与双service仍running/nonOOM。
末步rank0总84.1307秒、actor40.3598秒、prep38.8650秒，双rank下一policy一致。
非100步均值，不更新前55步阶段比较结论。下一普通观察1794841294.163，
即15:01:34UTC；训练配置及100步自动收集/审核比较链保持不变。

## 2026-11-16 14:37 UTC：用户请求的相同前55步阶段对比

新增固定只读脚本 `tools/reports/compare_context_service_batch_prefix.py` 和 portable
`context-service-batch-prefix.json`，本地宏 `service-batch-prefix55-v1.json`。
用户主动请求分析触发一次远端已提交文件读取，不新增轮询。实际执行80207退出0，
读取两组相同前55步，逐步匹配样本records与reference misses，检查双rank计时、
commit/activation策略身份及双服务激活收据。只统计已提交计时，不替代完整训练/
容器transport/梯度正确性审核；明确禁止据此选胜或准入正式训练。

| 指标 | batch4前55步 | batch32前55步 |
|---|---:|---:|
| 真实曝光数 | 7040 | 7040 |
| 累计步骤耗时 | 4739.5748秒 | 4607.9180秒 |
| 平均每步 | 86.1741秒 | 83.7803秒 |
| query/s | 1.48537 | 1.52780 |
| actor平均 | 40.4347秒 | 40.4355秒 |
| rollout平均 | 35.7364秒 | 33.2223秒 |
| reference平均（属于rollout） | 16.1148秒 | 14.8902秒 |
| behavior+credit平均（属于rollout） | 17.8012秒 | 16.3997秒 |
| 其他准备平均 | 5.1024秒 | 5.0598秒 |
| 导出/保存/同步残差平均 | 4.8995秒 | 5.0618秒 |

阶段吞吐提升2.8572%，耗时减少2.7778%，55步节约131.6568秒；rollout耗时减少
约7.035%，actor基本相同。容量32确有阶段收益，但没有大幅端到端提速的证据。
两组每步完整时间取两个rank较大值，分项用rank0；微小rank等待差异不重复归因。
100步新结果仍待，旧完整100步144.0668分钟与上述55步不是同窗口比较。
证据 `/srv/multimodal-jev-xingchao/operations/service-batch-prefix55-v1/report.json`，
同目录保存两份逐步快照/文件hash和配置。7项匹配/错样本/缓存/缺步/计时/激活测试
通过（91710，1.93秒）。study/tail/parity三个驱动仍存在且命令hash匹配，73/81/95
pins全部未改；原100步收集比较链不变，未增加定时观察任务。

## 2026-11-16 14:31:24 UTC：55/100

固定observe40937终态0：双rank55更新、55提交，client/双service正常nonOOM；
末步rank0总84.0046秒、actor40.3340秒、preparation38.8506秒，下一策略一致。
next普通观察1794840384.3860202（14:46:24UTC）。完整100步统计仍待，不按单步选胜。

## 2026-11-16 14:16:06 UTC：44/100，继续原100步窗口

固定observe35897终态0：双rank44更新、44提交，client及双service正常/nonOOM。
第44步rank0总86.7269秒、actor42.7768秒、preparation39.0599秒；双rank下一policy
一致，完整100步比较仍待。下一普通远程观察1794839466.5055854（14:31:06UTC），
未提前选胜、改配置或重派。

## 2026-11-16 14:00:52 UTC：33/100，运行正常

固定observe（session63942终态0）核验双rank33更新、33提交，准确client及双service
running/nonOOM。末步rank0总86.0987秒、actor42.4581秒、preparation38.5859秒；
双rank下一policy相同。该记录是末步观察，未给出新100步总耗时/提速结论。
next普通远程观察1794838552.8294005（14:15:52UTC），保持900秒间隔。

## 2026-11-16 13:44:51 UTC：22/100，完整差异仍待

固定观察记录`service-batch32-v1/observations/1794836691735756540-result.json`确认
双rank均22步、22个已提交checkpoint，训练和两服务running/nonOOM。
rank0第22步总85.7268秒、actor41.1960秒、preparation39.3758秒；rank1总85.7481秒。
这些是末步计时，不是22步平均；当前没有batch32完整100步总时长或匹配窗口提速结论。
下一普通远程观察不得早于13:59:51UTC，沿用900秒固定脚本。

| 比较项 | 旧服务batch4 | 新服务batch32 |
|---|---|---|
| 拓扑 | 175卡4/5 DP2训练、6/7双TP1 | 相同 |
| 有效global batch | 每DP副本64×2=128 | 相同 |
| actor microbatch / 累积 | 8 / 8 | 相同 |
| 服务请求/token上限 | 4 / 16384 | 32 / 131072 |
| 完整100步耗时 | 8644.0089秒，即144.0668分钟 | 待完成及独立审核 |
| 完整100步吞吐 | 1.480794真实query/s | 待完成及独立审核 |

旧基线训练更新占46.795%，rollout占41.574%（reference、behavior+credit、其他rollout），
其余为输入准备和导出/同步等。提高服务batch直接作用于rollout，不能消除actor开销。
容量诊断的2.142倍是逐条对批量，不是batch4对32，更不是整步提速；128请求实际仍有
48次语言模型调用，因此配置上限不能等同实际占用。完整结果由原study自动收集比较，
同时报告全100步与去掉前2步的98步窗口；不提前切配置或另起相同实验。

本地再次核验study/tail/parity三个驱动准确身份均live，73/81/95pins未变。
后继精度面板队列已等待tail成功，仅在100步完成释放设备后执行；不参与本轮耗时。

## 2026-11-16 13:29 UTC：定时观察11/100

固定`observe_context_macro_training.py`读取准确句柄，双rank均11更新、11提交，
client/双service running/nonOOM；第11步rank0总83.7970秒，actor39.8209秒，
preparation38.9349秒。只是单步观察，不能代替完整100步匹配样本/缓存对照，
不据此提前选胜或停止实验。next普通观察1794836654.471367（13:44:14UTC）。
证据service-batch32-v1/observations，exec28099终态成功，未修改原训练。

## 2026-11-16 13:14 UTC：容量通过，100步训练已启动

原study v3驱动已产生capacity32-audit.json：128输入固定策略，逐条两轮31.2584/
31.1905秒，批量两轮14.6039/14.5458秒，均值31.2245对14.5749秒（2.142倍）。
实际logits/prob/G16完全一致；语言模型调用每轮128降至48。worker峰值allocated
批量3370391552 bytes，reserved6094323712 bytes；这是受限输入诊断，非全量
训练峰值/双奖励GRPO验收，尤其不能把2.142倍当作batch4对32的提速。

原驱动随后自动启动训练：preflight终态0，client于13:12:19UTC启动；13:14:06UTC
准确docker inspect确认client和两个service均running、nonOOM。独立本地证据
`operations/service-batch32-v1/launch-observation.json`；client句柄
`f70eaedf9bffc29b9456d6c3c8769ef0afc76d368932ed5c206b032ae7075808`。
该观察未读取实际step数，无100步新成绩；普通下一次远程观察>=13:29:06UTC。
保持原驱动收尾/审核/比较，不重复launch或collect。

## 2026-11-16 13:09 UTC 核验

本次“两个训练节点”按175同机两张训练卡、两个DP进程执行；另两张卡各驻留
一个TP1推理实例，不是175/181跨机训练。study v3驱动的PID/start_ticks/cmdline
身份核验通过，73个锁定文件hash一致；旧100步baseline-audit已产出。容量probe
已有准确launch和start收据，尚无capacity32-audit或study终态；100步新训练的
operations目录尚未创建。因此当前没有batch32训练步数、总耗时或实测提速结果。
这里只读本地收据，未增加远程轮询；原固定脚本负责容量验收后继续100步与比较。

两组global batch均为每DP副本64×2=128，microbatch=8、每副本累积8次；服务
上限由4/16384 tokens改为32/131072 tokens。保持完整macro交付和rollout32切分，
同初始策略、同100步样本顺序、同缓存条件、同算法与导出同步路径。batch32是
单个服务内部上限，实际批量受输入长度和token预算约束，不承诺所有批次恰好32。

旧100步实测144.0668分钟、12800真实曝光、1.480794 query/s。每步互斥分区为
actor40.4497秒、rollout35.9363秒、其余准备5.0728秒、导出/保存/同步残差4.9799秒。
服务调优直接覆盖约41.57%的总时间；训练更新没有减少。下方1.5/2/3倍rollout
敏感性表仅是假设计算，不能作为性能预测或验收结果。最终比较器要求两组完整
100步独立审核，另报去掉前两步的98步窗口；启动和结果收集不计入此步耗时口径。

## 后续实际状态及完整100步信号分析

study v2完成旧100步独立审核后，在capacity probe创建远端目录时失败：
`mkdir /data/xingchao/jev-cap-v1/b32`的父目录不存在。已记录attempt，但没有create/
launch容器句柄，尚未开始模型执行。新增capacity operations在原lease、设备空闲、
存储检查之后先`mkdir -p`父目录，再以不带`-p`的mkdir创建本次根目录；保留既有
attempt拒绝重用规则。12项容量/顺序回归通过，覆盖父目录先创建、根目录存在则
拒绝且不create容器。

另封capacity-probes-v2-prepared，新batch32 manifest
`62e4ee4ea0433ce7cd316b9e2cfeb466a67987b8de7ce84dfe11c145ae7ab4eb`，输入hash/
初始策略和模型源码不变。study v3驱动PID1891308/start341002255/cmdhash
`d9307f8e75479bfe2810f6b5f995c71cea25f94d55b6bc90eef2d7b854e3a728`，宏
`.local/config/service-batch32-study-v3.json`，仍使用原100步训练封包。
v2失败输出保留，未原位重试。最新本地ps确认v3存活，正在重审旧基线。

旧micro8完整100步固定脚本分析已结束，实际
`operations/macro-efficiency-v1/micro8-attempt2/efficiency-analysis-v1.json`，本地宏
`macro-micro8-100-efficiency-analysis-v1.json`，准确session72646终态0。全100步
排序755200分量中13593个非零优势组（1.7999%）；存在性768000分量中19707个
（2.5660%）。平均加权采样reward分别0.990782和0.952090；它们不是query准确率、
排序准确率或测试集效果。该分母对分量等权统计，不能当成loss有效梯度占比，
零优势组也可能全错，不能直接判断样本已经学会。

耗时分区：actor46.795%、behavior+credit20.689%、reference18.687%、rollout之外
准备5.869%、导出/checkpoint/同步5.761%、其余rollout2.198%，rank完成时间差约
0.0017%。这支持优先验证服务批量吞吐与actor激活内存，而非聚焦小RPC开销。
数据重采样和双信号调整应在相同效率基座上另开受控实验，本次batch32对照保持
算法与数据顺序不变；没有新的任务/迁移提升结论。

## 最新状态：micro16 显存不足，独立执行服务容量对照

2026-11-16后续：新增固定入口`report_context_actor_oom.py`及portable
`context-terminal-actor-oom.json`，以本地宏`macro-micro16-oom-v1.json`实际生成
micro16目录下`device-oom-report-v1.json`。核验四个容器的命令/镜像/设备及退出身份，
绑定10个输入文件hash，记录双rank设备OOM；6项测试通过。不推断micro32一定失败，
也不将该故障记为core dump或容器主机OOM。

正式准备入口`prepare_context_macro_formal.py`新增显式`comparison_kind=service_batch`。
portable模板`prepare-context-service-selected-formal.json`及
`context-service-formal-selection.json`通过本地宏绑定现有baseline/candidate。
读取两组真实归档重新独立audit，再调用既有容量比较器，按完整100步耗时最小选取；
相同时选batch4。batch32退化时保留退化并选baseline，不要求失败的micro16/未测micro32
完成才能推进。旧microbatch选择规则保持原有语义，不能混用两种比较。
新入口仍只封包，使用相同Stage1重新起步、全量40513两轮634步、LoRA/head保存、
源selection曲线；不自动派发，尾批/恢复/NPU验收仍需完成。
31项选择/封包/原microbatch回归通过；实际两份本地宏解析成功，但batch32未完成，
所以本轮没有执行正式prepare或选出胜者。
本地宏：`.local/config/service-batch-formal-selection-v1.json`、
`.local/config/service-selected-formal-v1.json`。运行中的study v2共73pins复核未变化。

2026-11-16 12:33 UTC：本地终态记录显示，micro16 actor 在 12:27 UTC 抛出
`torch.OutOfMemoryError`，NPU 容量60.96 GiB，已分配57.69 GiB、reserved59.52 GiB，
再次申请178 MiB失败。容器的 `OOMKilled=false` 仅表示未被容器主机OOM杀死，
不能否定日志中的设备显存不足。preflight及双service均退出0，client退出1。
原microbatch序列1849133和等待驱动1885273均已退出，micro32未继续派发，
service-batch32-study-v1因前驱失败退出，**此前等待状态已失效**。

保留全部失败记录，重新配置独立研究驱动service-batch32-study-v2，以已完整审计的
micro8-attempt2成功终态为前驱；原失败序列没有重启。两份已封存的容量probe和
训练包完全复用，训练micro8/global128不变，不修改冻结runtime源码。新宏
`.local/config/service-batch32-study-v2.json`，驱动PID1888130/start340897560，
cmdhash c5f3e9816f4b76b5bee389e8890fb84b7e2e7ecb5c0a1625029674c0e57d0419。
启动前核验原驱动均已消失、micro16四容器收据均已终态、旧pins一致、新probe/训练
尚无attempt。固定启动器仍须拿到设备lease并检查实际设备空闲才会派发。
顺序保持：重审旧100步→batch32实际容量/精度→100步训练→独立审核→效率比较。
任一失败即停，不自动重试或放宽精度标准；当前没有batch32的100步成绩。

旧版每步86.4401秒中，actor40.4497秒（46.8%），preparation41.0091秒（47.4%），
导出/checkpoint/同步残差4.9799秒（5.8%）。rollout35.9363秒包含在preparation中，
不能重复相加。若仅rollout加速且其余部分不变，敏感性计算如下；这些不是实测或
性能承诺，音频逐条编码、token分组、padding和批次内长度差异均影响实际结果。

| 假设rollout加速 | 每步秒数 | 100步分钟 | 总耗时降低 |
|---|---:|---:|---:|
| 1.5倍 | 74.46 | 124.10 | 13.86% |
| 2倍 | 68.47 | 114.12 | 20.79% |
| 3倍 | 62.48 | 104.14 | 27.72% |

公式为 `T_new = 86.440089 - 35.936303 + 35.936303 / rollout_speedup`。
服务batch4→32不能直接视为八倍推理或训练提速。100步两组同为12800真实曝光，
比较完整step墙钟、query/s、推理/actor/同步分解、峰值内存及训推一致性；启动、
初始化、收集和评测时间另记，不能混入上述144.07分钟基线。

用户指定当前两卡训练/双推理实例，服务batch32，比较100步效率。这里指175同机
DP2 ranks4/5与TP1实例6/7，不是跨两台机器。每DP副本有效64，global128。

旧版micro8-attempt2已独立审计通过，12800真实样本曝光、100完整更新累计
8644.0089秒（144.07分钟），平均86.4401秒/步、1.480794条/秒；平均actor40.4497秒，
preparation41.0091秒，导出/checkpoint/同步剩余4.9799秒。reference全部miss12800；
没有暖reference全命中窗口。完整step计时不包含启动、数据初始化、收集或评测。
证据operations/macro-efficiency-v1/micro8-attempt2下audit.json、efficiency-summary.json
及finalization-terminal.json。不要用末步或两步前缀替代此100步基线。

新包保持全部冻结runtime源字节、初始权重、微批8、global128、完整候选60、源数据
前100批顺序、joint GRPO两路信用、FP32/BF16协议、每步LoRA/head保存同步一致。
仅两个service的max_batch_queries由4改32，max_batch_tokens由16384改131072，
以及隔离输出/IPC/冷reference路径。增加token预算是让32条最长4096输入可进入
同一encode的容量设置，不证明真实scheduler一次forward32。

训练包operations/service-batch32-v1-prepared，manifest
6d6c8a6eccae8e27625f0aea6bf5bf73c9a7204fdbd75e3904546f7b4632e2e9。
两个实际封包配置规范化后逐字典匹配。actor microbatch保留8避免将actor提速混入。
另封存同一真实TRAIN128 serial/batch32推理容量包，manifest
626bae2630a35ec82dc1c84380a524b4a725926907fd6ca0c0a72b6c7030f9c4；四pass核验
actual logits/prob/G16、真实worker语言调用数和峰值内存后才准入100步训练。
batch48包仅准备，本次指定对照不派发48。

固定驱动`run_context_service_batch_study.py`严格等待原microbatch序列准确PID终态，
重新核对72个配置/源码pins及完整旧运行审计，再执行batch32容量probe→训练100步→
全步收集审计→compare_context_service_batch。失败保留句柄，不重试、不降低batch、
不启动正式训练。该测试矩阵不会混入DP3或复杂任务。

已启动独立后台驱动PID1885273/start340836719/cmdhash
3a8fe2f0f447a57ae76d49170b262652536e16e2bdb351421b99ccf4ec9003d6，
宏.local/config/service-batch32-study-v1.json，配置模板
configs/portable/operations/context-service-batch-study.json。
当前waiting.json准确等待原sequence1849133/start340484633，实际ps确认两者live；
原序列已记录micro16-dispatch，勿手动重派或并行收集。新batch32尚未开始NPU执行。

比较器验证每步实际输入记录与reference misses一致，核对除声明容量及路径之外
完整训练和服务配置一致，报告全100步与去前2步的吞吐比和节省秒数；两个完整
独立审计必须成功。不声称不同批次引起的实际数值差异无关紧要，逐步behavior/actor
对齐仍沿原强制检查。24项新顺序/失败停止及旧operations/比较回归2.22秒通过。
目前只有旧版实测，不能提供batch32加速结论。训练及容量任务均由固定脚本接续，
无模型周期分析；普通进度观察仍900秒。

## 比较与派发保护的后续验证

新增`test_compare_context_service_batch.py`使用实际100步JSON计时/双rank激活收据
调用真实summary与comparison，独立测试2倍提速及0.8倍退化（负节省时间保留），
完整100步和去前2步窗口分别计算。不使用模拟summary替代真实计时累计。
runtime完整审计收据为测试夹具，不能当作新的模型运行。

拒绝12类污染：microbatch、算法、预算、单个服务batch或token容量、实际样本、
reference miss差异、缺失/额外更新、镜像差异、未通过审核及manifest身份差异。
`test_context_capacity_probe_operations.py`测试batch32/48封包验证，并确认重新封
hash不能更改已声明容量、持有原设备lease时不会发起任何远程调用。
连原study顺序/失败测试共25项3.55秒通过；两个运行驱动共70/72pins均未改。

12:23UTC本地ps确认microbatch序列1849133和batch32等待驱动1885273均live。
对micro16执行普通观察器返回not_due_no_remote_call，下一允许时间1794832558.2137492
（12:35:58 UTC）；本次没有额外远程状态查询或派发。micro16尚无本轮实际step读数。
