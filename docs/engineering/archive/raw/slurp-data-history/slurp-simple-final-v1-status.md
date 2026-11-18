# 简单 Final 工程验收，2026-11-13

用户授权主动调整训练/推理配置。本轮固定任务是 clean SLURP 直接音频输入、
单原生意图 Final60。learning 40,513、selection 9,226；selection 是 TRAIN 内部
隔离选择集，不称官方 TEST。全量两轮日程共 634 更新，无重采样或丢样本。

## 当前摘要（2026-11-14 Stage1成功终态，观察时点逐项标注）

2026-11-14 03:57 UTC完整selection已验收，覆盖下方历史“待评测”状态：
d375c9d7成功退出0/non-OOM；独立审核7b7deb35退出0，collector重核3338文件。
最终634更新adapter在9226条内部selection上HF/vLLM均正确1864条，accuracy20.2038%，
macro-F1固定60类8.7107%，macro-recall8.6210%，invalid0。
全候选概率最大差1.90735e-6，argmax变化0。原基座准确率1.6692%，配对提升18.5346个百分点，
2146来源组/2000次bootstrap的描述性95%区间为[16.5062,20.7330]个百分点。
这仅是Stage1 CE适配效果，不是RL收益或官方TEST/业内横向成绩。

类别诊断：只预测24/60类，23类出现正确结果；calendar_set/play_music占8300/9226
预测（89.96%），因此仍有明显类别集中。后续先用learning样本检查最终checkpoint上的
训练损失复算、预测分布、目标绑定与实际学习覆盖，区分拟合/泛化不足及实现问题；
selection不进入训练或奖励。正式RL、跨任务迁移仍未完成。
结果见[完整验收](../../../slurp-stage1-selection-accepted-v1.json)及
[配对统计](../../../slurp-stage1-selection-paired-v1.json)。

吞吐候选更新（2026-11-14 04:52 UTC）：双卡实际反向micro8约28GiB、micro16约52GiB，
micro32设备OOM；已有两步计时非稳态，不证明分桶加速。确定性配置解决同一case重复漂移，
但v4相同起点micro1与micro8更新后的最大logprob差0.36202，未通过原比较门槛。
这是不同参数的轨迹对照，不等于同一checkpoint训推不一致。v5 `7fa8cdff`已成功结束，
同更新后参数的单条/批量概率最大差9.53674e-7、argmax0，通过原门槛。
但分组micro8导出后使用原HF实现，驻留64条最大差0.44165，未通过；实际候选profile
重载65c153ef已exit0并独立复算：驻留64条对候选重载最大差0，候选对原HF256最大差
0.4580321。这支持本次导出/重载正确，后续需对齐训练/推理计算实现。vLLM首轮a6f14378因遗漏
spawn启动失败，原HF256完整保留；新19e09d4e只重跑vLLM，卡5运行，原门槛不改。
DP4/global128、恢复与跨后端仍待验收，不准入正式训练。
原rc2不修改。见[更新后对照](../../../slurp-true-batch-update-dp2-v4-postupdate.json)。

类别时序：旧日程末8批仅两大类；新class-spread保留全量样本及权重、分散各类到全epoch，
独立人口审核已通过。真实processor `f9af7340`最近到期观察仍running，
30080/81026次使用，未完成验收或训练；独立收集/坐标复算和现有全量训练loader
消费验收入口已准备。

下方过程记录保留各时点事实；本摘要覆盖其中已过期的“待验收/运行中”状态。

进度纠正：运行快照的`progress.step`为零基索引，之前将其再减1的摘要少计一次更新。
本次核对运行源码及四份已写出的update-0269.json确认完成269；不修改训练日程。
rc1固定镜像275项测试通过及其归档仍保留；当前工作树已补恢复后再更新入口并封存rc2。
rc2共1360份源码/配置，19:39 UTC到期观察固定镜像容器exit0；304项全部通过、无跳过。
独立核验1358源码、2配置、归档和runner一致；当前工作树仍匹配rc2。
收据见`slurp-simple-rc2-cpu-acceptance.json`；外部模型/音频/主机替身不构成真实NPU生命周期验收。

| 项目 | 当前已取得的证据 |
| --- | --- |
| 全量 learning processor | 已退出 0，独立核验 81,026 次使用 / 40,513 唯一输入，最长 2,249 tokens |
| selection 输入 | 已退出 0，9,226 条完整核验，最长 2,098 tokens |
| DP4 两步工程前缀 | 4 个 rank 完成；112 LoRA 张量、FP32 master/Adam、持久提交和实际 PEFT 重载通过 |
| HF/vLLM | 256 条完整输入独立验收；最大 logprob 差 9.536743e-7，argmax 变化 0 |
| 恢复驻留 FSDP | 4 rank × 64 请求独立复算通过，最大 logprob 差 0，argmax 变化 0 |
| 全量基座推理 | 9,226 条完整评分及独立复算通过；accuracy 1.6692%，macro F1 0.0547% |
| Stage 1 继续训练 | 原容器实际exit0、OOMKilled=false，FinishedAt=2026-11-14T01:52:06Z；四rank update0634存在，完整634更新结束；最终checkpoint独立审核待验，仍非selection效果或RL收益 |
| 采样 RL + CE + KL | CPU GRPO/RLOO通过；真实NPU GRPO两次128音频/G16更新独立checkpoint审核通过；首更新概率/切换/恢复通过，第二更新概率已独立验收，常驻连续生命周期待验收 |

基座推理评分及前缀驻留恢复概率门槛已通过。剩余顺序：全量 Stage 1 收尾 →
最终 checkpoint 独立验收与同一 selection 评测 → 真实连续音频 RL 与恢复验收 →
不可变正式发布 → 全量 Stage 2。
所有配置调整使用新快照与身份记录；普通运行检查保持至少 600 秒。

原容器docker wait退出通知返回0后重新inspect及终态文件核验；证据
`.local/simple-final-v1/continuation-terminal634-evidence.json`。
最终checkpoint CPU审核容器fa76793b已启动running，下一普通观察至少02:04:09 UTC。
真实BF16批量前向探针45be989a在空闲物理卡5启动running，下一普通观察至少02:04:54 UTC；
消费已认证两步prefix adapter及learning256中的长度分位32输入，非最终Stage1效果评测。
卡4保留用于审核通过后的9226完整selection。0–3此前已被Omni TP4占用，未停止其服务。
新true-batch verl DP4/global128 CPU验证已通过：CE/GRPO/RLOO、micro1..32，
最大参数差2.98e-8；仅合成FP32骨干，非BF16/NPU验收，见slurp-true-batch-verl-cpu-v1.json。

最新终态更新：fa76793b实际exit0（FinishedAt02:16:50 UTC），独立收集核验2040文件，
112张量、精确FP32 master重建/BF16导出/实际PEFT重载及持久提交均通过。
最终adapter身份d8f5ec95234786174e76b3b6246c5f514ed6f8a8eb3ea5397a2744788c26dcf0。
完整9226 HF/vLLM selection容器d375c9d7已在物理卡4启动running，
下一普通观察至少02:31:35 UTC；尚无完整选择集结果或Stage2放行。
卡5批量探针micro2失败已收集，最大logprob差0.83085，不能提升；
额外隔离和逐层诊断已完成，显示音频及语言计算均有batch数值差异，见
slurp-true-batch-isolation-v1.json与slurp-true-batch-layers-v1.json。

本次只读补核已闭合540–571更新、128份四rank日志：平均加权CE 0.342956，
CE/梯度有限，RL与reference-KL贡献均为0；最慢rank更新均值60.77秒，
日志完成间隔均值68.72秒、中位68.60秒。原始记录和复算见
`.local/simple-final-v1/stage1-closed-0540-0571.json`与
`.local/simple-final-v1/stage1-0540-0571-diagnostic.json`。
从571观察点剩余63步按此节奏约72分钟，不含最终checkpoint验收、9226评测及
真实连续RL/恢复验收，不是Stage2启动时间承诺。CE不代表选择集准确率。
00:41 UTC附近175硬件快照八卡健康；4–7运行Stage1，0–3无运行进程。
当前四卡microbatch1及数值一致性配置仍未达到最大吞吐；本轮不改变运行快照。

## Stage1评测启动前校验效率（2026-11-13 18:54 UTC）

尚未执行的本地`launch_stage1_selection.py`将逐证据文件SSH改为一次远程分块hash校验。
保留完整文件清单校验，并绑定审核报告原始字节；不改变模型/数据/运行配置/rc1源码。
在175现有已闭合RL审核图上实际验证7923文件，单次SSH总耗时约0.44秒；本地替身验证
正确图通过、文件篡改/缺失/报告变化拒绝。首次尝试发现主机Python无file_digest，
已用分块SHA256兼容并重新通过实际远端验证，未改主机环境。
收据`.local/simple-final-v1/stage1-selection-batched-graph-verified.json`。
此项仅验启动前图校验，不是最终Stage1验收，不是完整评测启动或NPU生命周期结果。

## 已闭合训练日志诊断（截至更新269，2026-11-13归档）

只读分析已完成的更新3–269，共267更新、1068份四rank日志，没有额外推理或更改训练。
按运行源码的world-size缩放复算真实请求均值，确认当前RL和reference-KL贡献均为0，
实际是CE指令适配；日志中的legacy算法名称不代表正在执行该RL算法。

| 诊断项 | 结果 |
| --- | --- |
| 更新3–34平均加权CE | 5.54566 |
| 更新238–269平均加权CE | 0.37074 |
| 全267步记录 | CE非负有限、梯度有限且四rank一致 |
| 最慢rank更新耗时均值 | 60.92秒 |
| 日志完成间隔均值 / 中位数 | 71.97 / 68.11秒 |
| 最高已分配设备内存 | 17.86GiB |

两个窗口使用不同learning批次，CE下降不是selection准确率或RL收益。
完成间隔来自文件mtime，运行快照没有准备/导出/审核分项计时，不能准确归因额外开销。
若后续保持该节奏，剩余365步约7.3小时，仅是截至269的条件估计，不含最终评测与
工程验收，也不是Stage2就绪承诺。原始摘要与限制见
`slurp-stage1-prefix0269-diagnostic.json`；不为记录诊断改动rc1源码/配置。

## 正式主机派发与收集入口（2026-11-13 18:22 UTC）

固定镜像主机入口CPU `84157ce4…` 于18:31 UTC按600秒间隔观察exit0并独立收集：60测试通过，1355份源码hash与派发归档完全一致。该快照早于service/trainer恢复游标一致性补充和本轮末批测试；当前工作树主机入口29项、分段审核29项本地CPU测试通过。SSH/Docker、外部模型/音频使用替身，不是实际NPU验收。正式Stage2仍未放行。

恢复集成CPU `7aae66ca…` 于18:23 UTC到期exit0并独立收集：177测试通过，1351源码hash与归档一致；真实张量及合成全9226评测，外部模型/音频替身，非NPU恢复。该快照早于主机launcher/collector。

共用两阶段launcher增加--resume-audit-tag与唯一--run-id：保留原配置/参考/生命周期证据，
新目录/服务/训练快照，禁止覆盖旧任务。派发前实时核源容器exited，并在远端逐hash复查
已收集审核及其依赖；不完整逐步策略链/到期评测、损坏依赖、变化实验均在写入前拒绝。
恢复模块认证在torchrun前运行，模型加载后仍走同一actor概率准入。

新增launch_simple_stage2_audit.py独立CPU审核派发，显式prefix游标或完整终态634；
collect_simple_stage2_audit.py复查远端原始证据，完整保留报告字节与收据。
observe_due自动发现stage2-*-launch.json，保持每句柄至少600秒，不自动续派/重启。
具体启动/终态/中断恢复命令见docs/engineering/simple-stage2-runbook.md。

28项本地主机入口测试通过，实际脚本分支执行，SSH/Docker使用替身，不是真实NPU派发。
固定镜像CPU `84157ce4de36d59f65f77b0a7669e660778896c1ddebb2de330265ced181c529` 已启动，按launch600秒；上一resume CPU仍待自己的deadline。
远端tar约39–72秒mtime时差，仅提示；身份继续用内容hash，不因mtime重启。
派发后另补service/trainer恢复审核与游标一致性检查，启动器13本地测试通过，
含新增拒绝路径；该修改晚于host-launch CPU快照，不混记固定镜像验收。
正式开关false，尚待固定镜像结果、真实连续NPU/恢复、最终Stage1完整验收。
学术基线、RL持续收益与跨数据集泛化仍未完成，不把主机入口通过当目标完成。

## 正式恢复加载与分段审核（2026-11-13 18:14 UTC）

中断前缀CPU `081a1dc0…` 已独立收集：131测试通过，1349源码hash与归档一致。
该快照早于本轮加载/评测恢复/策略链/分段审核，不能扩展验收范围。

新增simple_stage2_resume把正式release、最终Stage1固定参考、已退出前缀来源串接。
共用actor加载器显式恢复FP32 master与Adam，保存stage2-resumed.pt；不清空RL优化器，
不把新策略变成KL参考。continuous共用循环从恢复游标开始，首次新行为仍执行actor
概率准入。ResidentSimpleSelection复算全部既有原始评测，再从下一预定时点继续；
不重跑第0点、不覆盖历史文件。采样随机种子已有update/slot/policy身份绑定。

独立审核支持分段：继承认证前缀的逐步检查及完整policy序列，只审核新段更新；精确
比较恢复初态master/Adam，历史评测沿原文件/四rank引用复算，完整终态仍要求634及
全部11评测点。新恢复源必须具有逐步policy链；旧未包含该字段的CPU替身报告不准入。

本地47恢复/状态/评测测试、94入口/阶段/循环/审核回归、110扩展恢复回归与4新增分段
测试通过，计数有重叠。含实际张量下一次Adam更新与不中断运行精确一致；完整9226
合成概率原始重算、外部模型/音频/来源认证替身，不是NPU验收。NPU路径仅语法检查。
固定镜像CPU `7aae66cadeda82936fb40e51c3e63291d84b398752e934ef7961027f902f568d` 已启动，按launch至少600秒观察。
尚缺正式恢复主机启动器/审核派发接入、真实NPU生命周期与最终Stage1验收；formal
false，已跑快照不变。学术基线复现、复杂语义任务与跨数据集收益仍未完成。

## 中断前缀审核与恢复来源（2026-11-13 18:05 UTC）

正式启动器CPU `1d2fbf66…` 已独立收集：47测试通过、1349源码hash与归档一致。
范围是发布认证、真实启动器分支（SSH/Docker替身）及阶段切换，早于本轮恢复修改。

发现旧恢复准入要求源任务exit0，无法处理训练中断。本轮增加显式stopped_prefix审核：
必须固定容器已经exited及确切整数退出码，可缺少因崩溃未写出的terminal.json；绝不
从缺文件或超时推断任务已停。完整终态审核仍要求634步成功，不被该模式放宽。
每个正式checkpoint额外持久化四rank BF16 hook证据；审核指定点前每次采样、状态、
导出与概率，并重算所有到期selection。未来失败后缀保留但不认证；缺到期评测则拒绝
该恢复点，需要明确选择更早的完整点。恢复来源验证保持原算法、参考gate、奖励配置、
评测日程和精确master/Adam游标，工程fixture不能冒充正式checkpoint。

本地78审核/重建/旧流程回归、4正式前缀集成、49恢复相关测试通过，计数有重叠。
实际张量、完整G16采样和概率检查执行；外部模型/音频/来源认证在对应测试中使用替身。
固定镜像CPU `081a1dc0d10ffa0ed3f583375afd170f6221c2781869ff8e632039e7cc4e58f5` 已启动，依launch至少600秒观察。
目前仅补齐审核与来源认证；正式恢复actor/评测游标/循环/启动器尚未接通，也没有实际
NPU恢复或完整634审核结果。formal dispatch仍false；原运行快照不修改。
Stage1最新18:03 UTC四rank正在第236更新，235/634完成。

## 本轮收口与已验收范围（2026-11-13 17:55 UTC）

当前运行Stage1在17:53 UTC观察已完成229/634、准备第230批；四rank一致，原快照继续。
常驻选择集CPU `ee061c1a…` 已独立收集：60测试通过，1344份源码与归档hash一致。
正式终态审核CPU `3a966fc0…` 已独立收集：82测试通过，1346份源码一致。
两者均有外部模型/音频替身，不是实际NPU选择集或完整634步审核验收。

正式两阶段启动器 `launch_simple_resident_train.py --formal`、发布认证及训练入口已实现；
16项发布测试和7项启动器测试通过（SSH/Docker替身）。阶段切换兼容原Stage1配置省略
microbatch字段时已有的默认值1，不改变运行配置。固定镜像CPU `1d2fbf66…` 待按自身
600秒间隔收集，不提前记为通过。正式恢复接入仍待完成，training_dispatch_allowed=false。

收口仅剩：实际常驻连续NPU更新与新进程恢复验收、最终Stage1及完整selection验收、
正式恢复入口和发布快照。PPO、复杂Q、新数据集及吞吐候选不增加到简单模式发布条件。
工程锁定以源码归档hash与配置/模型/数据/镜像身份为准，不以工作树或历史测试总数代替。
已运行版本不原位修复；任何影响数值的修订另发版本，定向验证后形成新实验曲线。

## 正式Stage2独立审核准备（2026-11-13 17:44 UTC）

阶段切换CPU `2abd7c8b…` 到期exit0，56测试通过，1340源码hash与派发归档一致；
证据stage2-initialization-cpu-verified.json。该快照早于常驻评测和本轮终态审核，不能
扩展其验收范围。Stage1最近17:41观察四rank在第221更新（前220完成），继续原快照。

新增simple_stage2_audit核固定镜像成功进程、最终Stage1依赖、四rank初始精确FP32
master与空Adam、完整BF16执行hook，并复核11个评测点原始计分与各rank引用。
原verify_simple_resident_train增加显式正式分支，复用完整采样/参数重建/概率审核，
范围1..634、末批真实请求分母来自schedule，stage=simple_stage2_rl与工程fixture隔离。
模型加载路径新增stage2-initial.pt和执行hook记录供独立检查；运行快照不变。

本地52项初始审核/张量重建/旧连续审核回归通过；扩展21项来源与评测绑定通过，
另1项CPU-only拒绝测试通过，计数有重叠不相加。真实张量和文件执行，外部来源认证/
模型/音频/全量评分在相应单测用替身；完整634正式分支尚无全流程或NPU验收证据。
固定镜像CPU `3a966fc038cd4516928e6b568ae58e4b507f62a0c93129345a0e71c2299c37e5` 已启动，按launch600秒观察；
其快照早于后补CPU-only检查及该1项测试，后续需区分版本。
正式launcher/恢复仍未接通，training_dispatch_allowed=false；学术复现、正式RL收益
及跨任务迁移仍未完成，不把上述工程组件通过作为目标达成。

## 常驻全选择集评测接入准备（2026-11-13 17:38 UTC）

新增ResidentSimpleSelection，按固定0/64/128/192/256/317/384/448/512/576/634步，
对完整9226选择集以73批（72×128+10）执行常驻vLLM推理；完整身份绑定/有限归一化
概率/覆盖通过后才发布指标。gold在推理后本地评分，不进入输入/奖励；原始每批概率
和checkpoint证据hash保存。verify_result独立读取73份原始结果重算全指标，拒伪造汇总。

复用simple_prepared_inputs构造器，原HF/vLLM门槛入口改为委托同一实现，resident复用
已有processor；不是另写转写分支。连续循环评测回调在已接受和提交checkpoint后执行；
每批yield由各rank同步错误/进度，避免非零rank整轮9226评测期间一直等待同一collective。
评测失败或中断毒化owner/session，不静默跳点、不自动重试，不发布部分指标。

本地评测15项、循环12项、共用输入8项通过；之前测试fixture曾因tmpfs存储门槛及错误
schema/依赖构造失败，现仅测试替换外部环境，生产存储门槛保留。模型/音频/engine为替身，
真实张量身份、完整9226概率/计分重算和落盘被执行，不等于NPU结果。
固定镜像CPU `ee061c1abcc0f34cecb2bc1bab3f3743604069b2886333e76e3263bf41dd18b1` 已启动，按launch600秒观察；
前一stage2-initialization CPU仍依自身deadline，未提前观察。
Stage1最近17:31观察213完成、准备214；正式启动器、恢复/独立训练审核接入仍缺，
training_dispatch_allowed=false，已有运行快照不变。

## 正式Stage2阶段切换实现准备（2026-11-13 17:30 UTC）

新增 `simple_stage2_initialization`：要求最终Stage1累计634步及全9226选择集HF/vLLM
独立概率验收，绑定四分片/导出/运行配置和成功终态。共享FSDP恢复器保留精确FP32
master，验证来源Adam后清空状态及残留梯度；RL新日程从0开始，参考固定为最终Stage1。
明确拒绝CE两步前缀、RL工程fixture、部分选择集、旧游标及未清空优化器。

`probe_simple_sampled_npu`增加共享加载分支，`probe_simple_resident_train.continuous`
复用已有循环支持0→634，不复制trainer。原工程fixture仍按1→3执行；运行中快照不改。
正式分支写stage=`simple_stage2_rl`，现有工程恢复器不静默接纳；后续必须显式接正式
启动/评测/恢复与独立终态审核。当前尚无正式启动器，training_dispatch_allowed仍false。

49项本地相关测试通过，包含实际张量的首个新Adam更新精确对照、真实审核器全9226
合成输入概率复算和末条损坏拒绝；模型/发布输入为替身，不是NPU阶段切换验收。
固定镜像CPU容器 `2abd7c8b2429f62e3f4a8350acb30a10f0f73d35166ae0f2e552a9b80258720f`
已启动，按stage2-initialization-cpu-launch至少600秒观察。归档时出现远端时钟落后约
36秒的tar mtime提示，任务已成功启动；身份检查继续使用内容hash，不据mtime重启。

## 连续恢复CPU验收与发布范围（2026-11-13 17:22 UTC）

固定镜像CPU容器 `046ef8c8…` 到期观察exit0：83测试通过，归档的1336份源码逐hash
与远端一致。包含恢复来源/独立审核、精确状态和resident启动/RPC/存储回归；外部
模型/音频/SSH/Docker使用替身。快照早于后续2项输入清单绑定及7项恢复启动器测试，
不混记验收范围。证据 `.local/simple-final-v1/continuous-resume-cpu-verified.json`。

新增恢复启动器7项本地入口测试全部通过，核已有句柄/目录拒绝、卡占用、来源失败、
独立验收缺失、hash变化及成功路径的源码/原审核路径绑定；已接本地观察、独立审核
和收集入口。尚无真实常驻连续checkpoint，不派发恢复，不将CPU结果升级为NPU验收。
Stage1在17:20 UTC观察仍running，四rank正在第207更新（前206步完成），目标634。
正式Stage2仍禁止派发；下一缺口限定为常驻连续更新/恢复、最终Stage1选择集评测和
全量RL统一入口，不新增复杂任务或PPO作为发布前置项。

## 第二更新概率独立验收通过（2026-11-13 17:15 UTC）

finalizer `89be746c…` 已到期exit0并收集，独立复算更新2的HF/vLLM全256输入：
最大logp差9.5367431640625e-7，argmax变化0；更新后resident128对HF差0、对vLLM
最大9.536743e-7、argmax0。原始checkpoint/CPU PEFT/master证据同时绑定且文件hash复查。
证据：`.local/simple-final-v1/sampled-npu-grpo_v1-continue-parity-finalize-verified.json`。

两次分进程RL更新的行为/固定参考/Adam/新策略概率衔接已验收，仍不是常驻连续训练。
真实连续NPU2/3更新及其新进程恢复、最终Stage1/9226选择集和正式全量RL尚待完成。
现有Stage1继续原配置，普通观察不短于600秒。恢复CPU046ef8c8尚待首次到期观察。

## 连续恢复入口与审核准备（2026-11-13 17:13 UTC）

新增simple_continuous_resume在模型加载前绑定独立审核终态、训练终态、四分片/导出、
FP32 master/Adam、算法/参考/完整日程及next cursor；新增probe_simple_continuous_restore
真实DP4重载入口与verify_simple_continuous_restore独立CPU逐元素状态/128概率审核。
精确比较器只在显式continuous=True及正确stage/lineage时支持3–634，旧1/2默认边界保留。
31来源/状态测试、16独立恢复审核测试通过；模型/音频为替身，实际NPU入口仅语法检查。
当前还没有真实常驻连续checkpoint，不能称已恢复或正式RL准入。
启动CPU快照之后，恢复source追加输入/概率文件必须在独立审核清单中的检查；
两项针对性测试通过，此后续改动不计入已启动快照验收。

resident-launch-cpu0b605fc7到期exit1：34通过/1失败，失败为RPC测试tmp_path过长，
没有进入真实服务/训练。测试socket改用短/tmp专用目录并finalizer清理，特意长basetemp
运行9项RPC测试通过；原失败保留。新continuous-resume-cpu固定镜像任务已启动，
包括恢复审核和联合启动/RPC/存储回归；按launch至少600秒观察，不重启旧任务。
Stage1最新17:10观察完成199/634，准备200。第二更新概率独立finalizer仍待到期。

## 第二更新概率任务完成，独立复算启动（2026-11-13 17:05 UTC）

`0809792d…` 已按十分钟到期exit0，62固定镜像测试通过，HF/vLLM各256输入完成。
独立finalizer `89be746c4a7e152d7d80212b41577b6759e9e6cda1c22d770c78deab01365b11`
已启动，依continue-parity-finalize-launch至少600秒，暂不声称独立概率验收完成。

联合启动器新增成功路径测试，实际执行service/trainer两个入口分支，以模拟SSH/Docker
验证五卡隔离、共享IPC、只读模型数据、服务/观察配置hash绑定与任意cwd源码归档。
共5入口测试通过，仍不等于真实五卡执行。已启动resident-launch-cpu快照早于成功路径
测试与cwd改动，验收范围单独记录。下一步实际联合运行仍等待原Stage1释放卡；
最终Stage1/selection、常驻连续更新/新进程恢复、学术复现与效果迁移仍未完成。

## 常驻两批独立验收（2026-11-13 17:02 UTC）

resident-batch-finalizer `1453ee93…` 到期exit0并收集，真实vLLM两批128、固定参考/行为
共四份原始分布独立复算与已接受输出差0；完整日程人口与G16抽样回放通过。
这证明常驻批次采集，不等于同一actor连续更新。证据
`.local/simple-final-v1/sampled-npu-grpo_v1-resident-batch-finalize-verified.json`。
联合启动器现具备该前置凭证，但还需五卡空闲及实际联合执行。第二更新概率仍按其launch等待。
启动器归档路径改为从repository解析，避免从其他cwd运行时漏收源码；4本地前置测试再通过，
已启动resident-launch-cpu快照早于这一仅路径解析改动，保留版本边界。

## 常驻联合启动准备（2026-11-13 17:01 UTC）

固定镜像连续审核测试39d4fda7…已到期exit0，54测试通过，1327源码文件与派发归档
逐hash一致；证据resident-audit-cpu-verified.json。这是CPU合成审核，外部模型/音频
加载为替身，不是实际连续NPU验收。

新增可归档入口 `tools/jev/launch_simple_resident_train.py`，service/trainer显式两阶段：
先验证独立switch/restore/resident-batch凭证及五卡空闲，再启动NPU4 TP1；训练启动前
核实际docker inspect、镜像、设备、只读共享挂载、同一IPC及ready/config。准备0–3 DP4
恢复原RL step1、连续执行2/3；不抢占Stage1、不因超时重启、不覆盖已有root/handle。
输入与导出经共享只读base可见，IPC单独绑定本地/jev-ipc；每容器cache/source隔离。
31本地service/RPC/storage测试、4实际launcher前置失败测试通过；CPU固定镜像新任务
resident-launch-cpu已启动，按launch600秒；尚未启动真实NPU服务或联合训练。

Stage1最新17:00观察完成192/634，准备193。实际连续运行、终态新进程恢复、最终
Stage1参考及正式全量RL仍未完成；算法/数据/现运行快照不改。

## 第二次更新独立验收通过（2026-11-13 16:54 UTC）

第二次更新审核 `7bd5bd23…` 已exit0并收集：固定镜像53测试通过，四rank/112张量、
精确master重建、Adam1→2、PEFT重载、完整128音频采样回放与固定参考通过，
7923证据文件hash复查一致。更新adapter为
`4c4fdb36028f84921fa4e47dc46d3d5fe2ec93f79c895fca2540f708f52644a4`。
这确认两次分进程工程更新的状态衔接，不是同一常驻服务的连续训练或效果提升。

收集时发现safetensors输出mode0600、root所有，主机普通账号无法读取。仅对已终态
第二次更新的单个模型文件改为0644，前后SHA256与独立审核绑定一致，权限收据
`.local/simple-final-v1/sampled-npu-grpo_v1-continue-export-permission-receipt.json`。
`seal_adapter`现于同步前显式fchmod0644；5项本地持久提交测试通过，包括字节不变和
权限先于fsync。没有修改运行快照或禁用fsync；该新源码仍需固定镜像覆盖。

常驻两批验证 `2b754be4…` 已到期exit0；独立finalizer
`1453ee93070166cdb85e6cae7115e4f72bea9c779ee34aefa7950cb89cb1ff16` 已启动。
第二次更新策略HF/vLLM256验证
`0809792d65ca591b66c907e7517e1d4bcc107eaf0b15e94aa2f480bfc43b925f` 已启动NPU0。
均未观察终态，依各自launch至少600秒；正式Stage2仍false。
Stage1最新16:50观察完成183/634，正在184，原配置继续。

## 连续审核补齐（2026-11-13 16:49 UTC）

`verify_simple_resident_train.py`新增独立完整日程采样复算、四rank master/Adam与BF16
导出重建、post-update128全候选概率、各rank下一游标/日程/固定参考核对；终态实际
CPU PEFT重载检查已接入。初始checkpoint也纳入证据hash。缺失/错误commit拒绝。
新测试真实读写四分片张量、执行G16和身份/概率比较；音频/模型/PEFT外部加载为替身。
19个审核用例加10个覆盖/概率用例共29通过，包括刷新commit hash后篡改master或
回退Adam、错参考/日程/游标、重复slot、概率偏差、非有限/零梯度和错误PEFT加载拒绝。
这是本地测试，不能称实际模型重载或NPU连续验收。凭证见
`.local/simple-final-v1/resident-train-audit-local-tests.json`。

独立审核启动器/收集器新增显式sampled-resident模式，尚未派发连续训练。
固定镜像CPU测试容器 `39d4fda7b5fd1d6bf327ca6314327948c1b6b2b6197b195399313f0472ca82a1`
已启动，首次观察按resident-audit-cpu-launch至少600秒。常驻服务与DP4联合启动器、
真实连续运行、终态新进程恢复及最终Stage1初始化仍待完成，正式RL保持false。

## 发布收口与最新证据（2026-11-13 16:42 UTC）

任务定义已冻结为全量 clean SLURP 单 Final60；正式 Stage2 发布仍未完成。
此次只补公共基座已有验收缺口，不增加新算法、复杂任务或吞吐候选作为前置项。
运行中的 Stage1 快照不改；175 卡4–7最新到期观察完成175/634，正在准备第176批。

固定镜像连续 CPU 循环已通过：28项测试，原生 verl GRPO/RLOO、真实 Gloo DP4、
micro1/2各3次更新、独立梯度/master/Adam对照及第二步恢复。40份产物hash与
1323份源码文件匹配派发归档；这是合成CPU范围，不能代替真实音频NPU连续循环。
证据：`.local/simple-final-v1/sampled-loop-cpu-v2-{collected,source-check}.json`。

第二次完整128音频GRPO更新容器 `eaf12481…` 已到期确认exit0，独立审核
`7bd5bd23…` 已启动；未把容器成功等同检查点/新策略概率验收。
常驻引擎两批输入验证 `2b754be4…` 已启动NPU0，尚无结果。二者依各自launch
至少600秒观察。最终Stage1/selection及真实常驻DP4+TP1连续更新仍待完成。

发布只设一个公共版本，算法作为配置项：GRPO先验收，RLOO-clipped随后复用，
PPO-critic独立预留。每项标明“设计锁定/实现验证/正式可派发”，不将配置存在当作验收。
发布清单固定源码hash、镜像、模型、数据、模板/类别映射、输入处理、精度与LoRA、
优化器/日程、奖励/归约、评测与恢复入口及验收凭证。修复生成新快照，旧曲线保留版本边界。

复杂模式先冻结Q语义、监督、动作分布和Final指标，再接同一工程。Direct、Q-input、
Q-credit分开比较；后两者保持相同输入/输出支持与CE/KL，仅改变过程奖励。
标准GRPO/RLOO先不改；固定过程预算且缺标mask，不逐节点额外标准化，不沿Span序号用GAE。
若实测出现Final信用稀释，再独立注册条件信用变体，并核对期望梯度、方差及Final曲线；
不能将变体悄悄替换标准基线。过程奖励预算不等于梯度预算，联合裁剪与节点裁剪也不等价。

## 独立切换/恢复通过，第二次更新启动（2026-11-13 16:30 UTC）

恢复审核6c0e7945…exit0并收集，四rank FP32 master/Adam精确、全256 FSDP/HF/vLLM
最大logp9.536743e-7/argmax0；7889文件hash复查。切换finalizer-v2 95eb24f0…exit0，
old/new/old各256、精确切回、无HF输入行为256精确相等独立复算通过；6555文件hash通过。
两文件集合有重叠，不能合计成唯一文件数。旧v1写出失败记录保持不变。

第二次实际GRPO更新已启动0–3卡：
`eaf12481a8bdbb59e19290e21189c4d65ddd71ef4e4fdda1164fff0ec1b76ed7`。
使用完整第二批128音频、更新行为、固定原参考、Adam1→2和统一CE游标；尚无结果，
依sampled-npu-grpo_v1-continue-launch.json至少600秒观察。非正式Stage2。

新增本地Unix socket常驻服务/客户端及连续NPU回调入口。JSON控制、SHA256绑定的
weights_only张量传输、label-free renderer、严格序号、异常不重试；临时张量成功消费
后删除，失败保留。46本地socket/采集/循环测试通过（真实socket、engine替身）。
`probe_simple_resident_train.py`准备复用同一DP4加载/Adam、独立TP1服务、更新后全128
actor-vLLM对齐及持久checkpoint，未派发、仅语法检查，不能称NPU验收。
需独立空闲推理卡；当前Stage1占4–7、工程更新占0–3，不抢占运行任务。

连续循环CPU v1在phase0的CPU Adam测试触发torch_npu初始化退出，20通过/5失败，
未进入native DP探针。纯CPU测试显式TORCH_DEVICE_BACKEND_AUTOLOAD=0与非fused/foreach
Adam，新v2 `d3bb859f15e1e6392648f779503950edaaa0bab7fd08cbdc3dbab3bd2040026c`已启动；
native verl阶段保留原镜像环境，尚无结果。原快照不改，依launch至少600秒。

Stage1最新16:29观察完成165/634，正在166。正式全量RL仍未放行；下一独立审核第二次
更新与新概率、常驻批次实际运行、连续NPU回调验收和最终Stage1/selection仍待完成。

## 连续循环与恢复执行（2026-11-13 16:19 UTC）

实际DP4新进程恢复dfabd18d…按600秒到期确认exit0，四rank×64恢复概率对HF差0。
独立CPU checkpoint审核已启动`6c0e79455f774732682ef9d14565ed599b4d0bd54783e3b2aa411284c0223509`，
尚未取得独立状态验收。switch-finalize-v1 041b7686…exit1：32固定镜像测试通过，
verify函数返回后写出run/verified.tmp因父目录未创建失败。保留旧失败，补CLI mkdir；
新CPU finalizer-v2 `95eb24f0e679ba1e57eac86fc68cfb85191ed7570328afeeff0876a53e7f24a9`
已启动，复用原NPU成功证据，不重跑NPU；下游GRPO gate明确指向v2。

新增`simple_rl_loop.run`驱动同一个已加载的session：全量日程、采样→原生actor更新→
不可变导出→独立验收→checkpoint/记录；失败禁止隐式重放、每epoch边界和终态保存，
各rank核对日程/游标内容。28本地CPU生命周期测试通过，包含真实小张量Adam续接；
23切换/批次审核测试通过。尚无该循环的真实NPU回调整体验收，不冒称完整trainer。

固定镜像实际verl四Gloo进程GRPO/RLOO、G16、CE/KL、micro1/2、三次更新与Adam恢复
探针已启动`ff33abf7c0276f1d91cf166789e813f2d2c63e0a26b3425361f091ee405aaa86`。
该不可变快照含最初25生命周期测试，不含后加日程内容一致性/末epoch测试，未观察结果。
源码archive清单在.local/simple-final-v1/sampled-loop-cpu-source-manifest.json。

Stage1最新16:19观察完成156/634，正在第157批。继续按各launch至少600秒观察，
优先独立switch/restore收口→第二次真实更新；常驻批次探针已准备待资源，不派发正式RL。

## 驻留切换成功及全量批次接口（2026-11-13 16:09 UTC）

按600秒检查点确认switch 18ed559b…exit0，原始结果旧→新→旧各256且切回输出精确一致；
第四次无独立HF输入的更新策略行为256也精确一致。独立CPU finalizer已启动：
`041b7686e89b0430a07966891524d23ca0afa96bb193f966342b4f56600bbb5c`，尚未观察结果。
实际DP4新进程恢复已启动0–3卡：
`dfabd18d6f44e3931b96cddbb3585336f25f56b368ef43387e27421f2be7d6e5`，尚未观察结果。
按各自launch凭证至少600秒观察；不得提前声明独立切换/恢复通过。

新增`simple_resident_batch.ResidentSimpleBatch`：完整128日程每批音频仅编译一次，
只把不含标签的renderer传入同一vLLM engine，固定参考与更新行为显式分开；初始两者
同identity时复用同一分布，允许连续FP32更新导出相同BF16字节。保持尾批65真+63零权
padding、固定参考配置副本、精确连续游标；失败禁止重试。采样仍复用actor_batch。

实际两批NPU探针和独立审核入口已准备，原始每次rollout与采样结果保留供重建；
使用现有启动器`--resident-batch`，依赖独立switch准入，尚未派发。
50 transport相关、50 schedule/switch相关以及17新collector/审核测试通过（有重叠，
不可相加为唯一测试数）；音频和engine用替身，非真实NPU或全量trainer验收。

Stage1最新16:09观察完成147/634，正在第148批。正式Stage2保持关闭；剩余恢复、
连续第二次更新、常驻完整训练循环和最终Stage1/selection必须继续完成。

## 更新策略独立验收通过（2026-11-13 15:57 UTC）

按600秒检查点确认finalizer 5dc33a11…exit0并重新核对证据hash：HF/vLLM完整256，
最大logprob差9.5367431640625e-7、argmax变化0；128驻留FSDP对HF差0，对vLLM同上述界。
概率报告5234文件、下一行为报告6545文件hash均通过（两集合有重叠，不当作独立文件总数）。
第二批128音频采用更新行为/固定原参考、游标1及原生DataProto/优势传输通过；
全40513人口/81026使用/634更新与权重复核通过，尚未执行第二次optimizer更新。

驻留旧→新→旧及无重复HF行为采集验证已启动NPU0：
`18ed559b6c03691757e9edeebb0854f9d3e97644598a58aa96dc97e7e07d8cea`。
后续观察遵守sampled-npu-grpo_v1-switch-launch.json的600秒下限。
新进程恢复、第二次更新及连续全量trainer仍待完成；正式Stage2准入false。

## 本轮收敛与效率接口（2026-11-13）

统一执行规约见`docs/research/23-task-engineering-rl-baseline-lock.md`顶部。
任务已冻结；Stage2配置设计与正式工程发布分开，`training_dispatch_allowed=false`不变。
GRPO主线、RLOO-clipped对照、PPO-critic未实现预留；不扩展当前发布前置范围。

resident rollout已准备无需逐批独立HF的行为采集入口；保留输入/策略版本、BF16 LoRA、
候选概率和执行检查，evaluate仍要求独立参考。旧→新→旧切换后增加一次更新策略的
无HF采集，由独立审核与已验收输出精确比较。22项本地替身/审核测试通过，真实vLLM
尚未执行；不能算新的后端验收，也不代表连续全量RL trainer已完成。

## 更新policy对齐执行成功（2026-11-13 15:47 UTC）

parity-v2 87a98d…到期确认exit0；62固定镜像测试通过。实际HF/vLLM各256条，
后端记录maxlogp9.5367431640625e-7，argmax变化0；待独立复算，不能提前声明正式gate。
/jev-tmp实际ext4/rw，UUID socket路径45bytes<107，短本地路径修复在NPU运行通过。

独立CPU finalizer已启动：`5dc33a11aef9a1e04322313b2458a2e691353202622396c95adc8ceb082cae3f`，
同一次验收复算256HF/vLLM与128驻留结果、再跑完整第二批音频新行为/固定参考/DataProto。
首次观察按sampled-npu-grpo_v1-parity-finalize-launch.json至少600秒；不提前读其结果。
后续resident switch、新进程恢复与第二次更新仍待派发，正式RL准入保持false。

## 全量RL日程接入（2026-11-13 15:45 UTC）

`simple_rl_schedule.SimpleRLSchedule`读取原完整两轮日程，核manifest/epoch/hash、
每轮40513唯一录音、固定来源绑定/类别权重、终端63个零权padding、真实分母与跨epoch
游标。对actual official-clean-intent-v1逐项核对人口hash，并独立重算60类sqrt-cap4
再归一权重，全81026使用/634更新通过；末批65真实+63padding，不额外重采样。

已接`probe_simple_next_behavior.py`及第二次更新入口，第二批CE系数取统一游标；
28项相关测试通过。实际收据`.local/simple-final-v1/rl-full-schedule-verified.json`。
这不是全量RL trainer已经验收；连续resident采样/更新、完整selection曲线仍待完成。
共同actor配置与已通过的NPU v2源码做AST对照：25执行字段完全相同，只有policy绑定
提取为显式参数；独立凭证shared-actor-config-equivalence.json，非运行验收。

Stage1在15:38 UTC到期实查running、四rank零基step120正在更新，即完成120、正在121。
新策略parity-v2尚未到首次观察；依原handle/600秒，不重启、不提前读取。

## 新策略parity启动故障与修复（2026-11-13 15:37 UTC）

809f72ee…到期观察exit1；57固定镜像测试通过，HF完整256成功，vLLM engine启动
ZMQ IPC路径超107字节而失败，0条vLLM完成；没有概率漂移结论，非OOM。
根因是实验目录加UUID过长，不是checkpoint损坏，不重启v1。

新增local_ipc长度/真实本地挂载检查，新容器把本地cache绑定`/jev-tmp`并设置
TMPDIR/VLLM_RPC_BASE_PATH。22本地相关测试通过（含真实Unix socket bind）；
相同短路径改动应用于未派发的switch/Stage1 selection启动器。未修改旧运行快照。

新v2容器`87a98d8265db5e35677d291931da43ffa7c426a7ee95b8d7cbb1efbc443bf28b`已启动NPU0，
配置/源码另存sampled-npu-grpo_v1-parity-v2；下次观察按launch凭证600秒下限。
正式Stage2准入仍false。原参考与更新checkpoint身份、精度和概率阈值不变。

## 第二次更新工程准备（2026-11-13 15:34 UTC）

新增`probe_simple_sampled_continue.py`复用相同FSDP加载/actor配置，先精确恢复已验收
第一步master/Adam与256驻留概率，再用第二批128真实音频、新行为和固定原参考执行
一次更新；Adam接续1→2，CE系数使用634预算的第1游标，不重置优化器。
第二步导出/四分片持久提交/同驻留重载保持awaiting_acceptance。

`verify_simple_sampled_continue.py`先复用独立恢复审核，再重建第二步112张量和Adam
游标、实际CPU PEFT、完整第二批音频/采样重放。启动器依赖已验收parity、resident
switch和fresh-process restore，尚未派发。第二步HF/vLLM比较显式绑定behavior而非
fixed reference，复用同一parity/finalizer，不复制训练基座。

60状态/输入/恢复相关测试、9第二步独立审核测试、27概率审核测试通过；测试中重模型
与音频/恢复外部加载用替身，真实NPU第二次更新及独立验收仍未发生。原在跑快照不改。
Stage1于15:28 UTC观察仍running：完成111更新、正在112。新策略parity首次检查仍按
原launch凭证600秒下限，未提前读取。

## RL checkpoint独立验收完成（2026-11-13 15:24 UTC）

到期观察审核容器133f8e0e…已exit0。固定镜像93测试通过，独立重建112张量/
四rank FP32 master与Adam、精确BF16导出、实际CPU PEFT重载、完整128音频/G16重放通过；
重新检查3917个证据文件hash全部一致。state收据sha256为
`f7b45d24cc89716bcc3f33be137d3b1d10a0d200bb58540589a02c69f7f61a61`。

实际2048抽样中30次正确；128组有19混合、109全错、0全对；混合组请求权重
31.11088585/181.10153673。仅学习工程fixture，不能外推最终Stage1能力或训练收益。

新策略HF/vLLM全256检查已在NPU0启动，容器
`809f72ee53540dce55d27d12af8e330be514ffa280e9cf4da6de8937b4d494ce`；首次普通检查按launch记录至少600秒，
尚无概率结果。Stage2准入仍关闭，后续依次独立parity、switch/fresh restore及连续更新。
下一批行为绑定新增10测试及关联测试共51通过，追加状态机/恢复相关44测试通过；
音频替身测试与真实NPU执行分开记。

## 下一行为版本绑定（2026-11-13 15:22 UTC）

上一轮有实际进展：GRPO NPU单次更新成功并启动独立审核。本轮新增
`simple_next_behavior.next_batch`，从更新policy的完整后端结果选取下一批原始日程，
固定参考仍为原policy，重新绑定请求/采样游标，保留类别权重、padding及全部G16抽样。
51相关测试通过，覆盖错误旧行为、移动参考、缺失/重复输入、错误音频、游标及标签独立性。
测试音频加载使用替身，不声明实际第二次NPU更新。

`probe_simple_next_behavior.py`已接新策略parity独立finalizer：只有实际新策略HF/vLLM
验收成功后，才用完整第二批128音频与固定原参考构造GRPO/RLOO原生DataProto并复算优势。
尚未执行。它是连续更新的输入准备，仍需真正第二次优化与Adam接续验收。

Stage1在15:17 UTC按到期间隔观察仍running，四rank均零基step102/update，
即102更新完成、正在执行103，不把开始更新当成完成。原快照不变。

对已完成GRPO探针的原始记录做只读汇总：四rank RL surrogate数值和约-4.66e-10，
CE贡献和30.7698978，总梯度norm24.84954。组中心化且old=current时RL标量可近0，
不能据此判定RL梯度为0；总梯度也不能证明RL分量有效。记录采样混合组与后续真实
selection曲线，必要时单独诊断梯度。上述跨rank和不是归约后的全局loss指标。

## 最新收口（2026-11-13 15:14 UTC）

GRPO v2于15:12 UTC到期观察exit0，四rank报告收集通过；128真实音频、G16、
一次更新，四rank梯度范数报告24.84954，原始checkpoint恢复256概率差0。
这是两步CE工程fixture上的更新，不是最终Stage1参考或持续RL收益。
独立CPU审核容器`133f8e0e274c0586fddd5fd16a49839716c4666af91dc3022ec71aac7d5e3ac2`已启动；
下一检查依launch凭证至少600秒。不提前读运行结果，不重启旧任务。

策略切换独立审核已接共同finalizer，支持`--sampled-algorithm grpo_v1 --sampled-switch`；
审核old→new→old三轮完整256输出、实际BF16 worker与精确切回。
本轮checkpoint/switch审核20项测试在系统Python(torch2.10.0+cpu)通过；项目.venv缺torch，
同测试2项模块跳过、exit5，未计通过。重模型/音频加载的测试替身不替代实际固定镜像验收。
启动器语法检查通过；真实switch、updated parity、fresh restore及连续新行为更新仍待执行。

统一任务、工程发布边界与算法规约见doc23顶部；Stage2 training_dispatch_allowed仍false。

## 本轮补充（2026-11-13 14:36 UTC）

micro2吞吐探针已实际退出0：global128、DP4、256真实请求，两步2.21091 req/s，
较micro1的1.88157约+17.5%；allocated约28.77GiB、reserved约60.06GiB。
不升级正式配置，尚无全量容量/概率与恢复验收。

micro1独立状态/PEFT审核通过，但随后的跨运行对照退出1，报告initial adapters differ。
只读逐文件核验显示初始权重SHA完全相同，配置仅target_modules列表顺序不同且集合相同。
最终跨运行master/Adam比较未执行，整体对照未通过；不重启失败容器、不修改旧产物。
因此尚不派发依赖该对照的NPU采样RL探针。详见doc23当前执行契约。

## 后续实施（2026-11-13 14:43 UTC）

Stage1于14:37 UTC按到期间隔核验原容器仍running，四rank均在零基step65，
即累计第66步更新，不代表该步已完成。

adapter对照增加显式语义比较：只允许target_modules唯一字符串列表的集合顺序差异，
权重及其他资产仍逐字节相同，其他配置字段严格相同；原始artifact/policy ID保留。
两端FP32 master和Adam仍逐张量精确比较。26项相关测试通过。
新固定镜像CPU对照容器`b9d66d15bebeaee10c5549d1750292c7e984bfbf262d5a47e35b6772caa0c3bd`
启动时实际running；凭证为`.local/simple-final-v1/batch-g128-m1-audit-v2-launch.json`。
首次普通观察按凭证至少600秒；原v1失败容器不重启，原文件不修改。

新增RL独立checkpoint审核入口`tools/jev/verify_simple_sampled_checkpoint.py`与
四分片重建模块：固定算法/参考/Adam、112张量导出、真实CPU PEFT重载、完整128
音频编译和G16采样重放，以及已记录的恢复概率覆盖。重建模块16项正负测试通过；
连同artifact/state/resume相关测试共58项通过。该入口尚未对真实RL产物运行，不能
声明验收完成。NPU探针保留两套artifact身份，只有精确权重/配置及master/Adam
确认后才绑定已验收行为ID；独立新策略HF/vLLM和新进程恢复仍需另验收。

## RL策略闭环准备（2026-11-13 14:49 UTC）

原Stage1于14:47到期实查仍running，四rank完成75更新、准备第76批。
micro2独立CPU checkpoint审核已启动，容器
`fe8fb731a32a63176cc9416b1531b930104850dca8873a773f83869f6a2f2355`，
首次观察至少600秒，不占NPU。

新增`simple_sampled_parity`与现有独立parity审核接入：重新绑定新旧policy身份，
将已审核的128驻留FSDP结果与fresh HF/vLLM全256输入中的对应请求比较完整Final60。
缺少已审核输入/输出、错版本、错算法、错游标和概率漂移均拒绝；25相关测试通过。
更新现有启动器支持RL新策略评测和CPU独立终审，不改已有运行快照。

另准备`probe_simple_sampled_switch.py`，同一resident vLLM引擎跑旧→更新→旧三轮
完整256音频，逐轮比独立HF参考并要求切回原策略输出精确；依赖新策略独立验收，
尚未派发或取得真实结果。fresh-process FSDP恢复和连续新行为RL更新仍未完成。
本地追加resident rollout测试因无vLLM在收集阶段退出2；不计通过，固定镜像前置测试包含该项。

## 实际对照与恢复路径调整（2026-11-13 14:52 UTC）

micro1对照v2到期确认exit1；113固定镜像测试和独立checkpoint审核通过，跨运行
master/Adam比较失败。逐张量诊断仅rank0出现差异：143360元素中master1个、
exp_avg2个、exp_avg_sq2个；最大绝对差分别1.4551915e-11、1.4551915e-11、
1.1102230e-16，其余三rank逐位相同。BF16导出文件完全相同。
这是跨运行重新训练的微小漂移，不是精确checkpoint恢复失败，具体数值差异来源未定。
不放宽原对照或恢复阈值，v2失败保留。

NPU RL探针改用已验收checkpoint直接初始化：恢复原始四卡FP32 master/Adam，
在新进程核验256输入与已验收HF概率，保持状态精确，然后显式重置阶段Adam、
执行一次全128音频G16 GRPO+CE+KL更新。移除“必须重新训练两步且逐位相同”
这一非恢复前提；保留固定镜像状态/概率及音频transport准入，实际更新前仍精确比较
恢复后的原始状态。已有Stage1与历史容器均不修改。

GRPO NPU工程探针已启动：容器`bbf94cae10580c845bb727687909f12539d3e60f549ad5caa771bffddbbbffc2`，
启动时实际running，物理0–3；完整配置/源码快照见
`.local/simple-final-v1/sampled-npu-grpo_v1-launch.json`。
首次检查不早于凭证中600秒下限，尚无更新结果，不提前宣布RL验收通过。

## 新进程恢复审核准备（2026-11-13 14:57 UTC）

已复用共同BF16/FSDP加载路径增加只恢复模式，不重训CE，不重置已保存RL Adam，
也不执行下一步优化器更新。前提为新策略独立checkpoint审核及HF/vLLM全概率通过；
加载原始RL四分片后验证全256驻留输出，保存实际恢复状态。
`verify_simple_sampled_restore.py`从独立CPU进程核验容器成功终态、依赖身份、四卡
master/Adam精确相同及全部256请求对HF/vLLM的完整候选概率。
7项新审计正负测试与恢复/状态机相关共50测试通过；尚无真实RL恢复结果。
启动器`.local/simple-final-v1/launch_sampled_restore.py`和独立审核模式已准备，
不改运行中的GRPO探针源码快照，正式Stage2仍未准入。

micro2审核已于14:56到期确认exit0，独立CPU重建112 LoRA张量、四rank master/Adam、
实际PEFT重载、durable commit和256监督记录通过；全部24证据文件hash复核通过。
收据`.local/simple-final-v1/batch-g128-m2-state-verified.json`。其独立HF/vLLM入口
已准备但尚未运行；短前缀吞吐+17.5%不能外推全量稳定性或升级现有曲线。

Stage1于14:57到期实查原容器仍running，四rank在零基step84，即累计第85步更新；
GRPO探针首次普通观察不早于15:01:25 UTC，当前尚无结果观察。

## GRPO首次终态与修正（2026-11-13 15:04 UTC）

v1于15:01到期检查确认exit1，失败发生在phase0前置测试：将CPU-only独立对照测试
放到挂NPU的容器触发设备拒绝，1失败/109通过；未进入模型恢复或RL更新，非OOM。
删除NPU启动器中的CPU-only对照测试项，该测试继续保留于独立CPU审核，不改验证
逻辑或数值阈值。旧容器/日志/配置保留，不重启v1。

v2新快照已启动，容器`faaa4e9f2ad104311ca51dfd3a14dbd5c70b9de5b7c28d58cdb648c034ed7d1c`，
175物理0–3，启动时实际running；首次普通检查不早于`2026-11-13T15:12:46.781399+00:00`。
配置、源码和句柄见`.local/simple-final-v1/sampled-npu-grpo_v1-v2-launch.json`。
后续checkpoint审核明确绑定v2句柄；正式Stage2仍未派发。

增加终态结果收集入口及独立审核收集入口：核成功容器/配置、四rank、原始状态恢复、
完整128/G16和证据hash，不把进程退出0当成全部验收。独立checkpoint审核新增从
实际动作与重新编译目标计算全错/全对/混合reward组及权重覆盖；该诊断不代表RL
独立梯度或选择集收益。新增收集器已语法检查，尚无真实RL成功产物可收集。

## 过程证据（按实施阶段保留）

## 已完成

- 实际固定镜像四 rank CPU verl 更新与 Adam 恢复通过：microbatch 1/2，
  三次更新、非均匀权重、padding 与非法输入拒绝；最大梯度误差
  2.9802322387695312e-08。此结论仅覆盖合成 FP32 CPU 计算。
- 类别权重按完整 learning 冻结，使用真实请求数归约，padding 排除。
- 新增本地 checkpoint 原子提交：文件 fsync 后 rename，再同步目录；
  adapter 完成同步、全部四 rank 分片完成后才发布带 hash 的 committed 标识。
  写失败和同步失败测试验证不会发布部分文件。历史 checkpoint 不改写。
- 本地轻依赖验证 80 passed，torch 模块跳过；系统 Python 补充 objective 和
  checkpoint 测试 8 passed，1 个需要本地 verl 的测试跳过。
- 共享清理子进程删除 1,247 文件、1,842,093,958 逻辑字节；归档结论及配置。
  收据：`/mnt/cephfs/xingchao/asset-registry/jev-cleanup-20261113-shared-v1/summary.json`。

## 运行状态

实际容器、不可变配置身份和下次观察下限见
[progress.json](../../../slurp-simple-final-v1-progress.json)。普通运行检查至少间隔 600 秒。

1. 全量 CPU processor v2：修复 v1 漏挂历史发布依赖文件导致的退出1。
   原失败证据保留；新容器增加逐文件只读挂载和显式本地缓存。
2. 四卡 NPU prefix：已实际退出0，固定镜像89项测试通过，四个rank均完成两个更新。
   Qwen3-ASR-0.6B，BF16 q/v LoRA r4/alpha8，FP32 Adam master，LR 1e-4，
   seed317，物理4–7，global128、micro1、每 rank 累积32，context4096。
   这是全量日程的工程前缀，不是完成两阶段训练或效果验证。
   两步各约63.08/58.75秒，各卡峰值约14.94–15.06GiB；计入模型加载等后每rank
   总计约156秒。两步不能证明全量稳定吞吐或效果。
3. 独立 CPU 首256输入生成：已实际退出0，固定完整前两批的波形、token、Span映射，用于后续
   checkpoint 重载和 HF/vLLM 全候选概率对齐。
4. checkpoint CPU独立重载审计已实际退出0：112个LoRA张量、四rank master/Adam、
   256条监督输入及持久提交核验通过；实际PEFT重载精确。原导出和重载adapter身份
   同为 `b347b175f7c3418049cd8cda9f85facdd81901e56ed8af6035ccd846eec45136`，
   `legacy_export_repair=false`，未发生命名修复。普通用户读取新原子提交文件
   曾因600权限失败；只对本次输出增加读取权限，内容和运行源码快照未改。
   后续保存实现改为提交前设置644，4项保存失败/同步/权限测试通过。
5. 当前prefix的HF/vLLM全候选概率任务已启动，容器
   `df4c917869843d695c2cc07090ac21ead57acae76034d0eb8bfda17e91048726`，
   物理NPU4，原固定runtime profile，显式spawn，256完整请求；已实际退出0。
   本地从原始HF/vLLM结果独立重算全部概率，最大logprob差9.5367431640625e-07，
   全请求通过；完整远端输入/状态证据审计现已实际退出0，`verified.json`通过。
   原始训练状态、适配器、完整输入及概率绑定均已核验；argmax变化0。

所有任务输出/缓存位于175本地 `/data/xingchao/multimodal-jev/simple-final-v1`；
Ceph、模型、数据及历史依赖只读。源码使用新快照，不修改运行中快照。

## 后续准入

全量评测入口补齐（本轮）：复用独立 processor 工具，新增 `full_selection` 模式，
加载同一正式发布的全部9,226条选择集，模型输入只由音频和任务定义编译。
已逐条核对全部9,226输入与训练direct/native编译结果完全一致；真实CPU processor
全量任务已派发，尚未观察终态。原首256输入任务源码快照不受本地修改影响。

评分器新增原始HF/vLLM结果入口，核对实际容器成功终态、配置与依赖身份、独立输入
artifact、每条请求hash/波形处理/token/Span/候选绑定，然后用完整正式选择集评分。
22项测试通过，包含错容器、错策略、错配置、部分/重复预测的拒绝；这不是实际推理
或scorer全链路通过。checkpoint独立验证入口增加四rank持久提交标识/hash核验，
新验收配置要求 `require_durable_commit=true`。

独立概率验收工具 `verify_slurp_simple_parity.py` 重新计算全部256请求的HF/vLLM
候选概率界限，不依赖结果内passed字段。6项测试覆盖伪造通过、缺失请求、错输入、
未持久提交与容器仍运行等拒绝。当前已有prefix实际HF/vLLM概率结果，仍须完成完整
恢复驻留FSDP验收；完整证据审计已通过，仍不能宣布全链路训推一致。

恢复与全量输入核验补齐：恢复接口支持显式原始processor证据路径；简单模式要求
独立账本核验与原结果hash绑定。`simple_processor_audit`逐项核验两轮全部非padding
日程坐标、音频身份、请求编译、Span位置、复用输入及各轮完整覆盖。61项相关测试
通过。全量学习processor和全量选择集processor最近按600秒间隔均确认仍在运行，
不把进度当成通过结果。

已准备但未执行的配置生成入口位于 `.local/simple-final-v1`：
`verify_full_processor.py`核原始完整账本；`prepare_continuation_config.py`仅在前驱
概率及processor验收完成后物化剩余632步配置；`prepare_base_evaluation.py`要求全量
选择集输入成功，确认初始LoRA的56个B矩阵全零和物理NPU4空闲，再派发完整基座推理。
当前没有基座推理结果或全量继续训练派发。

继续训练启动器 `launch_continuation.py` 已准备，先检查4–7卡空闲，完成元数据/
存储准入和固定镜像相关测试，再启动恢复训练；恢复入口先比较驻留FSDP与已接受HF
概率，之后才更新。该启动器尚未执行，仍等待当前概率及全量processor验收。

标准RL接口开始接入（不改运行快照）：实际读取固定镜像 `/opt/verl/verl/trainer/ppo/core_algos.py`，
SHA仍为 `fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24`。
`span_group_advantage`直接调用原生GRPO/RLOO估计器，固定完整G组、保留恒reward组，
不额外whiten。现有typed native actor增加算法配置及跨rank一致性校验；clipped
surrogate保留该固定verl的dual-clip=3行为，RLOO明确是clipped版本。
本地8项通过、2项因缺verl跳过；已派发固定镜像原生估计器及四rank CPU更新/Adam
验证，尚未观察终态。该分支尚无CE混合、padding归约、音频/NPU RL验收或效果结果，
不能据此宣称标准RL训练基线已锁定。原CE hybrid actor和运行中的prefix不变。

采样RL+CE接口 `span_sampled_hybrid` 已实现：对每个请求的G个完整动作取均值，
辅助CE每个请求一次，类别权重在请求外层，两者按全局真实请求数归约；显式padding
零权重，恒奖励组仍保留。调用固定verl原生clipped surrogate，不改历史精确边缘PPO。
本地5项拒绝测试通过、1项需verl跳过；固定镜像13项相关测试已派发，核对独立公式与
模拟跨rank/microbatch梯度，目前未观察终态。尚未接入actor生命周期或NPU RL，
不把损失函数实现当成训练准入。

最新全量learning processor实际运行观察已完成19,200/81,026次日程使用，当前最长
2,249tokens；记录为中途进度，不代表完整长度上界或准入。

全量selection processor现已退出0；独立逐条核验9,226输入全部通过，最长2,098tokens。
初始adapter的56个LoRA B矩阵实际读出全零；NPU4空闲检查后已启动完整基座HF推理，
容器 `75865a6d4a0d76ebcbbd92673e3d48dd00f268c42a535fc43fca5155354fe8d8`。
这是直接音频+单Final的零训练基座控制，不使用转写或TEST；推理和指标尚未完成。

GRPO/RLOO固定镜像验证现已退出0：10项函数测试通过；分别完成真实4rank Gloo/DDP
typed native actor、micro8/16、两步更新及Adam检查，G8合成输入。
GRPO最大梯度误差1.4901161193847656e-08、RLOO为7.450580596923828e-09；
最大参数误差分别2.9802322387695312e-08、7.450580596923828e-09。
此结果不含音频/BF16/NPU、G16、padding及辅助CE actor集成，标准训练准入仍未通过。

采样RL+CE固定镜像v1已退出1：12通过、1失败。失败为测试在独立参考求导后复用
已释放计算图；损失值比较已通过，梯度等价未完成。测试改为新前向构图，未改损失
公式或放宽误差，v2新快照已派发。原v1失败日志保留，不记为梯度验证成功。

基座评分启动器 `score_base_selection.py` 已准备，要求实际全量推理退出0，固定镜像
先运行30项计分/证据测试再评分。评分器还会独立将全部模型输入与正式选择集及任务
定义逐条核对。尚未派发评分，等待完整推理；不输出部分成功名单的指标。

采样RL+CE公式v2固定镜像已实际退出0，13项测试通过，包括独立损失/梯度以及模拟
rank/microbatch归约。`span_hybrid_actor`已接标准采样分支，保留instruction和旧exact
默认路径，限定单Final全监督；采样旧概率/正确性组优势在更新准入时复算，各rank
核对algorithm/G/真实分母。新的真实4rank CPU联合更新任务已派发：GRPO/RLOO、
G16、类别权重、零padding、三步衰减CE、Adam保存恢复及旧instruction回归；尚无终态。

最近按600秒间隔核验：基座推理3,579/9,226、learning processor32,896/81,026，
两者均running。仅进度，不评分、不派发剩余632步、不声称音频RL已准入。

简单RL候选已单独登记 `configs/experiments/slurp-simple-sampled-rl-v1.json`：G16、
一次optimizer更新/behavior批次、clip0.2、固定verl dual-clip3、衰减CE、固定Stage1
参考精确KL0.01。明确区别于旧研究候选“两次更新、无dual-clip”，不冒称同一配置。
精确KL接口按每真实请求一次归约，padding为零，必须有固定reference身份；
本地相关9通过、1因无verl跳过，含KL四rank实际更新验证另启新快照，尚未观察终态。
配置training_dispatch_allowed仍false；不修改正在运行的CE或推理快照。

不含KL的联合actor验收已实际退出0：GRPO+CE、RLOO+CE与instruction回归均通过。
各包含4rank × micro1/2 × 3步共24项rank-step记录，含G16、类别权重、padding、
恒奖励组、Adam保存恢复和异常元数据拒绝。两种采样算法最大梯度误差均
5.960464477539063e-08，最大参数误差均2.9802322387695312e-08；instruction
梯度误差2.9802322387695312e-08。含参考KL的后续快照仍待验收，音频NPU RL未准入。
最近基座推理实际运行进度7,116/9,226，尚未全量评分。

核验真实终态 → 独立四分片/Adam/PEFT重载验收（新导出设置
`legacy_export_repair=false`）→ HF/vLLM及恢复驻留模型概率验收 → 从原日程游标
继续全量训练；同时补齐全 selection 基座与训练后评测。全量 processor 和独立 HF/vLLM
验收已通过，恢复驻留 FSDP 验收仍待实际执行。原生 GRPO/RLOO 联合 actor
已通过不含 KL 的 CPU 验证，NPU RL 生命周期待集成；当前 instruction 更新是 CE，
不能把旧定制精确边缘 PPO 或此次 CE 称为标准 GRPO 训练结果。

2026-11-13 全量训练输入已完成并独立验收：40,513 唯一录音、两轮 81,026 条
日程记录逐项匹配，最长 2,249 tokens，4,096 上下文无截断/溢出。恢复配置继续
原日程第 3 步至第 634 步，不重放已完成两步、不改变 LR/人口。尚未派发；
等待完整基座 selection 释放 NPU4，随后先核验恢复驻留模型再更新。

含精确参考 KL 的联合 actor 已实际 exit0：GRPO/RLOO 各 24 项 rank-step，
G16、micro1/2、衰减 CE、KL0.01、权重/padding、三步更新和 Adam 恢复均通过。
最大梯度误差 1.1920928955078125e-07，参数误差 2.9802322387695312e-08。
同镜像 instruction 回归通过。仅合成 FP32 CPU 范围，NPU 音频 RL 未准入；
后续仅日志字段 reference_kl_contribution 的改动不包含在本次快照验收。

2026-11-13 13:23 UTC 实际派发收据：基座推理容器已核验退出0。独立评分容器
`e6c8923b866a6a3541c400df73ed17864b6064a8f9b04e9fb85307d202076d32` 已运行。
NPU4–7 空闲检查通过，Stage1 恢复容器
`fe9ed5e21a933cbe3e2351b3025914697235c464918a4c82a56e68c4e7cc2f8a` 已运行。
恢复前置元数据/存储/测试及驻留概率验收结果尚未观察，不宣称新更新已完成。
继续配置从 completed=2 执行 632 步，检查点每64步、epoch边界和最终步保存。
首次普通观察不早于 2026-11-13T13:32:59.839283+00:00；
评分同样按其收据至少600秒检查。

终态验收扩展：`slurp_instruction_audit` 和 checkpoint verifier 现可核对完整634步
终态，区分632步本次执行和前驱两步；独立核对完整日程、非导出步策略标识、epoch
边界、四rank更新账本、112 BF16执行hook、最终master/Adam和PEFT重载。85项
相关测试通过。固定镜像真实两步checkpoint回归容器 `1f4f3aefc1caa362367047c26b4be43adc8cfaeb8d7d757f82268b748be06893`
已派发，结果待观察；全量训练终态尚不存在，不宣称全量验收通过。

Stage1 训练后衔接启动器已准备但未派发：先独立全量终态checkpoint验收，再用
与基座完全相同的原推理源码/profile/9,226输入做 HF 和 vLLM 全量候选概率核对。
新adapter必须来自累计634步、cursor=(2,0)、四rank master/Adam及PEFT重载均通过
的实际终态。已有容器收据时拒绝重复派发；共享目录只读、所有热写本地。

首个到期观察：恢复容器仍running、非OOM，前置65项固定镜像测试通过；四rank均
在zero-based step9执行更新。完整256恢复概率已独立重算，最大差0，argmax变化0。

基座完整selection结果：154/9,226正确，accuracy 1.6691957511%，fixed60 macro F1
0.0547263682%，supported macro recall 1.6666666667%，失败/无效0。全部预测为
qa_definition；这表示当前冻结Final60读出未获得有效任务判别能力，不可推断基座
没有声学信息。该结果属于TRAIN内部选择集，非官方TEST、非业内学术基线复现。
固定镜像30项计分测试、实际容器/输入/策略身份认证及独立原概率计数复算均通过。
保持原Stage1全量适配配置，不根据此结果使用TEST或变更标签。

零训练结果已归档为 `experiments/slurp-simple-base-selection-v1.md/json`，包括实际
容器/配置/预测hash、完整指标和单类塌缩诊断。不同音频的候选概率向量各不相同，
平均top概率0.5563，不能将统一argmax解释为完全忽略音频。

终态checkpoint验收扩展回归已实际exit0：固定镜像85测试通过，真实前缀112个
LoRA tensor、四rank master/Adam、持久提交和CPU PEFT重载均通过，adapter身份
与原验收一致。该回归仍仅覆盖真实前两步，不代替未来634步终态审计。

完整selection概率验收扩展：绑定实际训练终态634/cursor(2,0)、相同runtime profile，
逐条重算全部9,226请求，11项本地测试通过；计分相关30项通过。新增独立终态
评测汇总入口 `finalize_slurp_simple_selection.py`，写入新审计目录、推理证据只读。
报告HF/vLLM实际argmax变化和指标差，不以概率误差门槛强行断言预测完全相同。

`simple_sampled_rollout`补齐无标签G16采样/固定参考绑定接口。请求/策略/更新/slot
绑定独立随机种子，有放回采样、不按gold筛组；固定参考要求相同音频/任务/支持集。
10项接口本地测试通过；与采样loss合计16通过1跳过（本地缺verl）。固定镜像联合
测试已派发，待观察。尚未接入真实音频vLLM rollout→DP4更新生命周期，不新增RL
效果结论，不改变正在运行的Stage1快照。

2026-11-13 到期检查：四rank均在zero-based18更新，container running、非OOM。
从不可变已完成update日志独立收取3–18共64条rank记录，完整性、real128分母、
全零RL贡献、CE逐micro贡献与native pg_loss相符。平均单步61.20秒、峰值16.24GiB；
加权CE从第3步7.4602到第18步5.6140，不同批次/权重，不能据此声称泛化收益。

RL actor采样衔接：新增actor_item核对CPU张量身份/位置/候选/策略后再附加targets，
监督不参与采样。18项本地测试通过，包括改gold采样不变、padding slot唯一、
错音频/支持集/位置/过期策略拒绝。固定镜像四rank probe扩展可选rollout-bridge，
使用完整60候选模型概率真实G16随机采样，联合GRPO/RLOO+CE+KL与Adam恢复；
新快照已派发，待终态。保留旧固定动作测试作为独立梯度控制；新分支不使用其动作。
该验证仍合成FP32 CPU，实际vLLM/NPU音频生命周期及训练收益均未验收。

采样批次衔接补齐：actor_batch按完整当前策略/固定参考的唯一请求集合匹配，拒绝
缺失、重复、额外、HF替代或过期策略；保留全部slot，padding权重0且不计真实分母。
28项本地接口测试通过。真实音频传输回放容器已派发：重编译原256已验收音频，
严格核CPU张量/身份，消费实际vLLM候选概率，G16采样后构造verl DataProto并独立
核GRPO/RLOO优势。仅复用前缀工程证据，不更新参数、不用其代替正式Stage1参考，
不占NPU，终态待观察。

固定镜像后续联合测试已实际exit0：58项通过，含11项完整选择集概率验收、10项
采样身份、30项计分/输入和7项采样损失测试。该快照早于actor_item/actor_batch，
不覆盖后续扩展。完整Stage1评测独立验收/计分启动器已补齐，但因训练尚未终态
不派发；要求原推理容器成功，所有推理与训练证据只读，从新目录输出验收结果。

简单RL候选预算已补入配置：全40,513 learning、两轮634更新、global128/DP4，
从新stage游标0开始、重置optimizer但加载独立验收Stage1终态权重。评测点预先定为
0/64/128/192/256/317/384/448/512/576/634，每点完整9,226 selection。KL参考固定
Stage1终态不刷新；仍是candidate，training_dispatch_allowed=false，待真实音频
生命周期验收。此安排不新增RL与SFT胜负对照，也不以小先导替代全量训练。

采样桥接四rank CPU任务已退出0，独立核对GRPO/RLOO及instruction回归各24条rank-step，
包括micro1/2、3次更新、固定参考KL、真实请求归约和Adam恢复。G16全60类采样收据在
四rank逐项一致、slot/版本/参考绑定通过；详见.local/simple-final-v1/sampled-bridge-verified.json。
仅合成FP32 CPU；真实音频transport与NPU生命周期仍单独验收，不据此派发RL。

2026-11-13新增授权：175物理0–7全部可用，0–3实查健康且无运行进程；4–7继续原Stage1。
推理入口支持显式单卡0–7，严格检查容器设备集合，18项本地测试通过；新0卡prefix256
HF/vLLM复验使用独立快照，尚未验收。真实音频CPU传输回放已exit0，固定镜像28测试通过；
256请求/4096采样、GRPO与RLOO各36混合奖励组、220全错组，原始前缀权重仅工程fixture。
不据此宣布新音频RL更新或效果提升。NPU RL生命周期仍未验收。

用户要求收敛任务→工程→训练，当前统一契约收口到docs/research/23-task-engineering-rl-baseline-lock.md顶部。
现4–7卡全量Stage1继续原快照；global256/micro2仅独立效率候选，未切换或派发新全量训练。
实际完整batch256日程已生成并与原日程逐真实slot比对一致：40513×2=81026次、每轮159步。
候选入口支持global128/256、micro1/2/4；分阶段耗时与allocated/reserved记录已补充，
可选checkpoint边界冻结权重hash检查，默认every_update不变；这些新训练代码尚无NPU验收。

2026-11-13 14:12 UTC新卡验收：NPU0实际HF/vLLM256请求exit0，固定镜像32测试通过；
独立逐项重算概率maxlogp9.5367431640625e-7、argmax0变化。证据npu0-parity-verified.json，
仅原已验收前缀checkpoint、不是RL更新或新策略全量验收。

隔离吞吐对照global128/micro1已在175物理0–3卡启动，容器
4a98585746b4f2ad48fae6cd4432b773bf50b12e428e0d279d4cf976f440e875。
使用完整learning日程前两步仅工程容量/吞吐；checkpoint_interval2、frozen_audit=checkpoint，
不推广到现有全量Stage1。后续同global128/micro2比较，global256是否运行取决于实测。
配置见configs/experiments/slurp-simple-throughput-candidate-v1.json；端到端吞吐汇总13测试通过，
按每步四rank最长耗时和真实请求数统计，包含准备/审计/保存、排除启动与外部评测。
独立checkpoint审计入口已支持显式batch-probe，新运行尚无结果或提速结论。

完整日程RL输入接口新增simple_audio_batch：固定完整128/256 batch，逐audio hash核对，
直接音频Final60编译一次，actor/行为renderer/固定参考renderer复用同一CPU张量。
只在批内复用padding输入，明确真实分母与全人口类别权重；targets不进入模型身份。
41项相关本地测试通过。固定镜像scheduled-audio-transport新CPU任务已启动，
会将原两批256个新编译输入与原已接受音频/vLLM证据逐项比对，再构造GRPO/RLOO传输。
不是新的NPU rollout/更新，结果待观察。

吞吐global128/micro1完成后新增同输入/同micro控制：独立checkpoint审计后比较原前缀
四rank所有FP32 master、Adam及BF16导出，差异不隐藏于BF16量化；16相关测试通过。
原检查节奏/设备变更与microbatch改变分别验证，不直接用吞吐数字宣称训练等价。

2026-11-13 14:22 UTC吞吐基准global128/micro1实际exit0：两步完整256真实请求，
按逐步四rank最长整步耗时合计136.0566秒、1.88157请求/秒；更新63.43/59.85秒，
准备5.56/6.00秒，第二步审计0.50秒、checkpoint0.79秒。allocated峰值15.06GiB，
reserved34.86GiB。主要开销在模型更新，不能宣称减少hash检查已明显提速。
原始结果和独立汇总在.local/simple-final-v1/batch-g128-m1-{raw,summary}.json。
同输入micro2对照已在0–3卡派发；CPU独立checkpoint/Adam/PEFT及与原前缀精确状态
比较已派发batch-g128-m1-audit，均待到期观察。当前全量Stage1快照不变。

2026-11-13 14:28 UTC完整日程音频编译已exit0，固定镜像41项测试通过；两批128的
256输入与原验收artifact逐项一致，本地独立对照原日程全部权重/id/padding一致。
原vLLM概率→G16→native DataProto、GRPO/RLOO优势已复算；scheduled-audio-transport-verified.json。
无新NPU RL更新。新probe_simple_sampled_npu复用训练入口callback：重现前缀精确master/Adam，
执行一个完整128音频GRPO/RLOO+CE+KL更新，保存/恢复Adam并检查驻留概率，导出后保持
awaiting_acceptance；111项相关接口测试通过，探针自身尚未运行。正式Stage1参考与
全生命周期验收仍未取得。修正global256候选入口遗漏的固定128检查，未修改任何运行快照。

## 本地容量检查（2026-11-13 18:59 UTC）

175的/data当前可用约3.42TB（十进制）、32.9%空间和99.95% inode。
按已闭合Stage1更新256与RL更新2实际文件计，四份Adam/master加LoRA各约8.78MB。
Stage1真实运行配置每64步及epoch/终态保存，完成277后尚有7个计划checkpoint，
对应张量约61.48MB；Stage2即使634步各保留一份，张量外推约5.57GB。
这些估计不含概率/输入缓存/评测/其他任务，不声称完整预算或空间预留。
部分root缓存无法由主机用户du读取，未用不完整du汇总作结论；没有删除任何运行产物。
测量与限制见`slurp-simple-storage-capacity-v1.json`。

### 2026-11-14 全量日程验收与组件隔离

class-spread processor `f9af7340`成功退出0/non-OOM；独立核验325项依赖、40,513条唯一
输入及81,026条使用记录的完整日程坐标，最大2249 tokens。第一轮真实处理音频，第二轮
复用同输入结果并生成新坐标；不是两遍独立音频处理。未修改的`load_full_schedule`接受
实际收据，加载634更新、每epoch317。见[processor收据](../../../slurp-class-spread-processor-v1.json)
和[正式入口消费验证](../../../slurp-class-spread-training-consumer-v1.json)。新日程未启动训练。

原HF/vLLM重跑`19e09d4e`成功，256条完整候选最大logprob差9.53674e-7。
投影与attention单独干预均已成功执行并独立收集：相对原HF，投影最大差0.5995884、
9/256条argmax变化；attention最大差0.4344473、14/256条变化。两者都需对齐，不能将
全部偏移归于一种组件或把各自误差线性相加。见[组件对照](../../../slurp-batch-updated-components-v1.json)。

新增独立`span_batch_rollout_candidate`，为vLLM TP1单条完整prefill调用与候选actor
相同的TND sparse3/inner2 attention；保留原vLLM权重、LoRA、QKV及输出投影。
布局/投影流/拒绝非法TP及packed核心共9项CPU测试通过。不是生产profile，也未修复投影。
真实256输入attention-only HF/vLLM对照`c517e200`已在确认空闲的175卡5启动；
exact docker wait session42599，普通检查不早于launch+600。完成后独立收集
`collect_batch_updated_backend.py --tag batch-updated-attention-vllm-v1`，再运行
`compare_batch_updated_followups.py --kind attention_vllm`。正式训练仍未准入，原rc2不改。

### 2026-11-14 attention跨后端通过，推进完整候选

`c517e200`成功退出0/non-OOM，独立核验326项依赖及全部输入身份；attention-only HF
与采用同一TND sparse3/inner2内核的vLLM在256条上完整候选概率全部通过，最大差
9.536743e-7。该比较的参考是attention-only HF，不能误读为原eager HF。
见[attention对齐收据](../../../slurp-batch-updated-attention-vllm-v1-parity.json)。

新增独立`span_fixed_rollout_projection`：使用vLLM当前权重，对音频Linear、拆分音频
QKV、语言基础QKV/MLP/o_proj及词表头执行与actor相同的128行×8 tile投影；
原vLLM LoRA选择与执行保留，尚未声明LoRA算子等价。偏置/返回结构、非等宽QKV、
原生LoRA调用、MLP/输出头及已有核心共29项CPU测试通过。
完整候选HF256对照`af376351`确认175卡5空闲后已running；exact wait session76247，
普通launch+600。完成后collect_batch_updated_backend.py --tag batch-updated-fixed-base-vllm-v1，
再compare_batch_updated_followups.py --kind fixed_base_vllm。冻结rc2原1360文件hash全相同；
正式训练仍false，DP4/global128/长期RL收益与跨任务迁移尚未完成。

### 2026-11-14 完整候选跨后端通过，DP4新日程前缀启动

`af376351`成功退出0/non-OOM，独立收集及完整输入身份比较通过：固定基础投影、音频
投影、输出头加packed attention、保留原生vLLM LoRA，与full candidate HF256最大
logprob差9.536743e-7。见[完整候选后端收据](../../../slurp-batch-updated-fixed-base-vllm-v1-parity.json)。
这仅证明本次更新后adapter及256输入，尚非所有checkpoint/多请求vLLM/正式RL准入。

独立候选入口`tools/jev/train_slurp_batched_instruction_candidate.py`复用原生verl/FSDP
及既有loss、导出/本地存储机制；安装候选计算profile并使用真实批量前向，保持global128
成员、标签和权重，以真实长度整体重排请求载荷。新batch配置只允许micro8/16；
配置/零权重载荷/顺序可逆及既有长度规划共21项CPU测试通过。该入口当前明确拒绝继续
训练，以免借原生产lineage声称候选已通过恢复；需独立新恢复验收后开放。

`class-spread-batch-dp4-prefix-v1`在确认175物理4–7均空闲后启动，容器
`4944f7980c428e7fe3172ab801d124e3ab3d12f2d8ec5f36c384b33209b87f1e`。
BF16执行/FP32 master、LoRA q/v r4 alpha8、lr1e-4、seed317、global128、micro8×累积4，
读取已验收完整40513×2日程（634更新），本次仅执行新初始化的前两次更新并导出step2。
未复用旧日程游标，未挤占0–3卡其他项目。全部热写/data，Ceph只读，fsync保留。
exact docker wait session13393；普通检查不早于launch+600。完成后运行
`python3 .local/simple-final-v1/collect_class_spread_batch_prefix.py --tag class-spread-batch-dp4-prefix-v1`，
再独立核对日程/四rank state/BF16导出及同参数HF/vLLM；收集本身不代表这些验收已过。
原rc2的1360文件hash不变，正式全量训练与RL尚未派发，总目标未完成。

### 2026-11-14 DP4前缀、checkpoint与新权重后端验收

新class-spread global128/micro8前缀`4944f798`成功exit0/non-OOM，四rank各完成两步。
第二步实际更新约13.78–13.83秒，峰值allocated约27.42–27.68GiB；含准备/审计/checkpoint
总步时约21.23秒。见[实际计时](../../../slurp-class-spread-batch-dp4-throughput-v1.json)。两步含冷启动，
不作为稳态吞吐或RL收益。全部256真实样本来自新全量日程的前两个batch。

独立CPU审计`dcb00ebc`exit0，实际音频processor重算长度及完整载荷排列、四rank112LoRA
FP32 master/Adam结构及游标、BF16导出、实际PEFT重载和durable commit全部通过。
报告与24项产物证据图再次hash复核。adapter identity
`1b2b661483ceae5f78f40c2eaa179b861b4fa24cf6012668b5a00cc7ddb8e113`，
见[checkpoint收据](../../../slurp-class-spread-batch-dp4-checkpoint-v1.json)。

新权重HF/vLLM256任务`eba03a0e`成功，独立原始概率复算max9.536743e-7/argmax0；
CPU证据链审计`cf598711`exit0并完整graph复核，通过新权重、输入、checkpoint关联。
见[后端收据](../../../slurp-class-spread-batch-dp4-parity-v1.json)。不是正式RL/全量效果验收。

新增候选独立恢复lineage及同参数驻留单条/micro8验证，配置/游标/原生恢复34项CPU测试过。
恢复启动器首次仅本地SSH argv过长失败，尚无远程运行；改为stdin传输预检脚本后，确认
4–7卡空闲并启动`0cd237c72a1c0cb4e640c9833bea568add9f2caaf5194fe5d903d9d82dddf454`。
从已认证step2精确恢复FP32 master+Adam，先256输入分rank单条/批量对照，再新日程step3/4。
普通检查launch+600；确切docker wait句柄见当前会话。完成后用
`collect_class_spread_batch_prefix.py --tag class-spread-batch-dp4-resume-v1`收取，独立核验恢复和
新endpoint；不能由启动推定恢复通过。原rc21360文件hash不变，正式全量/RL未派发。

### 2026-11-14 精确恢复通过并启动全量micro8训练

恢复任务`0cd237c7`成功退出0，四rank参数与Adam精确恢复；同step2的单条/micro8共512次
完整概率对照由独立CPU重算通过，max9.536743e-7；新日程update3/4完成。step4更新
约14.94–14.96秒，总步时22.45–22.47秒，峰值allocated最大28.30GiB，仍非稳态。
独立endpoint审计`0d868b60`通过32项证据图、真实processor排列、master/export/PEFT。
step4 adapter `7c33b29b0862bca1027cfb6b315b7c4dd8c2baa8b80d415898bc2d2fde3974ce`。
HF/vLLM256 `b19bab49`成功，独立审计`41607d09`成功并重新hash369项证据，max9.536743e-7，
argmax0。见[完整step4准入](../../../slurp-class-spread-batch-dp4-resume-parity-audit-v1-accepted.json)。

恢复/批量配置/长度/固定投影/attention/原生Adam恢复等96项相关CPU测试通过。
新增训练指标聚合按各micro贡献求和再DP平均，不再把微批均值重复平均；实际四步加权CE
为6.3165、6.2986、5.5609、5.2211，各步样本不同，不能视为固定集改善或RL收益。
见[训练指标](../../../slurp-class-spread-batch-training-metrics-v1.json)。

确认175物理4–7空闲后，全量接续已启动：
`934b106a6563f87f29e343f66b10e75485fd7c1b5de93864f920e00f02b931df`，
tag `class-spread-batch-dp4-full-v1`，从已验收step4继续update5..634（630更新）。
固定global128/micro8×累积4、两epoch完整40513学习录音、BF16/FP32 master、LoRA r4/alpha8、
lr1e-4、seed317；同global内部长度组批和类别时序分散日程，checkpoint间隔64及epoch/终态。
每次恢复先做同参数单条/批量概率检查；0–3其他项目不动。配置
`configs/experiments/slurp-class-spread-batch-stage1-v1.json`记录本次实际派发。

普通检查不早于launch+600，exact wait句柄见当前会话。完成后collect_class_spread_batch_prefix.py
--tag class-spread-batch-dp4-full-v1，随后launch_class_spread_batch_dp4_full_audit_v1.py；
完整selection入口launch_class_spread_batch_dp4_selection_v1.py已准备但必须等terminal634和独立
checkpoint验收才可运行。尚无新全量训练完成/selection/RL/迁移效果。原rc21360hash不变。
