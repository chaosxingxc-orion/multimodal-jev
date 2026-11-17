# context-prob-v1：任务、工程与算法的权威入口

本入口取代历史Final60分类/CE/rc1/rc2文档对当前概率头实验的执行指导；历史证据保留。
冻结登记：`configs/releases/context-prob-v1/LOCK.json`、`experiment.json`、`algorithms.json`。
锁包含原已验收356个训练源码文件、两份配置、镜像digest、模型/数据/日程身份及5份证据。
状态是Stage1局部执行准入、Stage2尚未发布，不能将此登记称为完整RL工程稳定版。
任务变化先改设计与数据协议，再发布对应工程配置，最后训练；不边训练边改同一条曲线。

## 2026-11-15 最新收口状态（北京时间）

最新终态：Stage1终点selection完整9226已收集，通过原始预测独立复算：
set-F1=73.8041%、top1 accuracy=82.0724%、macro-F1=77.4315%。
原收集器仅NLL/Brier浮点末位严格相等失败；独立v3限定这两项8 binary64 ULP，
保留身份/概率/计数精确比较，原模型不重跑。真实连续两步RL及恢复重放已收集；
初始化selection和checkpoint频率仍待。正式634步RL仍未发布。
当前selection574594、sequence574965（v3，507pins）、comparison v5（510pins）；
旧失败队列保留。最新详情见`context-metric-audit-repair-v1.json`与研究执行决议。
以下远程运行中观察已被上述终态覆盖，不再观察已退出的selection容器。

本次最新到期观察remote1794685890.445376：原selection v2容器
`63e9abb7ac60` running/nonOOM、8576/9226、推理2137.79秒，
下一次普通观察>=1794686490.445376。364冻结身份、498序列pins、500比较pins
全部通过；完整selection及正式RL收益仍待。任务→工程冻结→训练的执行范围、
变更规则和算法适配分析已收口到
`docs/research/context-baseline-execution-decision-20261114.md`。
本轮不修改运行包，不追加简单任务发布条件。以下记录为此前观察。

后续到期观察remote1794684604.146284：原selection v2容器63e9abb7ac60
running/nonOOM，完整选择集3136/9226，推理计时837.20秒。下一次普通观察
不早于1794685204.146284。六队列本地ps仍存活，未重启原任务。
尚无完整选择集指标或正式RL收益。FSC双策略编排仅CPU候选，独立于当前准入。

当前状态更新：Stage1 完整634步已独立收集，原容器04dd31b748d4 exit0/nonOOM，
于北京时间02:41:57结束。正常614步平均17.54秒，约7.29 query/s。
它是ranking+binary NLL训练，不是正式RL；新概率头完整selection尚待。
初次评测256条HF/vLLM概率、输入及位置完全一致，但全量阶段因遗漏只读发布
依赖而失败。另两次RL启动失败为IPC路径过长和绕过Ascend入口导致缺acl。
独立修复短IPC、镜像入口及9项精确只读挂载，未改数据、损失或训练参数。
真实固定镜像CPU父/子acl导入及9226条加载通过，新版编排45项CPU测试通过。
证据：`context-runtime-launch-repairs-v1.json`；连续NPU及正式RL收益仍未验收。

本次364项冻结身份、498项序列pins和500项比较pins通过。当前六驱动本地存活：
selection548477、continuous549655、replay549657、checkpoint549658、
sequence550847、comparison550848；旧518938/533061已因真实前序失败退出。
最近远程1794683960.6309571确认selection v2容器63e9abb7ac60 running/nonOOM；
后续观察脚本NameError未取得完成条数，不是任务失败。下次普通观察
不早于1794684560.6309571。现有链成功后按共同起点运行GRPO和RLOO-clipped，
不再增加任务/算法前置条件。以下记录保留作历史，当前以上述状态为准。

本次收口复核：364项冻结身份及469项比较链pins均通过；原训练/验收、正式序列518938
与比较v2驱动533061本地存活，均不重复启动。最近已保存远程观察为
1794681437.851721：Stage1四rank618/634、running/nonOOM，最近最慢步17.6378秒；
下一次普通远程观察不早于1794682037.851721。本次仅读取既有观察，不冒称新实时检查。
Stage2仍未发布，无正式连续RL收益。比较器正式收集路径已改为evaluation/selection，
38项CPU测试通过；旧等待驱动528969已退出并归档，由533061替代，原训练序列未改。
证据为context-rl-curve-comparison-cpu-v2.json。

以下观察及驱动编号为历史记录，以上述复核为准。

最近到期观察remote1794680779.038841：Stage1四rank581/634 running/nonOOM，
最新最慢17.634秒；下次普通观察>=1794681379.038841。十个原/新驱动本地ps存活。

新增独立只读比较`compare_context_rl_curves.py`，重跑实际完整训练审核和11点曲线
审核，绑定相同配置、源码/初始资产、Stage1 reference及step0逐条概率；完整报告
匹配差值、最终/选中点增益、退化和成本。共享service不重复算物理卡，selection
选点不称TEST或统计显著。26新+12既有共38CPUtests/2.11秒通过，模型及实际训练
审核在集成夹具用替身；目前无真实RL比较结果。证据`context-rl-curve-comparison-cpu-v1.json`。
driver528969等待原518938双算法曲线成功后只读比较，469文件身份锁定；不派发模型，
不修改原364锁/467序列pins，不新增正式RL前置条件。

2026-11-15 正式实验编排固定镜像CPU v2独立收集：526707c59bf0 exit0/nonOOM，118 tests/36.59秒通过。v1的93通过16失败来自测试包缺动态导入夹具test_context_rl_evaluation_plan.py；补齐夹具后复验，生产数值代码未改。完整训练→收集→audit→11点评测→曲线候选已接。新增run_context_baseline_sequence.py仅串联既有入口，7队列+14编排tests共21通过/2.11秒（实际执行替身）；467文件pins、driver518938等待原Stage1选择链及checkpoint链，成功后共同Stage1 source→GRPO全曲线→RLOO全曲线；任何失败保留句柄停止，不重启原队列。没有实际正式包/正式RL派发。364锁与原369/4/5/10 pins无变。到期remote1794679856.1223068四rank528/634 running/nonOOM，最新最慢17.155秒；下次普通>=1794680456.1223068。详见context-baseline-sequence-cpu-v1.json、context-formal-experiment-cpu-v2.json及runbook。

以下为此前阶段记录，当前状态以上述记录为准。

最新到期remote1794678214.0578518：原Stage1四rank435/634 running/nonOOM，
最新最慢17.466秒；下一次普通观察>=1794678814.0578518。原八驱动仍存活。

正式包发布器与启动器候选已接上完整Stage1起点重验、三次NPU证据重验、同容器
runner和原设备租约。默认本地校验，显式launch才创建；attempt/准确句柄预存，
观察失败保留句柄和锁。新包/初始策略只读，学习cache父目录ro而cache叶rw。
48本地tests过；固定镜像CPU容器7d60454aef01 exit0/nonOOM独立收集，79tests
全部通过/35.89秒。见`context-formal-package-launch-cpu-v1.json`和
`context-formal-publication-cpu-v1.json`。完整来源/NPU准入与SSH使用替身，尚无
真实完整来源包、正式发布或正式RL容器。接入说明`context-formal-training-runbook-v1.md`；
完整长任务中断后的恢复调度尚未发布，不能凭两步恢复证据称已具备该入口。

正式入口候选由`prepare_context_formal_runtime.py`从精确checkpoint候选生成，
仅替换两步守卫为独立`context_formal_runtime.validate`，损失/采样/精度/逐步策略
激活代码不变；原排队包不动。运行时要求正式发布文件、精确配置和完整Stage1共同
起点，不能只改dispatch开关。`context_formal_acceptance.py`绑定三次独立成功
NPU收集，重新执行恢复与checkpoint频率比较；该入口尚无真实验收输入。
32项及44项相关CPUtests通过（两组有重叠；验收编排使用替身）。证据
`context-formal-runtime-candidate-cpu-v1.json`。独立候选位于
`.local/decision-objective-research/context-formal-runtime-candidate-v1`，尚未发布，
正式package发布器/launcher及完整起点接入仍待，不声明已能启动完整RL。

最新到期观察remote1794677542.2843072：原Stage1四rank396/634，running/nonOOM；
最近步最慢17.748秒。下次普通观察不早于1794678142.2843072。当前完整终点、
选择集及原排队连续/恢复验收仍待，不改运行包或重新派发。

Stage2共同起点打包候选`tools/jev/prepare_context_stage2_source.py`已接完整来源绑定，
复制原Stage1 policy/processor并针对原policy ID重新派生native adapter；完整终点
及selection会重新核验，不能使用已写出的自由binding JSON代替。任务阶段重新建
optimizer，不复制Stage1 Adam。22CPU tests通过（真实小型BF16资产及safetensors，
打包测试中完整来源准入为替身）；不是实际634终点包或正式RL发布。
证据`context-stage2-source-package-cpu-v1.json`。原369/4/5/10四组pins均无变。

固定镜像无NPU CPU回归v3 c060a78631cd已exit0/nonOOM，独立收集确认134 tests
全部通过，用时36.51秒；证据`context-rl-training-audit-cpu-v3.json`。
前次v2 a5ef6931abed为116 passed、4 failed、14 errors：CPU测试夹具的AdamW
默认foreach/fused自动探测进入torch_npu设备初始化，在无NPU容器内失败。
夹具改为显式foreach=False、fused=False，与生产trainer既有设置一致；未改冻结
运行代码或排队包。该回归覆盖训练审核、收集、奖励组、起点绑定、checkpoint与
rollout候选工具；部分模型/SSH为替身，不代表NPU连续训练或正式RL效果通过。
原364项身份锁复验通过，Stage2 released=false；八个原训练/验收驱动本地仍live。
最近普通观察仍为下述361/634，本轮未提前轮询，后续按既定600秒间隔执行。

## 2026-11-14 历史收口记录

remote1794676923.4982142到期Stage1四rank361/634 running/nonOOM；下次普通
>=1794677523.4982142。完整634审核编排10新tests加既有25项共35CPU过，两算法/
81026曝光/10保存点及末步拒绝路径覆盖；transport/数据/模型/组核验等在编排测试用
替身，不是全模型训练。证据context-rl-training-full-orchestration-cpu-v1.json。
当时固定镜像无NPU CPU回归容器a5ef6931abed已启动；其失败和v3修复后结果见上。
原364及四组排队pins保持，八原驱动及新collector经本地ps确认live。

新增collect_context_rl_training.py正式训练transport候选：准确终态句柄、只读
volumes-from且无NPU设备，归档run/service/supervision/reference-cache与执行配置，
流式hash避免整包读入内存，记录bad pins/package/启动绑定，失败产物保留后拒绝。
audit入口要求完成且无漂移的transport。35CPU tests通过，SSH/docker替身、实际
reader子进程/tar/hash/解包；首次失败为测试夹具check_output调用已mock的run，
修正夹具后通过。证据context-rl-training-transport-cpu-v1.json。无真实远程收集。
原364及四组排队pins保持，原八驱动存活；最近remote326/634，下次仍>=1794676918.8473048。

新增audit_context_rl_training.py离线正式634步候选入口：核对固定镜像终态/配置、
监督器正常退出、两轮全人口日程、每步targets与GRPO/RLOO组核验、策略/adapter及
marker、四rank训推parity、间隔checkpoint的FP32/Adam与BF16执行张量。
45项CPU tests通过，覆盖组成模块/数据拒绝/实际CPU Adam状态验证；未完整执行
634步审核编排。另读取既有真实Stage1 step2四分片及128执行张量通过新helper，
不冒称正式RL来源。证据context-rl-training-audit-cpu-v1.json及
context-rl-training-checkpoint-real-v1.json；正式transport collector/发布接入仍待。
原364和四组pins保持，八驱动live；最近remote326/634，下次普通仍>=1794676918.8473048。

remote1794676318.8473048到期核验Stage1四rank326/634、running/nonOOM；
下一次普通>=1794676918.8473048。八驱动本地ps live，原364及四组队列pins保持。
新增context_rl_group_audit.py独立候选支持GRPO/RLOO：完整G16集合奖励、优势、
原始概率采样重放、联合loglikelihood及真实/padding权重核验；零信号真实组保留。
初始32tests过、增加恒定奖励后独立模块15tests过。真实历史每算法128条音频groups
独立复算通过：GRPO优势零误差、RLOO最大1.1176e-8；历史采样种子协议不同，
不冒称在线采样验收。见context-rl-group-audit-cpu-v1.json、context-rl-group-real-check-v1.json。
尚未接正式634 collector，原排队GRPO collector不改，未声明连续RL收益。

新增离线context_stage2_source.py候选绑定完整Stage1 step634与其已收集9226选择集：
终态/全日程/commit/四分片/策略资产、256精确parity及全人口记录绑定，再实际核验
训练与评测张量/processor/native adapter。固定reference和两算法初始化相同，不选
中间检查点、不开启正式训练。12新绑定tests+29评测入口共41CPU通过（模型用替身）；
真实既有step2导出张量核验通过，错误step1拒绝，仅证明认证函数、不冒称全634完成。
见context-stage2-source-cpu-v1.json及context-stage2-source-tensor-check-v1.json。
原364和四组排队pins保持，八驱动ps live；本轮无提前remote观察，最近291/634，
下次普通仍>=1794676289.6870782。该候选尚未被正式发布器或完整终点调用。

remote1794675689.6870782到期观察原Stage1四rank291/634、running/nonOOM；
下次普通>=1794676289.6870782。独立checkpoint候选接续driver483225经ps确认live，
等待原resume chain锁；成功重验原连续/恢复后才同设备租约派发两步→独立collect→
与interval1逐步策略/采样及末步四rank master/Adam精确比较。新10文件pins封存勿改。
48项CPU tests过；launcher prepare-only与AST通过，collector采样/标签/奖励/优势
核验块逐字保留，适配间隔恢复和每步策略验证。尚无候选远程容器、终态collect或
NPU比较结果；不声明launcher/collector完整运行成功。见context-checkpoint-queue-v1.json。
原364/369/4/5保持，原七驱动live；Stage2未发布。

独立prepare_context_checkpoint_package.py已封存372文件checkpoint同容器候选，
manifest87a141b7；仅trainer与两份重定位/pins配置变化，原runner/算法/数据/种子不变。
原两步守卫保留，checkpoint_interval64→只存step2、仍发布step1/2；不启动。
37项CPU测试通过，含每步保存与间隔保存两臂逐步权重及末步模拟四rank Adam精确相等，
设备/通信/导出/激活为替身。证据context-checkpoint-package-v1.json；原连续及恢复
验收通过后仍需版本化launcher/collector和真实NPU比较，不能直接正式训练。
364/369/4/5 pins保持，七驱动本地ps live，最近远程252/634；本轮未提前观察，
下次普通仍>=1794675608.572515。

新增独立`audit_context_checkpoint_lifecycle.py`候选核验收集产物：每步manifest/
资产hash/引擎激活回执/四rank记录，间隔及段末四分片与execution/commit绑定；
缺失、错配、多余恢复文件或软链接拒绝。14新测试加既有10项共24CPU通过；
实际变换保存块使用CPU模型/Adam与FSDP/通信/导出替身，引擎回执模拟。
证据context-checkpoint-lifecycle-cpu-v1.json；该核验不替代张量、优化器内容、
采样或容器终态审核，未接入原排队collector，未实际NPU运行。原364/369/4/5
pins保持，七驱动本地ps live，最近远程252/634，下次仍>=1794675608.572515。

remote1794675008.572515到期观察：Stage1四rank完成252/634，原04dd31b748d4
running/nonOOM；下一次普通观察>=1794675608.572515。七个原驱动本地live，
364冻结身份及369/4/5三组队列pins通过；不重启或修改原运行包。
独立checkpoint候选保留每步策略导出/激活，只按间隔及段末保存完整Adam/master。
10项CPU测试通过（实际变换后的保存块，FSDP/通信/导出替身及精确恢复比较），
工程两步守卫仍保留；无NPU验证、正式准入或自动清理。
见context-checkpoint-runtime-cpu-v1.json；Stage2仍未发布。

固定镜像CPU评测工具回归f87bcc176e14 exactwait退出0，独立collect确认nonOOM、
physical_devices=[]、源码/runner hash一致，101tests全过/146.73秒。
证据context-rl-evaluation-cpu-v1.json，覆盖prepare/package/模拟launcher/collect/
audit/curve；模型及SSH部分用替身，非NPU实际评测或正式RL准入。
prepare新增诊断开关拒绝，当前Stage1模板没有这些开关；原364和三组排队pins不改。
本轮未提前观察Stage1，下次普通仍remote>=1794674944.0114124，最近214/634。

remote1794674344.0114124原Stage1四rank完成214/634，04dd31b748d4 running/nonOOM，
下次普通>=1794674944.0114124。七个原驱动本地ps live，冻结364及三组队列pins保持。
新增prepare_context_rl_evaluation_package.py/launch_context_rl_evaluation.py候选，
正式完整来源检查后独立封存单点策略/source/evidence及配置，包内路径迁移，默认
无远程调用。显式launch使用原租约、7卡和固定镜像，保存attempt及准确句柄；wait
连接失败不重建并保持租约。12新测试+23入口+9计划共44CPU过，含模拟SSH与实际flock。
未创建真实正式RL包、未远程派发或证明NPU运行；证据context-rl-evaluation-launch-cpu-v1.json。
现有prepare→launch→collect→audit→curve候选入口齐备，仍需真实产物和运行验收。

新增`audit_context_rl_curve.py`候选将单点独立审核接到曲线汇总，共同正式训练来源/
算法/experiment只供给一次，逐点重验产物，拒绝混用运行与缺点乱序，前缀不选模型。
保留source及inference身份，不改写预测policy ID。16新测试加既有21项共37CPU通过，
本组逐点审核用替身，无实际完整RL曲线；见context-rl-curve-audit-cpu-v1.json。
原364及三组排队pins不改，七驱动本地ps live；本轮未提前远程观察，下次仍
remote>=1794674314.053779。既有178步记录最近32常规步均17.562秒、7.289query/s，
仅live窗口，不是终态成本。两步候选每步写Adam且拒绝正式运行，正式634步版本
须区分每步行为策略刷新与预定checkpoint保存；说明见context-rl-selection-runbook-v1.md。

remote1794673714.053779到期核验原Stage1四rank完成178/634，04dd31b748d4
running/nonOOM，下次普通>=1794674314.053779。七个原训练/验收驱动本地ps存活。
新增独立候选collect_context_rl_selection.py（准确终态只读归档，保留失败）及
audit_context_rl_selection.py（重建正式训练计划、真实张量/processor/adapter、
256parity、9226人口/请求/逐条记录/指标复算）。最新23项CPU测试通过，模型用替身、
完整9226合成记录；原入口/计划/曲线44项不改。新authenticator对既有真实Stage1
step2/resident导出通过，错误step1拒绝，非正式RL或完整selection证据。
见context-rl-selection-audit-cpu-v1.json、context-rl-selection-export-check-v1.json，
使用边界见context-rl-selection-runbook-v1.md。尚未真实远程collect或接入正式launcher/
11点曲线；Stage2仍未发布，364及三组原队列pins保持。

remote1794673103.3507223到期核验原04dd31b748d4 running/nonOOM，四rank均完成
step143/634；下次普通观察>=1794673703.3507223。373142/373144/373145及probe、
连续更新和恢复接续进程均本地ps确认存活；保留原运行链，不重启。

新增独立候选`tools/jev/run_context_rl_selection.py`，消费已完成收集的正式RL与
Stage1来源，重建11点计划后绑定单点真实权重；复用既有HF/vLLM探针和9226选择集入口。
step0明确取Stage1 step634；不改写推理policy ID。执行前校验pins，实际导出后
逐张量/processor/adapter核对，256条精确概率与输入对齐后才进入完整selection。
新增23项绑定/调度/失败退出测试，连同已有21项共44通过；调度测试模拟模型子进程，
不是NPU结果。修正候选提前创建HF输出目录的问题，原探针未改。
见`context-rl-selection-entry-cpu-v1.json`。尚需正式发布包/容器launcher和独立collector
接入；没有实际RL评测、正式Stage2发布或收益。364冻结与三组队列pins均保持。

本次用户要求的执行决议见
[简单任务工程收口与RL实验顺序](../../research/context-baseline-execution-decision-20261114.md)：
一套公共工程、GRPO/RLOO-clipped两项执行配置、PPO-critic仅预留；并行Q不套时序GAE。
364项锁再次通过，本次未修改冻结源码或配置、未重新观察远程状态。
正式Stage2仍须收口连续运行/恢复及634步保存与11点评测调度后封版；不扩展研究前置项。

新增离线候选`tools/jev/context_rl_evaluation_plan.py`，从独立收集绑定完整11点策略：
step0必须是已完成Stage1 instruction step634，后续必须同一个正式task运行，
核对algorithm/reference/发布数据/commit/策略资产。9项绑定拒绝测试加既有12项曲线测试
共21项通过，见`context-rl-evaluation-plan-cpu-v1.json`。仅合成产物验证，尚未接入
实际RL评测launcher，不代表11点推理已完成；原Stage1评测和两个RL条件队列未改。

最新实际观察remote1794672472.4356728：Stage1四rank均完成step108（接续106次更新），
04dd31b748d4 running/nonOOM；下次普通>=1794673072.4356728。原全训/选择链保持。
新增恢复接续driver444235等待同容器两步driver441409成功收集，再绑定step1的
master/Adam/行为策略并保持Stage1 reference，创建新进程重放step2及独立精确比较。
准备器7项绑定/拒绝tests与既有8项重放tests通过；实际运行仍待，未创建恢复包或容器。
见`context-shared-resume-preparation-v2.json`。旧369与两新队列pins均未漂移。

同容器v2条件接续driver441409已经启动，本地ps确认存活，当前
`waiting_original_probe_collector`；尚无远程容器。原probe425576/collector426809
保持，先由其独立证明设备初始化和映射通过，再取得同一租约派发两步工程验收。
新launcher/collector/driver已固定hash，见`context-online-shared-queue-v2.json`；
collector保留原完整采样/奖励/优势/四rank checkpoint审核后缀，并新增同容器终态、
服务退出和候选快照绑定。它尚未执行实际终态收集，不能称验收通过。
两步通过后仍需单独新进程恢复及重放，不自动派发正式Stage2；失败保留句柄不重跑。

remote1794671842.9149876最新实际观察：Stage1四rank完成至step72（本次接续70步），
容器04dd31b748d4 running/nonOOM；下一次普通检查不早于remote1794672442.9149876。
最新32常规步（40–72排除保存步64）均17.664秒、7.246有效query/秒、4096/4096
cache命中；actor均17.016秒，输入等待均.000103秒。运行中数据不含完整job成本，
不代表终态/权重审核或任务收益。原全训与选择链保持。

新增同容器候选包context-online-shared-npu-v2-package仅prepare-only，372文件校验，
原engine仅显式替换device-scope validator import，其余数值源码相同；独立生命周期
监督器和实际CANN映射校验共18项CPUtests通过。见context-online-shared-preparation-v2.json。
未派发/未接入旧失败候选；设备probe通过后仍需版本化launcher、独立collector及实际
连续两步/恢复验收，不据此声称Stage2已发布。冻结364与旧369pins保持。

本次再次核验冻结364项通过，lock ID为
`2ece49811d1e11eabc75f3e95e73ef532bda3fe9f16e9a399da7725ad0d95bfc`。
执行口径固定为一套工程、三类用途配置（评测/Stage1/Stage2），以及Stage2的
GRPO主配置、RLOO-clipped对照配置；PPO-critic仅保留设计，不称为可运行基线。
配置冻结与执行准入分别判断：冻结配置中的`training_dispatch_allowed=false`
不原位改成true；Stage1按已有独立准入证据运行，Stage2仍禁止正式派发。

Stage2只收口三项发布工作，不追加研究功能：

1. 设备拓扑与进程生命周期：验证服务和DP4 actor共存，并保证actor早退时服务正常退出。
   设备小探针通过只证明初始化/小计算，不替代实际模型共驻与连续更新验收。
2. 连续更新与恢复：实际同actor至少两次更新、下一批消费新策略、固定reference不漂移，
   保存后在新进程恢复并重放；LoRA和概率头作为同一个策略版本核验。
3. 完整评测与发布：接入0、64…576、634共11点的全9226评测，绑定各自真实策略，
   封存新Stage2执行配置/源码/证据后才启动正式GRPO；随后同起点同预算运行RLOO-clipped。

任务/奖励或数值路径变更另立版本；不修改当前Stage1曲线，不为算法复制trainer。
复杂Q先完成语义和监督支持，再做Q-input/Q-credit对照，不作为简单主线的发布条件。

最新全量观察remote1794671232.8118825：04dd31b748d4 running/nonOOM，四rank
均完成step38（原step2接续），第11–38共28常规步均17.508秒、7.311有效query/秒，
cache全命中、前台等待均.000099秒、峰allocated37.107GiB。仅live诊断，未达32步
窗口/未终态或权重审核，见context-stage1-live-throughput-v1.json。
下一次普通>=remote1794671832.8118825，原全训/选择链保持。

最新终态优先于以下启动历史：连续RL actor a0198f4b在set_device初始化报
aclInit507899/Resource_Busy，exit1/nonOOM、0更新；专属service已正常close并exit0，
失败独立收集完成，恢复未派发。设备共享拓扑待验证，详见
`context-online-device-init-failure-v1.md`，不认定镜像损坏或RL完成。
原Stage1全量接续04dd31b748d4已启动，remote1794670592.483905实际四rank通过
设备初始化进入load，rank0 prepare_full_schedule；原step2→634/global128/micro8/
本地完整cache配置保持，普通下次>=1794671192.483905。选择链等待其完整终态。
旧分类分支的全9226选择集已独立复算：accuracy82.3651%、macroF1 79.5706%，
HF/vLLM零argmax差，见`slurp-categorical-full-selection-final-v1.json`；
它不是当前概率头或RL效果，新概率头完整成绩仍待。

本次本地复核：冻结364项全部通过，连续RL候选369项pins全部未变；
未修改训练源码、算法配置或排队启动器，没有新增实验分支。
全学习40513与选择9226输入缓存均已独立验收完成，分别见
`context-learning-cache-v1.json`、`context-selection-cache-v1.json`；
`context-full-input-readiness-v1.json`确认原排队Stage1消费的缓存身份一致。
下文较早的“等待缓存”与旧driver PID仅为历史观察，不代表当前状态。

2026-11-14 15:30 UTC旧selection cd263613准确wait返回0，租约桥正常释放。
连续RL验收已先获得设备：actor a0198f4b、service b685080b，远程epoch1794670284
实际inspect均running/nonOOM，固定镜像52ba94e1，actor仅见4–7，service仅见7。
首次普通后续观察不早于remote1794670884.0934343；原driver362663负责准确wait、
独立收集与新进程恢复链，尚未有连续更新完成或恢复结果。

当前本地Stage1接续driver373142在`launch_admitted_full_schedule`等待共享设备租约，
从已验收step2接续至634；选择集driver373145等待完整训练链完成。
连续RL两步及新进程恢复验收driver362663已启动上述容器，Stage1全量容器尚未创建。
旧分类分支selection的原driver229436继续独立收集与CPU指标归档，不重复执行。
到期远程观察epoch1794669676.0691872确认旧评测容器仍running/nonOOM，
HF9226完成、vLLM7635/9226，最新进度距观察不到1秒；距上次623秒推进1474条。
这是同一明确容器的verified wait，不能将等待设备视为训练失败或重启理由。
下一次普通观察不早于remote epoch1794670276.0691872。

研究发布顺序固定：完成Stage1及全选择集评测→连续RL工程验收与Stage2封存→
GRPO全曲线→同起点、同预算RLOO-clipped。连续RL短探针可先完成工程验收，
但其早期Stage1起点不能替代正式RL的Stage1终态参考。
当前没有新概率头完整selection成绩或连续RL收益，不能承诺已验证持续改善。

简单任务仍以候选概率排序与校准为目标，binary NLL是概率监督项；
独立sigmoid不自动保证概率已校准。Set-F1优化离散集合，NLL/Brier检查数值质量，
三者必须分别报告。Stage1适配同一SLURP任务协议，尚非跨任务通用指令能力证据。
若要独立归因RL项的贡献，后续须另登记保持辅助损失/预算一致的控制；
本轮只检验既定混合RL方案能否相对同Stage2起点持续提升，不新增RL/SFT胜负实验。

## 2026-11-14 较早收口核验与固定发布规则

### 发布责任与停止扩项规则（本轮补充）

用户要求的执行顺序保持为任务设计→工程/config冻结→研究训练。
这里的工程探针只验证执行与恢复，不属于研究训练，也不能报告为RL效果。
复用一套固定镜像/骨干/数据/概率头/actor/rollout实现，分别配置评测、Stage1和Stage2；
不为每个算法复制工程，不在运行曲线中修改代码、reward、任务语义或batch。

| 发布关口 | 已有依据 | 剩余工作 / 通过标准 |
| --- | --- | --- |
| 简单任务定义 | 完整60候选、exclusive_event真值、固定划分/指标/奖励 | 已锁；不混入未准入的槽位存在性或复杂Q |
| Stage1执行 | DP4更新、保存、新进程恢复及跨后端概率证据；40513/9226输入缓存已独立审核 | 既有链完成634步及9226选择集评测 |
| Stage2执行 | GRPO/RLOO单次更新；常驻engine已保存策略切换 | 同actor连续更新、下一批确用新策略、新进程恢复重放、固定reference及逐micro契约 |
| Stage2发布 | 算法/预算/评测点已登记；当前未发布 | 封存通过的候选源码和启动配置，并落实0、64…576、634全部选择集评测 |
| 学术效果 | 尚无新概率头完整选择集结果或RL曲线 | GRPO完整曲线→同起点/同预算RLOO-clipped；同时报告最终决策、概率质量和成本 |

PPO-critic、复杂Q、节点信用、扩大microbatch均不作为本次发布前置。
若必要修复改变数值行为或目标，产生新发布并重新标识起点；不能拼接旧曲线。
阶段完成以结果证据判定，不以已写好入口、CPU测试或模型参数量代替。

2026-11-14 14:06 UTC远程只读核验：物理6上的PID3656024属于本项目旧
`class-spread-batch-dp4-selection-v1`的`check_span_candidate_reload.py --phase hf`，
容器`cd263613020955be94f688ab3544d6244cd00bf5008867390a2b615bf6bdf9bf`。
新的`context-online-npu-v1`在设备预检被挡下，尚未创建训练/服务容器；
本次未终止旧评测或更换DP拓扑。此观察不是缓存进度的新观测。
本地既有接续状态仍为`exact_cache_wait`和`wait_full_training_chain`。
低KV常驻切换及IPC候选的后续证据见`context-online-runtime-candidate-v1.md`；
它们尚不构成同actor连续RL准入。

本轮再次运行 `python tools/jev/verify_context_baseline_lock.py`，364项文件身份通过，
lock ID仍为 `2ece49811d1e11eabc75f3e95e73ef532bda3fe9f16e9a399da7725ad0d95bfc`。
未修改冻结训练源码、实验配置或算法登记；此检查只证明版本未漂移，不证明Stage2已发布。

- Stage1全量音频镜像40513条已校验；当前全训链仍等待输入缓存，未开始step2→634。
  最近已归档的缓存观察（remote epoch1794664769）原缓存8448/40513、逆序辅助18176/40513，两个容器均running/nonOOM。
  这是各writer进度，不能代替全缓存唯一人口的独立终态核验。
- 常驻vLLM A→B→A已通过16真实音频×3轮×60概率与HF全零差核验，
  两次切换分别约0.795/0.795秒；证据 `context-resident-switch-npu-v2.json`。
  它验证已保存策略切换，不验证同一actor连续生成新策略的闭环。
- `tools/jev/context_rollout_session.py`把固定reference缓存、当前行为刷新和策略切换
  接成候选批次会话；与online batches/reference cache合计25项CPU测试通过。
  engine使用测试替身，actor checkpoint提交仍由调用方负责，不据此准入连续NPU训练。
- 新概率头尚无完整9226选择集结果；现有自动评测链仅排队Stage1终点634和起点0。
  Stage2发布须落实配置中所有预登记评测点，不能把首末两点称为完整RL曲线。

收口的执行顺序固定为：简单任务定义（已锁）→Stage1工程与全量接续（既有链）→
Stage1终态及选择集→Stage2连续更新/恢复验收与独立发布→GRPO全量曲线→RLOO-clipped。
复杂Q、PPO-critic、microbatch扩展不阻塞这条主线，也不混入同一发布。

运行时以发布快照为准，不以不断变化的工作树为准。参数探索仅在另行登记的候选中进行；
改变任务语义、数值路径、损失、奖励或采样的修复必须生成新版本，写明影响及必要回归，
不得沿用旧版本号拼接曲线。常规运行保留轻量身份与行为契约检查，完整恢复/跨后端验收
在发布和指定保存点评估，不把反复全量工程审核当成每次实验的前置。

## 任务先冻结

第一条主线保持已定义的简单任务：SLURP official-clean学习40513条/选择9226条，
音频与完整60候选定义输入，同一语音骨干一次输出60个独立sigmoid概率。
exclusive_event标签表示“官方本体指定的事件是否为该候选”，不声称60种语义意图存在性。
输出不作跨候选softmax；agent harness可消费候选概率，但概率校准须实测而非由sigmoid保证。
不生成转写、不生成CoT、不做开放槽值生成。测试集不参与训练/选择。

主评测query macro set-F1@.5；同时报告top1 accuracy、固定60类macro-F1、空/多选比例、
NLL与Brier。独立sigmoid+Bernoulli允许空集/多选，训练奖励set-F1显式惩罚错误选项；
它与强制单选分类不是同一动作空间，不能混用原分类实验的结果。
Stage1为pairwise ranking+binary NLL各1，Stage2为RL+衰减辅助损失。
FSC只做冻结源域模型、替换候选定义的迁移，不能用FSC结果调整源模型或模板。

复杂任务单独定义后再准入：通用Q仅依赖部署可知本体/定义/要求，不按样本答案选题；
原生单意图标签不能直接产生多意图存在性和完整槽位负例。无可靠过程标签不授予过程奖励。

## 共用工程，分开实验配置

固定Qwen3-ASR-0.6B、语言q/v LoRA r4/alpha8、共享一层Transformer概率头，
BF16执行/导出、FP32 master/Adam；verl FSDP DP4、global128、micro8×4、vLLM TP1。
镜像digest和源码hash优先于浮动tag。运行时`qwen3-asr-context-bf16-packed-row-v1`。
学习率1e-4、AdamW、grad clip1、两阶段分别634更新；当前Stage1按64步及终点保存。

统一工程只有一套；三种用途配置分别是固定策略评测、Stage1适配、Stage2 RL。
算法只替换优势估计/明确登记的loss选项，共享数据、输入、head、优化器、保存、评分。
吞吐优化如micro16、head合批、FSDP粒度另立候选，校验后生成新版本，旧曲线不原位换配置。
共享纯输入缓存允许由独立进程加速构建，但绑定及最终全人口校验不变。

## 当前实际算法与准入状态

- 当前运行的是Stage1 ranking+binary NLL，不是RL。两步真实DP保存已验收，
  完整本地输入缓存已验收并按原授权接续全量；无新算法进入这条Stage1轨。
- 当前真实音频RL探针使用GRPO与RLOO-clipped，各自从同一起点做一次更新。
  GRPO另验重放；均不是634步曲线。采用固定verl的优势估计与PPO clipped surrogate。
- GRPO是首个全量RL主配置。G16完整60-bit动作，组内中心化/标准差归一化。
- RLOO-clipped是同工程对照。baseline为其余G−1动作的平均reward，不做组内std归一化。
  与原始REINFORCE RLOO配方有区别，不能用简称掩盖clipping适配。
- PPO-critic是第三个算法设计基线，暂不实现/派发；需要value训练、恢复及独立验收。
  复用PPO surrogate不等于已有PPO-critic。其缺失不阻塞GRPO/RLOO正式基线发布。

共同损失：联合动作clipped policy loss + .01 × sum-candidate KL(current||fixed Stage1)
+ alpha(t) × ranking + alpha(t) × binary NLL。alpha从.3在前80%更新cosine衰减至.1。
clip=.2、dual clip=3、一个optimizer epoch、保留零信号组、全局真实query分母。
同query的G16动作共用一次音频前向，动作采样不等于16次模型调用。
主配置不新增RL/SFT胜负对照。混合loss的改善不能独立证明RL项有因果贡献。

## step-wise的适配边界与研究顺序

当前Q1/Span1、Q2/Span2、Final/Span是输入预设的一次联合判决；编号不是环境时间。
目前并行head的Final不会消费同层已采样Span答案，所以不能沿Q编号套用GAE或
声称生成式CoT/真实状态转移。命名以“结构化决策信用”更准确。

第一阶段保持标准估计器不变，比较：

1. Direct：直接Final任务。
2. Q-input：任务级通用过程问题+Final，只给Final reward。
3. Q-credit：与Q-input相同输入、动作、辅助监督、预算，额外使用可核验过程reward。

Q-input和Q-credit之间只改变过程reward，才能解释其增益；与Direct的差异另报输入长度、
额外监督数量和NPU成本。过程族按固定预算归一化，缺标mask且不把剩余权重重分配；
具体权重在相应数据支持审计后锁定，当前不派发复杂任务。

GRPO将整体reward做组内标准化：即使过程项权重很小，当Final全组恒定时，过程项
也可能成为全部RL信号。原始reward权重不是梯度占比保证。需记录Final恒定组比例、
过程/Final方差、有效组比例、优势尺度和最终选择集表现。RLOO不做std归一化，可作为
观察这一影响的对照，但不是天然更优。

第二阶段仅在上述对照支持后引入节点条件信用：对动作j用
A_j = R(a) - E_{b~pi_old,j}[R(a_-j,b)]。二元节点可枚举0/1，若reward可由标签与动作
直接计算，无需额外音频前向。保持期望中的old policy与baseline停止梯度。
对于可分解奖励可降低其他节点噪声；集合F1等非加性reward不能直接套旧逐类正确性credit。
节点ratio/clipping与联合动作ratio/clipping不等价，须注册新算法变体，先验证梯度和
信用定义，再做同起点/数据/预算比较。不能称为未修改标准GRPO/RLOO。

验收以完整selection Final/集合指标、校准与跨FSC迁移收益为准；过程reward上涨本身
不算成功，平行判决准确率上涨也不证明中间答案因果影响Final。

## 发布只补必要缺口

Stage1现有链继续完成，不因文档或算法设计重启。
Stage2固定缺口：实际vLLM当前策略连续切换→同actor多次更新→新进程恢复再更新，
完整逐micro行为概率契约、固定reference、可审计checkpoint与选择集入口。
完成后封存该Stage2代码/配置并跑GRPO曲线，再跑RLOO-clipped；不把PPO-critic、
复杂任务和进一步性能优化一起追加为发布前置条件。

方法来源：GRPO https://arxiv.org/abs/2402.03300 ，
RLOO https://arxiv.org/abs/2402.14740 ，PPO https://arxiv.org/abs/1707.06347 。
2026-11-14已重新核对上述原论文、verl官方GRPO说明及core_algos源码文档：
https://verl.readthedocs.io/en/latest/algo/grpo.html
https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html 。
在线文档仅用于核对方法定义，执行仍以本地固定镜像中的源码hash为准，未升级依赖。
GRPO原论文的生成token级surrogate与本项目完整决策向量的联合ratio有区别；
报告须注明“GRPO优势估计+联合动作clipped surrogate”，不能称逐项复现原论文。
同样，RLOO-clipped保留leave-one-out估计，但不是原论文未改动的REINFORCE配方。
原论文的process supervision针对自回归生成步骤，不自动为平行Q索引提供时序定义。
这些方法来源不提供本项目任务收益证据；收益仍须由冻结配置的完整评测曲线确认。
