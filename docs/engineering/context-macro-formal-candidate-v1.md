# 正式634步GRPO候选入口

## 2026-11-16：正式启动链路接入

新增`launch_context_macro_formal.py`，在重审完整效率选择、正式包继承关系、真实全量
日程和实际尾批验收后复用共享生命周期启动。原formal operations继续提供validate/
collect/audit，包本身不翻转launch标志；实际准入证明由外部启动入口生成并归档。
详见[正式训练启动说明](context-macro-formal-launch-v1.md)。这取代下文早期“没有
正式派发launcher”的状态描述。当前完整100步/尾批仍未完成，尚未封正式包或NPU
派发；稀疏恢复仍未NPU验收、不自动恢复，不能据代码接线宣称训练/收益已完成。

## 2026-11-16：恢复部署与失败收集入口

`context_macro_recovery_operations.py`接入共享DP2训练/双TP1推理生命周期，支持validate、
run、collect、audit。部署前重新核验已终态父尝试的完整transport及学习前缀；仅允许
已声明的路径、恢复游标和初始行为策略变化，训练/服务配置及runtime源码保持一致。
恢复封包现在继承父包镜像/runtime信息，保留`training_dispatch_allowed=false`；实际
启动由本地`MACRO_RECOVERY_DISPATCH_ALLOWED`明确配置，当前没有实际恢复包或派发。

预检查验证固定reference adapter与Stage1匹配，同时由validate_resume独立验证
恢复行为adapter、checkpoint和双rank分片。actor仍torchrun DP2挂卡4/5，双服务
仍卡6/7，复用设备租约、隔离本地目录、固定镜像、准确容器句柄和单次尝试规则。
收集后再次执行完整容器/transport/输出审核，拒绝复用父容器；只有全部成功才在
`recovery-audit.json`置`resumed_runtime_transport_verified=true`。单独输出审核
入口仍为false；该字段并不代表连续执行等价、实际NPU恢复或任务指标改善。

新增`collect_context_macro_interrupted.py`解决失败父尝试无法进入成功收集器的问题：
准确四容器必须已退出、preflight必须成功，后续client或service失败/OOM允许保留。
不调用stop/start，不改写driver terminal或退出码；完整收集和hash验证后，只审核
指定稀疏点之前的有效学习记录，输出`prefix-audit.json`及重审后的
`recovery-prefix-evidence.json`，显式`parent_training_complete=false`。如果收集或
前缀审核失败，保留原始证据，不自动重试、不覆盖。当前仅支持fresh formal父尝试的
一次恢复；再次中断后的多段恢复尚未实现，不伪装成原fresh父尝试。

调用顺序（路径由三个portable模板和本地宏提供）：

```bash
python -m tools.reports.collect_context_macro_interrupted \
  --config configs/portable/operations/collect-context-macro-interrupted.json --local LOCAL_MACROS
python -m tools.reports.prepare_context_macro_sparse_resume \
  --config configs/portable/operations/prepare-context-macro-sparse-resume.json --local LOCAL_MACROS
python -m tools.reports.context_macro_recovery_operations \
  --config configs/portable/operations/context-macro-recovery-operations.json --local LOCAL_MACROS --action validate
```

同一operations入口的run、collect、audit接续验证后的包。恢复包仅包含LoRA/head和
必要optimizer/master/元数据，基座模型只读引用，不复制或重复保存主模型。

首轮封包/部署/输出49项通过11.99秒，容器/传输回归36项通过3.03秒，失败收集及
部署/传输38项通过3.08秒。真实固定命令构造、文件归档/解包/hash和审核状态机执行，
最后补preflight失败拒绝与失败报告字段后，组合66项通过12.60秒。
SSH、模型和父训练记录采用显式替身；没有formal checkpoint或真实NPU恢复成绩。
运行中的100步study、尾批和迁移精度队列依赖均未修改。

## 2026-11-16：恢复前缀与恢复段独立审核补齐

新增`audit_context_macro_formal_prefix.py`，从hash锁定的原共享审核器派生离线前缀
审核，未改任何运行中依赖。允许父训练异常终止，但只认可64..576指定恢复边界之前
的完整学习记录：原634全量日程、每步样本/标签/零权重、独立奖励与优势重算、宏日志、
BF16导出、双服务激活、rank parity与稀疏master/Adam都沿用共享检查。未完成的后缀
显式不计入学习历史；不要求异常父进程具有成功的最终result或服务terminal。

`prepare_context_macro_sparse_resume.py`现在首先绑定完整收集目录到transport，拒绝
额外/缺失/变更文件和软链接，再检查准确四容器终态、checkpoint和前缀。只有独立
前缀审核成功才输出`prefix_learning_history_verified=true`，并将完整证据放入
RECOVERY.json。此项取代下方早期prepare-only版本的false描述；派发仍为false。

新增`audit_context_macro_formal_recovery.py`：从RECOVERY所指父operations重新执行
checkpoint/transport/前缀审核，核对恢复资产与父提交逐字节一致、完整配置仅允许已
声明的路径/恢复游标变化、全部runtime Python未变。恢复段从实际初始策略与两rank
master/Adam恢复收据开始，独立遍历其余更新；reference缓存按新尝试冷启动重新核验。
前后update索引必须恰好组成0..633，否则拒绝合并，不能接受复制的审核JSON作凭据。

固定报告入口：

```bash
python -m tools.reports.audit_context_macro_formal_recovery \
  --config configs/portable/operations/audit-context-macro-formal-recovery.json \
  --local /absolute/path/to/local-macros.json
```

本地宏为FORMAL_RECOVERY_PACKAGE、FORMAL_RECOVERY_COLLECTION、FORMAL_RECOVERY_OUTPUT_AUDIT；
报告必须写入独立新位置，不写入已封存输入或覆盖旧证据。输出明确
`resumed_runtime_transport_verified=false`、`training_dispatch_allowed=false`：该入口
证明输出历史组合，恢复段实际容器/传输审核仍需部署层接入；不声明与未中断执行精确
等价、NPU恢复成功或任务收益。没有真实formal checkpoint，尚未执行实际恢复。

验证：前缀/封包/恢复入口61项通过17.17秒；前缀/后缀/封包组合50项通过25.20秒。
报告入口加入后的恢复审核18项通过11.16秒，实际宏解析和拒绝覆盖测试执行。
真实状态机逐步访问634日程及稀疏检查点、实际文件封包/搬迁/hash/重封攻击测试执行；
模型/奖励算术/容器观测沿用显式替身。测试覆盖前缀腐败、未消费后缀、缺少宏日志、
尾批分母、旧策略激活、错误Adam恢复声明、修改目标/源码/父证据以及合并缺口/重叠。

## 后续恢复封包与效率选择修订

2026-11-16 13:19 UTC：新增固定后继`run_context_tail_successor.py`，portable
`context-tail-successor.json`、local `tail-successor-v1.json`，全部机器路径通过宏读取。
在实际完整100步比较成功后运行已封存的尾批诊断；等待原study准确PID/start/cmdline
的pidfd，不远程轮询、不模型分析、不重试、不派发正式训练。再次核验81个pins、
父包身份、同设备lease、100步12800曝光及比较审核后，复用原lease/空闲/本地存储/
准确容器生命周期，最后独立collect/audit尾批所有输出。

驱动已启动PID1895486，start341178351，cmdhash
`bdd2aa924c84960fdee04048b2f6bfe0235cd92d4730ed32feba2fe7b3258fa5`；
`operations/tail-successor-v1/waiting.json`指向原study1891308，两进程实际身份均live。
10项顺序/失败/不重试/重复detach拒绝测试1.93s通过；早先与尾批组合29项4.57s通过。
单独本地operations仅启用`tail_diagnostic_dispatch_allowed`，正式训练flag仍false，
封包默认operations未改。尾批尚未NPU执行，现有batch32训练/采集仍由原驱动负责。

2026-11-16 13:13 UTC：尾批专用prepare/operations/独立输出审核已补齐。
`context_macro_tail_operations.py`通过锁定共享源码hash派生验证与审核函数，复用
相同四角色生命周期、准确容器/transport审核、DP2训练与双TP1同步，不改正在
执行的73个pins。显式尾批contract校验后，审核仍逐项检查原始日程标签/零权重、
65/128分母、每步LoRA/head导出、master/Adam、双副本activation及rank parity。
诊断状态为`finite_dp2_tail_outputs_audited`，不等同formal/恢复验收。

新增portable `prepare-context-macro-tail.json`和本地宏`tail-admission-v1.json`。
真实封包`operations/tail-admission-v1-prepared`，manifest
`a6974648e0ad4595506b16cd99a713a6e903496957ee8551602bb43b13b30f2c`；
1098个父包Python文件逐字节一致，隔离进程生成26898字符trainer成功，记录
`candidate-validation.json`。operations默认禁止派发，尚无NPU尾批运行；服务容量
沿用32，但不抢占正在执行的100步对照。封包/operations/原有限生命周期及审核
34tests10.08s通过，另尾批loader/operations25tests36.59s通过；测试模型、信用和
checkpoint算术使用替身，不作实际梯度或恢复结论。

新增`context_macro_tail_candidate.py`提供有限尾批诊断执行入口，完整源日程hash及
两轮40513人口核验后，显式选原日程indices316/317：第一轮尾批65真实+63零权重，
第二轮首批128真实。诊断从共同Stage1起点做两次更新，保留原epoch/batch/slot输入
身份，但optimizer/rollout游标为诊断0/1，contract与result均显式记录tail_admission。
它不是正式训练的第317/318步或正式前缀，不能合入正式历史，也不声明恢复已通过。
旧短程/效率包及日程不变；原生verl、BF16、LoRA/head导出、双副本确认路径保留。

12项生成/边界/真实CachedContextBatches消费测试31.03秒通过。另从实际portable v2
发布读取40513人口、完整634日程，验证所选65/128分母；复制已冻结service32父包到
本地隔离目录、补两项入口依赖，在独立Python进程生成26898字符trainer成功。
证据`/srv/multimodal-jev-xingchao/operations/tail-candidate-validation-v1/report.json`。
这些是源代码/真实日程验证，模型输入在单测中为替身，尚无NPU尾批更新；专用
prepare/launcher/独立尾批审核仍待。该本地验证副本未封包，不可直接部署。

2026-11-16：`prepare_context_macro_sparse_resume.py`及portable模板
`prepare-context-macro-sparse-resume.json`补齐稀疏恢复资产封装。只接受九个非终点
恢复游标64..576；读已终态的精确四容器命令/运行身份、绑定transport及原始归档，
从commit核对双rank分片、执行权重、完整policy/native adapter。执行已有checkpoint
Adam校验及FP32分片→BF16导出的逐张量重建后才复制。即使父尝试失败，也只能消费
已完整提交且通过校验的检查点，不能从退出状态推断此前学习历史正确。

新目录复制LoRA/head、marker/processor及恢复所需Adam/master/元数据，不复制主模型；
保留冻结runtime Python和reference策略、全量日程、global/microbatch、算法不变。
服务从恢复策略启动，reference cache使用新隔离路径；缓存重建会影响耗时，不作连续
运行等速承诺。最终重定位到本地包执行恢复校验，拒绝缺失分片、坏hash、改变目标、
不完整policy/adapter、活跃父容器或重叠目录。

41项恢复封包/原恢复入口测试3.62秒通过；实际复制、封包、manifest、恢复contract
校验执行，模型/容器/张量审核在封包测试中为替身。原恢复入口另含真实CPU Adam
连续/恢复比较。尚无正式训练检查点可供真实封包，未派发NPU恢复。输出明确
`training_dispatch_allowed=false`、`prefix_learning_history_verified=false`，仍需
正式恢复部署入口及前缀/恢复段合并审核，不能将资产封包当作完整训练验收。

micro16实际NPU OOM后，正式准备另支持固定micro8的batch4/32服务容量对照，
完整100步重审后选最低耗时（平局选4），不要求失败micro16/未测micro32完成。
旧三microbatch模式未放宽；新模式见`prepare-context-service-selected-formal.json`。

状态：候选实现与CPU验证，未封包、未准入正式派发。运行中的micro8/16/32效率包不改动。

继承已有研究预算：SLURP clean TRAIN 40,513、两轮634更新、seed317、global128，
DP2 actor与两个TP1推理副本。microbatch由匹配的100步效率对比决定。算法保持
`dual_credit_grpo_floor_clipped_v1`及`dual-credit-cpu-pair-fp64-v2`，BF16执行、
FP32 master/Adam；固定辅助排序0.1、NLL0.3、reference KL0.01。当前正式候选继续
完整均匀覆盖的源域日程，难例刷新留作后续单独对照，避免改变效率组之外的算法变量。

`tools/reports/context_macro_formal_candidate.py`从现有DP2 trainer生成新入口，原短程
与效率入口仍只允许原预算。正式contract绑定起点0/终点634、固定评测点及seed，
原生verl更新、每步行为精度检查、LoRA/head导出及双副本确认沿用已验收路径。
本候选已有torchrun兼容CLI，但没有正式派发launcher；直接放宽旧
`engineering_prefix`不能启动正式模式。

每步都生成`policy-NNNN-committed.json`并刷新两推理副本。只在64、128、192、256、
320、384、448、512、576、634保存两rank FP32 master/Adam与恢复commit；这些点
加上共同Stage1组成11点评测曲线。每步BF16执行和LoRA/head产物仍保留以供独立
核对，本候选不自动删除；归档后清理由统一规则执行。全主模型始终只读，不反复保存。
此改动减少恢复状态落盘，不代表已测得提速，也不保证总存储已经最小化。

真实日程核对：固定schedule manifest
`d9cd5832aa160df46a4c10a2051ff12c352d2a2609e2cffd6901267b28a814f6`，
全部81,026次真实曝光通过；第317、634步各65真实+63零权重padding，其他632步
均128真实。新增`validate_schedule`逐轮核对每个源样本恰好一次，不把padding计入
loss分母。现有双信用组审核核对每个slot的weight/real_request/global_real_requests，
两rank记录也须与对应真实分母一致。

`audit_context_macro_training.audit_outputs(..., formal=True)`是独立输出审核候选：
核对634份policy commit、10份恢复checkpoint、每步完整组与journal、导出BF16权重、
双副本激活和尾批分母；仅恢复点可从持久化master重建BF16并核对Adam。非恢复步
不声称具有持久化master审核证据。实际容器/运行包收集、正式初始策略来源和完整
selection曲线另需上游审核，当前旧`audit_collected`仍只准入短程/效率包。

验证：34项候选生成/日程/旧短程测试通过87.69秒；18项journal/master审核回归通过
7.30秒。新导出测试执行实际生成代码段，使用CPU张量和FSDP/collective替身，覆盖
普通步、64步边界、317尾批及634终点的落盘选择，并拒绝保存冻结主模型张量。
真实v6已完成两步产物用修改后的默认审核路径重审成功（`finite_dp2_training_audited`）。
实际全日程从portable v2及冻结schedule重新加载核验通过，无训练/目标评测派发。

补充：`context_macro_formal_operations.py`复用共享容器命令与收集生命周期，提供
validate/collect/audit，不提供run；portable模板只从本地宏解析机器信息。显式formal
验证要求尚未准入派发及已选microbatch身份，默认短程/效率校验保留。
完整634步审核状态机的9项合成测试通过（15.94秒），formal operations与旧入口等
37项回归通过（16.38秒）。这些测试使用运行产物/模型替身，不是634步NPU实测。
实际既有micro8-attempt2、v6及replay三个冻结包经默认validator重新验证通过，
manifest保持原值；未重新派发或收集正在运行的训练。

`prepare_context_macro_formal.py`现提供正式封包入口，配置模板为
`configs/portable/operations/prepare-context-macro-formal.json`。每次先重新执行效率
对比的独立审核，要求micro8/16/32三组各完成100步；预声明选择规则为100步完整
update总耗时最小，精确同耗时选较小microbatch，另保留去前两步统计。不以目标
成绩选工程配置，也不声称一次效率窗口能证明统计显著提速。
从所选包的共同Stage1起点重新开始，完整保留冻结Python源码，添加formal及恢复入口，
在隔离子进程从封包路径导入生成器检查兼容性。真实发布/日程完整核验后，更新
内部pins和运行身份；FORMAL.json绑定完整对比证据与来源包。仅在本地持久盘新目录
生成prepare-only包，不派发、不修改任何既有包、不复制完整主模型。
实际micro8冻结父生成器与新增formal入口已在独立Python进程生成26,922字符并通过
语法检查，未修改冻结包；这不是正式模型运行验收。三组实际效率对比仍未完成，
因此尚未生成真实正式包或选出实际microbatch。
封包/选择/旧对比与operations回归最终41项通过（2.68秒），模型、实际效率审核和
存储环境在封包测试中为替身。实际冻结包的v2 release与全量schedule另行复核通过：
81,026真实曝光、634更新、尾批65；封包器将远端package路径映射至本地已封存
package读取，外部共享release路径保持不变。首次真实核验暴露路径未映射问题，
已修复并增加回归，未修改远端数据或运行包。

剩余：正式模式启动准入及实际封包、NPU恢复检查点与尾批验证、已选
效率配置身份绑定、源评测点与三角色迁移包串接。尚不能从代码生成或既有两步重审
推断634步训练成功、稀疏保存提速、源域收益或迁移收益。

源评测的离线11点策略计划生产器及收集重审接入已补，实际推理与源曲线仍待，
详见`context-macro-source-evaluation-plan-v1.md`。

新增`context_macro_formal_resume.py`执行入口，只允许64..576的九个非终点恢复游标，
保持原634步实验contract、seed、目标、schedule及microbatch；不接受317尾批处
不存在的恢复状态。启动时核对commit、两rank分片、策略manifest与native adapter，
然后复用生成trainer原有集体验证、FP32 master/Adam恢复、BF16执行权重比较和
下一步双服务激活。它没有自动派发权限，也不能替代已执行前缀的独立完整审核。
正式封包现在同时冻结两个入口，从而恢复时不需给原运行身份添加新代码。

实际生成的恢复代码块已用CPU Adam执行：从64次更新状态恢复、继续第65次更新，
参数和Adam状态与连续执行逐张量相等。55项恢复/原候选/operations测试通过；
随后恢复与封包47项回归通过3.75秒。两rankcollective在代码块测试中为替身，
尚非NPU验收。实际冻结micro8父源码加两个入口在隔离Python中生成检查通过，
原70个串行驱动pins不变。首次检查发现恢复入口依赖未封装的离线audit模块，
已解除该运行时依赖，恢复contract另与独立审核contract比较测试。

仍待：恢复包封装与原训练前缀/恢复段的合并审核、NPU尾批及稀疏恢复验收、
正式派发。仅有恢复执行入口不表示完整恢复生命周期已完成。
