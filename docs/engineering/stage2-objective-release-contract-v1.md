# Stage2奖励扩展发布边界

2026-11-18 启动失败复核：下方 07:41 的 live/排队描述已失效。v77 actor 在首次更新前
因 extension namespace 路径污染失败，关闭服务时又错误解析 hello；v79 因前驱失败停止。
Stage1 完整曲线 v76 已独立审核成功。当前先执行[根因分类与系统自检](stage2-startup-root-causes-20261118.md)，
新修复均为未发布候选，未重新派发 Stage2 NPU 训练；旧 detach 命令不得再次执行。

2026-11-18 07:41 UTC：用户已授权启动 Stage2。可编辑奖励版本的前置控制器
`context-stage2-parallel-v0.0.77` 正在重建来源/数值证据，正式训练尚未开始。
独立后继 `context-stage2-objective-pipeline-v0.0.79` 已启动，等待准确前驱终态；
自动接续顺序为 editable prefix/recovery → tail 2 updates → efficiency 100 updates →
formal 634 updates。任一失败均保留现场、停止，不自动重试。

启动入口：
```bash
OMP_NUM_THREADS=1 PYTHONPATH=src:.local/verification-python:tools/jev:tests \
/usr/bin/python -B -m tools.reports.run_stage2_objective_pipeline \
  --config configs/portable/operations/stage2-objective-pipeline.json \
  --local .local/config/context-stage2-objective-pipeline-v0.0.79.json \
  --action detach
```
该配置已经派发，不应重复运行；合作团队复制宏并使用新版本路径。后继 PID2466029、
startticks356429082，实时状态仍须核验。控制端结果路径为
`/srv/multimodal-jev-xingchao/operations/context-stage2-objective-pipeline-v0.0.79/`。
各阶段使用 `context-objective-{tail,efficiency,formal}-v0.0.79` 独立路径。

本轮正式配置：175物理4/5训练、6/7各一个推理服务；BF16、DP2、micro8、
global=64×DP=128、service batch32；clean SLURP TRAIN40513、2epochs、634updates。
从固定Stage1 step318初始化，reference只读使用该策略的全TRAIN缓存，
不从效率检查的step100恢复。LoRA复用flag=true、optimizer重置；仅保存adapter/head与
恢复所需状态。预登记selection点为0/64/128/192/256/320/384/448/512/576/634。
正式训练还需完整Stage1曲线发布成功；曲线由CPU独立进程并行审核，不占用NPU。

`extensions/stage2_objective.py` 是后续奖励/loss编辑入口。本次采用已冻结默认公式，
验收真实轨迹与fresh-process恢复后才接续。长训练关闭详细逐microbatch目标轨迹，
仍保留原GRPO/optimizer/reference独立重放；该放行只覆盖与已验收默认文件逐字节相同、
options={}的公式。更换公式须重新做相应独立数值验收，不沿用默认公式证明。

验证：本机27项、固定训练镜像CPU27项全部通过；固定镜像容器
`b7dad5966aeb9abb2cda6c910dbd2b5e675491f26004ea3c157166c13707642a` exit0/nonOOM。
报告 `/srv/multimodal-jev-xingchao/operations/context-stage2-pipeline-cpu-v0.0.78/report.json`。
包含包重建、防篡改、目标contract、失败阻断、并行审计绑定、曲线输入身份检查，
模型/父产物使用fixture；不能据此宣称NPU验收或正式训练成功。
每900秒由固定脚本采集训练进度，不唤起模型。


2026-11-18 07:07 UTC接续更新：用户明确下一轮直接使用可编辑奖励/loss版本。
原固定目标controller2388164及等待其整轮结束的后继2433863已在四卡空闲边界退役，
旧运行包与结果保留；不再等待旧固定目标正式实验完成。交接证据见
`/srv/multimodal-jev-xingchao/operations/context-objective-handoff-v0.0.71/terminal.json`。
可编辑prefix+recovery控制器已独立启动（PID2452638），使用原已冻结v0.0.68配置；
这是有限验收派发流程，不是正式GRPO准入。默认目标仍复现原双信用公式。
后续策略见`configs/experiments/stage2-editable-objective-default-v1.json`。
以下关于v0.0.69等待整轮结束的段落为历史状态，已由本次接续取代。

2026-11-18 06:21 UTC：确定性后继`objective_acceptance_successor`已启动，
PID2433863，start ticks355954915，现场argv/start ticks一致，等待通道为pidfd poll。
它绑定原controller2388164及其成功终态，原控制器负责的Stage1曲线、效率、正式GRPO、
GRPO曲线和迁移全部完成后，才运行v0.0.68两步+恢复验收；前驱失败即停止。
没有远程轮询、抢占、重试或模型驱动日常调度；没有修改原冻结队列。
后继6项宿主流程测试通过（调度夹具，不是NPU证据），配置绑定1496项源码/证据。
本地宏`.local/config/context-objective-successor-v0.0.69.json`；操作根
`/srv/multimodal-jev-xingchao/operations/context-objective-successor-v0.0.69`，
`waiting.json`记录准确前驱，driver目录记录启动身份。**已条件排队，尚未执行NPU验收**。
此前“尚未启动后台等待器”的记录是此次接续前状态。

真实v0.0.64生命周期准入已exit0，结果`objective_lifecycle_validated_not_executed`：
独立候选重建、完整Stage1/reference来源、bootstrap固定镜像身份与只读挂载命令验证通过。
收据：`/srv/multimodal-jev-xingchao/operations/context-objective-lifecycle-v0.0.64-validation-receipt.json`，
原始JSON输出保留在同前缀`-validation.log`。这不是NPU执行或正式训练准入。

有限两步+恢复的具体配置已准备并通过配置验证：
`.local/config/context-objective-acceptance-v0.0.68.json`，绑定1494项源码/证据pins，
prefix使用已重建v0.0.61包，recovery使用独立v0.0.68目录，沿用4卡DP2+两个TP1服务。
本次现场核验原controller2388164的argv/start ticks一致，四卡租约仍由原任务持有。
因此**尚未派发此序列，也未启动抢占/重试/轮询等待器**。当前冻结主队列仍执行原配置；
将插件候选安排入队需要独立的调度交接，不能将已准备配置称为已排队或已运行。

验收串接入口`run_objective_acceptance`已实现并通过宿主10项流程测试：prefix独立审核
完成后才能封包/启动recovery，各阶段失败停止、无自动重试；保护本地与远端独立目录，
使用共享非阻塞四卡租约。固定镜像v0.0.67流程测试10项通过、0失败/错误/跳过，
容器`a257416b174e68026d0415b2ef4db1dd228a11bfa060ea1e6b4c0b4b704b842e`
于2026-11-18 06:07:23 UTC退出0、nonOOM，已独立收集。报告：
`/srv/multimodal-jev-xingchao/operations/context-objective-sequence-cpu-v0.0.67/report.json`。
测试使用生命周期夹具，不证明真实NPU顺序验收通过。
本入口尚未派发，当前Stage1主评测队列没有原位修改或被新增后台任务接管。

恢复链候选已接入：`prepare_objective_recovery`、`audit_objective_recovery`和
`context_objective_recovery_operations`复用原封包/生命周期/恢复比较，增加插件身份及
实际reward/loss轨迹核对。14项宿主CPU检查通过，覆盖重算清单后的插件/hash/options/
reference修改拒绝、Adam/采样比较、step2轨迹覆盖、恢复只读挂载及共享函数不被修改。
固定镜像v0.0.66已独立收集，14项通过、0失败/错误/跳过，容器
`f2e39e6e24cce03d0e37ee3bb9f2db1536973c09ddbffe91dd1ed74ae645f2d8`
于2026-11-18 05:59:23 UTC退出0、nonOOM。报告位于
`/srv/multimodal-jev-xingchao/operations/context-objective-recovery-cpu-v0.0.66/report.json`。
来源/模型/远端运行使用夹具，不据此声明NPU恢复完成。
实际恢复包必须等待真实两步NPU输出独立审核，当前没有发布恢复包或派发恢复任务。

两步部署入口已接入共享生命周期：`context_objective_training_operations`与
`context_objective_storage`。扩展目录整体只读，真实候选的命令计划绑定175卡4–7，
原reference输入保持只读。宿主生命周期/结果审核组合15项测试通过。
实际完整准入检查正在执行，配置`.local/config/context-objective-lifecycle-v0.0.64.json`，
日志`/srv/multimodal-jev-xingchao/operations/context-objective-lifecycle-v0.0.64-validation.log`；
此配置关闭派发，仅用于验证。固定镜像v0.0.63因测试包漏带reference夹具文件，
10项通过、5项未通过；结论与证据保留于对应operations目录的`report.json`。
补齐测试依赖后的独立v0.0.65候选已完成固定镜像验证：15项通过、0失败/错误/跳过，
容器`ac2a4dcb5c3d6046bf0d1f07f35b07450267ab907cac584ef29014adb47da951`于
2026-11-18 05:51:50 UTC退出0、nonOOM，已独立收集。
报告：`/srv/multimodal-jev-xingchao/operations/context-objective-lifecycle-cpu-v0.0.65/report.json`。
其中来源/远端执行/optimizer使用夹具，实际公式重放使用测试surrogate；这不是NPU证据。
主队列1478项源码pins重新核对一致。完整真实准入检查与恢复链仍待完成。

2026-11-18 更新：v0.0.61真实候选已完成独立重建审核，状态为
`objective_training_candidate_reconstructed_not_admitted`，前向源码和reference绑定保持不变。
报告：`/srv/multimodal-jev-xingchao/operations/context-objective-prefix-v0.0.61-audit.json`。
v0.0.62固定训练镜像CPU检查18项通过，0失败/错误/跳过，容器退出0、nonOOM，
已独立收集到`/srv/multimodal-jev-xingchao/operations/context-objective-bootstrap-cpu-v0.0.62/report.json`。
这两项均不是NPU训练准入。

新增输出审核入口`tools.reports.audit_objective_training`及宏模板
`configs/portable/operations/objective-training-audit.json`：保留原cached训练完整来源、
真实标签、采样/rollout、reference、transport、checkpoint和master导出审核；
在局部函数作用域加入插件checkpoint身份，再核对两次更新的完整rank/microbatch轨迹。
实际自定义奖励/loss报告在`objective_trace_audit`；原`updates[].branches`仍为原始采样
信用统计，不应当作任意插件的实际优化奖励。所有旧审核模块保持原样。
宿主CPU接入6项及既有轨迹20项测试通过（26项）；接入测试的来源/优化器审核使用夹具，
尚无真实插件NPU输出可供此入口审核。部署生命周期、新进程恢复和NPU准入仍未完成。

真实两步候选`context-objective-prefix-v0.0.61-prepared`已由
`tools.reports.prepare_objective_training_candidate`完成完整父包/reference来源重建后生成，
位于`/srv/multimodal-jev-xingchao/operations/`，486个文件，manifest SHA256
`4446436823901c1f48d829d84451c6c7fac484a3fc9cfd9de5450970d6c5a664`。
`source`和`RUNTIME-SOURCES.json`保持父包内容，`objective-extension`独立列入pins；
runner只将actor入口切到bootstrap，推理服务配置仍加载原source。reference完整绑定不变。
封包/独立重建的8项替身CPU测试及bootstrap的10项CPU测试通过；真实包已做文件完整性
检查；独立重建和bootstrap固定镜像验收现已完成，见上方更新。**NPU未派发**。
准备配置：`.local/config/context-objective-prefix-v0.0.61.json`；独立审核配置另存为
`.local/config/context-objective-prefix-v0.0.61-audit.json`，未原位改写准备来源。
共享模板：`configs/portable/operations/objective-training-candidate.json`。
准备成功不开放dispatch；launch、输出审核真实执行和恢复链还需接入。

部署接入候选：`tools/reports/context_objective_bootstrap.py`将训练侧扩展放在与原`source`
分离的只读目录，不向`sys.path`追加覆盖目录。显式安装三个新增模块
`objective_trace`、`objective_plugin`、`context_plugin_trainer`，其余依赖仍从原source加载。
完整扩展文件集、每文件runtime pin、目标文件hash、原Python源码清单和父包实际导入路径
均须匹配；拒绝覆盖既有模块、重复安装、软链接和未声明文件。原`verify_runtime`未放宽。
10项宿主CPU测试通过，包括新解释器真实模块加载、有限/正式源码生成和actor挂接；
此入口尚未接入可派发训练包，也没有NPU准入。源树保留不证明任意插件无副作用，
仍须结合已固定插件源码、真实前向数值和独立训练审核判断兼容性。

2026-11-18 05:08 UTC更新：实际奖励轨迹和脚本重放候选已完成固定镜像CPU验收。
`context-objective-trace-source-v0.0.59`保留父runtime 457个文件，新增的修改文件单独声明
overlay哈希；`context-objective-trace-cpu-v0.0.60`共47项通过，0失败/错误/跳过，
容器`958edf337bc5d01b45156c861881d42755f0429b156dd55a09048ef2db226714`退出0、nonOOM，
独立收集报告：`/srv/multimodal-jev-xingchao/operations/context-objective-trace-cpu-v0.0.60/report.json`。
覆盖原生verl公式重放、实际actor loss hook写轨迹、独立rank分组/人口绑定、
奖励/loss/版本篡改拒绝和CPU Adam恢复。训练源码生成已连接begin/finish轨迹接口；
本次没有运行真实NPU训练loop。使用说明见`editable-stage2-objective-v1.md`的轨迹节。
完整运行包审核仍需把此辅助重放接入原source/rollout/truth/reference/optimizer审核链。
当前不据此声明NPU前向兼容、DP2恢复或正式训练准入。

2026-11-18 04:49 UTC更新：`prepare_objective_source_candidate`通过配置化入口生成
`/srv/multimodal-jev-xingchao/operations/context-objective-source-v0.0.57`，保持父runtime
457个文件逐字不变，新增文件来自已验镜像源码快照，封存为`SOURCE-CANDIDATE.json`。
这避免工作树中4个同名模块与父包不同、2个历史入口缺失带来的验证混淆。
从此候选源单独构建的v0.0.58固定镜像CPU检查25项通过，已独立收集，报告：
`/srv/multimodal-jev-xingchao/operations/context-objective-frozen-parent-cpu-v0.0.58/report.json`。
该验证覆盖原生verl梯度和CPU Adam恢复；未证明NPU前向兼容性，也不是可派发训练包。
模板：`configs/portable/operations/objective-source-candidate.json`；源码候选准备只新增目录，
不覆盖父包、reference、运行或排队依赖。

2026-11-18 04:38 UTC更新：恢复身份专项在固定镜像CPU候选`v0.0.56`通过，合计25项
测试、0失败/错误/跳过，已独立collect。报告：
`/srv/multimodal-jev-xingchao/operations/context-objective-plugin-recovery-cpu-v0.0.56/report.json`。
新增4项使用真实CPU Adam和共享checkpoint capture/restore：相同插件搬迁路径后下一步
参数及Adam精确相等；文件hash、options或缺失插件身份均在任何参数/optimizer修改前拒绝。
这是CPU checkpoint helper的验证，不是跨进程DP2/NPU恢复验收，也未完成部署审核链接入。

## 当前候选与已验证证据

用户要求后续通过编辑一个Python文件调整奖励和优化函数。编辑入口是
`extensions/stage2_objective.py`；候选训练入口是`tools.reports.context_plugin_trainer`。
默认公式不改变当前双信用GRPO目标；自定义指标进入`jev/objective/`数值通道。

2026-11-18固定镜像CPU候选`v0.0.55`已封存源码快照并独立收集审核：

- 本地冻结快照：`/srv/multimodal-jev-xingchao/operations/context-objective-plugin-cpu-v0.0.55/context-objective-plugin-cpu-v0-0-55-package`。
- 报告：同operations目录下`report.json`。
- 源码清单SHA256：`99dfa08598b2f9eff76adc56e13f94f742cc362a5d9e06a9552db231bb15bfeb`。
- 容器：`3b48fb11c1d0a2a18f78b38887f9dfcdf0720c0e4160e246a87837c1fe9f83eb`，退出0、nonOOM。
- 21项CPU测试通过，包括4项直接调用固定原生verl的loss/梯度验证。

这个包是CPU验证源码快照，**不是NPU训练运行包或锁定rc**。不以其中的全量工具源码
副本建立第二套长期工程基座；后续运行包只绑定实际依赖。源码进一步修改必须发布新的
验证版本，不更新本次快照或改写本次报告。

## 接入中已确认的依赖

| 现有组件 | 硬约束 | 奖励扩展必须补齐的接入 |
| --- | --- | --- |
| `prepare_context_cached_runtime` | Python源码清单参与`dual_runtime_identity` | 显式记录扩展身份，不把新源码伪装成旧runtime |
| `prepare_context_cached_training` | 消费者源码逐字等于封存父包；runner固定旧入口 | 新候选包绑定扩展文件/hash/options及候选入口，独立重建审计 |
| `context_cached_training_operations.admission` | TRAIN reference、数值面板、消费者的runtime身份相同 | 若拆分推理与目标函数身份，必须先证明未改变前向路径，不能只忽略差异 |
| `audit_context_cached_training.expected_contract` | checkpoint contract必须精确相等 | 纳入插件API/hash/options；与训练器生成的contract一致 |
| `prepare_context_cached_recovery` | 新进程恢复必须保留原优化器及目标身份 | 同插件恢复通过；插件hash/options变化必须拒绝原轨迹续跑 |
| `audit_context_cached_training`与后继审核 | 原始采样行为及奖励可独立重放 | 原行为校验继续保留；自定义奖励与loss需另有可重放记录，不能将原奖励当成实际优化奖励 |

因此，仅将trainer入口改为`context_plugin_trainer`不足以发布。当前CPU测试证明候选函数
可运行、原生verl梯度符合预期，不证明封包、reference复用、恢复和独立结果审核已经支持扩展。

## 发布策略

reference仍固定为Stage1终点318对全TRAIN的输出。奖励函数不消费selection或迁移数据。
奖励/优化目标的身份与推理数值实现具有不同含义，但现行runtime采用整体源码身份，
当前不得直接拆掉该约束。应在独立候选中建立明确的兼容性证明：

1. 锁定旧推理运行包、模型/head/LoRA、processor、候选顺序和完整输入身份。
2. 列出新旧源码差异；扩展只通过训练侧loss接口调用，推理服务与actor forward依赖保持。
3. 绑定原reference来源与新消费者身份，验证相同真实输入的概率/log概率及阈值决策一致。
4. 保留原始动作/行为重放，并记录实际自定义reward、插件配置和必要loss分解用于独立审核。
5. 两次训练更新与新进程恢复通过后，才允许新入口进入效率/正式训练队列。

若无法给出上述兼容性证明，使用新封存runtime重建reference并重验，不能伪造旧身份。
不得仅因CPU默认梯度一致就复用旧NPU准入。默认奖励通过不自动授权任意未来新公式；
新公式需固定语义、尺度、归一化、padding、数据来源与对应区分性测试。

保持总预算四卡：训练DP2，两个TP1服务；短验收4推理pass/2次更新。现有Stage1评测
和已排队依赖不原位修改。旧基线结果与插件候选结果分别记录，不能拼接成同一训练曲线。
