# Stage1 终点到 Stage2 的接续与 TRAIN 打分续跑

69 的 1.7B Stage1 v146 已独立审核通过，69 主、备接续队列均已停止。原同机入口和 69 租期只作为历史证据；181 接续用新部署绑定，见 [跨机交接](host69-to-host181-handoff-20261119.md)。

入口为 [接续控制器](../../tools/reports/context_stage2_endpoint_successor.py)
和 [便携配置](../../configs/portable/operations/context-stage2-endpoint-successor.json)。
本地宏绑定 Stage1 最终审核器的配置与启动收据 SHA-256、目标主机、资源锁、硬截止时间，
以及新版本独立输出目录。用 `--action validate` 检查配置，再用 `--action detach`
启动。复用同一套共享基座，运行配置按版本冻结。重复启动前检查同一控制目录下的
`started.json`、PID 和 `terminal.json`。

1.7B Stage1 的前置入口是 [Stage1 启动配置](../../configs/portable/operations/context-stage1-start-pipeline.json)
和 `python -B -m tools.reports.context_stage1_start_pipeline --config TEMPLATE --local MACROS --detach`。
它用同一冻结输入顺序做前缀、恢复和真实尾批准入。在有截止时间的配置中，顶层的
`not_after_unix`、`lease_stop_grace_seconds`、`release_reserve_seconds` 和 `walltime_multiplier`
控制正式训练租期；脚本从前缀和尾批的独立审核取最慢步时，按完整步数核算时长并保存预算和
预留时间，随后把硬截止传给正式容器的远程关停保护。`formal-lease-budget.json` 会记录放行判定，
资源不足时保留已经审核的窗口。已启动的任务继续由原收集器收口，不再用启动入口重复派发。

分开执行正式部署时，使用 [正式训练配置](../../configs/portable/operations/context-infonce-formal.json)
和 `tools.reports.context_infonce_formal --action validate` 复验封包；
[最终审核配置](../../configs/portable/operations/context-infonce-finalization.json) 绑定同一次正式派发，
由 `tools.reports.finalize_context_infonce --action detach` 等待训练和收集，保存独立审核和 Stage1 终点计划。
69 本轮的正式绑定为 [v0.0.146 本地宏](/srv/multimodal-jev-xingchao/operations/context-1p7b-host69-v0.0.146/formal-lease-local.json)。

控制器等待 Stage1 的独立审核通过，绑定实际第 318 步策略。先做同模型、同运行时的
256 条 clean TRAIN 训推数值比对，再完成 40,513 条全 TRAIN 评分并独立审核。
发布的 Stage2 视图只用 TRAIN，按实际入选数生成两轮训练日程。
前缀、恢复、真实尾批分别通过后，使用前缀与尾批验收的最慢步时乘以计划步数，
再乘配置中的安全倍数。只有预算和预留时间都能放进剩余租期，才派发正式训练。
`successor/*-budget.json` 记录各阶段预算，`successor/terminal.json` 记录收口结果。
普通监控间隔为 900 秒；能读取终态时直接收集和审核。

全量打分容器派发后，可用 [打分观察模板](../../configs/portable/operations/observe-context-reference.json)
运行 `python -B -m tools.reports.observe_context_reference --config TEMPLATE --local MACROS`。
`REFERENCE_OBSERVER.operations` 固定原评分 `finalization/config.json` 的路径和 SHA-256，
`observations_root` 为同一任务独用的新本地观察目录。脚本按原容器 ID 和镜像读取原子提交标记，
报告已提交的批数、不同输入数，以及是否已有首批；900 秒内不会重复查询。
输入去重时，不同输入数可能小于配置中的录音数，完整覆盖由独立审核判定。
恢复任务的复用量及实际推理量由恢复收据单独记录。

也可以在接续器等待 Stage1 时用 [自动监听模板](../../configs/portable/operations/watch-context-reference.json)
运行 `python -B -m tools.reports.watch_context_reference --config TEMPLATE --local MACROS --action detach`。
`REFERENCE_WATCHER` 固定 `successor_config`、`successor_driver` 两份原收据的 `path` 和 `sha256`，
以及独立本地 `output` 和 `discovery_seconds`。发现原打分句柄前只查控制端文件；派发后沿用同一
900 秒观察入口，原容器退出即停止监听。监听结果只报告提交进度，视图发布仍以独立审核为准。

2026-11-19 这轮接续版本为 v0.0.151：69 的 1.7B Stage1 为 DP4，Stage2
候选为 DP2 加两个单卡推理实例；全 TRAIN 打分用八卡。69 的硬截止为 11 月 20 日
04:04:02 UTC。181 的 0.6B 实验和 69 使用不同检查点、打分与视图。

11 月 20 日 00:39 UTC，[69 的 Stage1 现场收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-host69-v0.0.146/formal/operations/jev-1p7b-formal-146-observations/1795135169044174660-result.json)
确认四个 rank 均完成 292/318 步，已提交第 288 步检查点，容器仍运行且未 OOM。原接续器继续等待独立审核；
[v0.0.157 的全 TRAIN 打分监听](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-reference-watcher-v0.0.157-driver/started.json)
已启动并绑定原接续器，打分尚未派发。两项定向测试验证了未写完的句柄不会触发观察、不同容器的退出收据会被拒绝，且观察沿用 900 秒节流。

若原接续器因租期不足停在打分准入，且 TRAIN256 和全量评分均未进入派发流程，
新授权后可用 [终点续租模板](../../configs/portable/operations/context-stage2-endpoint-renewal.json)
和同一控制器的 `--action validate` / `--action detach`。输入宏 `STAGE2_ENDPOINT_RENEWAL`
固定原接续器的 `original.config`、`original.driver`、`original.terminal` 路径及 SHA-256，
并提供新的 `deadline_unix`、`tag_suffix` 和独立 `output`。入口复核原终点和运行时，
沿用原主机与资源锁，把新写入放到独立目录。若原打分流程已进入，先处理其确切句柄；
有已提交块时用下文的打分恢复入口。

## TRAIN 打分中断

打分的最小提交单元是一块 `batch-*`，`COMPLETE.json` 必须绑定完整输入、各分片
记录和有序结果。中断后先按原容器 ID 核对 `wait`、`inspect` 和完整收集归档。
用 [prefix 模板](../../configs/portable/operations/context-reference-resume-prefix.json)
运行 `python -B -m tools.reports.context_reference_resume_release --config TEMPLATE --local MACROS`。
`REFERENCE_RESUME_PREFIX` 绑定原打分的完整操作配置 `prior_operations`、
新输出目录 `output` 和 `io_workers`。输出包括前缀索引、仅含已提交块的归档及
绑定来源、审核和归档哈希的收据。

为新版本使用 [reference 封包入口](../../tools/reports/prepare_context_reference_job.py)，
从原 `reference_job` 复制科学配置，分别给新包、新运行目录和新操作配置使用独立位置。
在封包规格 `resume` 中设置 `prefix_root`、`receipt_sha256` 和 `replay_last`。
中途退出用 `false`，复用所有有效已提交块，从第一块未提交样本推理；
全量已提交、需要验证关停故障时用 `true`，只重放最后一个已提交块并精确对比输出。
旧包、旧日志和运行时源码保持原样。

用原独立 TRAIN256 验收和新的包身份运行
`python -B -m tools.reports.context_reference_job_operations --config TEMPLATE --local MACROS --action validate`，
再在原授权和租期允许时执行 `--action detach`。运行后审核全部人口及恢复账目。
新包仅增加评分客户端和只读前缀归档；提交损坏、模型或运行时不匹配、输入顺序不符均停止。
该路径为评分续跑候选，和已在真实 NPU 验证过的
[0.6B 末块关停恢复](stage2-preflight-host181-20261119.md)分开记录。有限训练窗口恢复
也不等于正式 Stage2 任意检查点恢复。

v0.0.152 在宿主机上重审了 69 的历史真实归档：原容器退出码为 1，159 个已提交块
经完整传输核对后封存；以相同端点和运行时生成的 40,513 条评分候选包通过独立封包审核。
断点、单末块重放、输入漂移、损坏提交和租期共 101 项定向回归通过，续跑候选尚未派发 NPU。
证据：[前缀收据](/srv/multimodal-jev-xingchao/operations/reference-prefix-resume-cpu-v0.0.152/prefix/receipt.json)、
[封包清单](/srv/multimodal-jev-xingchao/operations/reference-prefix-resume-cpu-v0.0.152/package/PREPARED.json)。

如需把恢复自动接回视图和验收，使用 [评分接续入口](../../tools/reports/context_stage2_reference_successor.py)
及 [便携配置](../../configs/portable/operations/context-stage2-reference-successor.json)。
本地宏绑定原接续配置与进程收据的 SHA-256、新输出目录、唯一标签后缀、I/O 并行宽度和关停预算。
先 `--action validate`，再 `--action detach`。控制器等待原接续进程退出；仅在全量打分容器
非零退出、归档完整并重新核验通过时接手。其余情形保留原终态。恢复前后都检查原租期，
视图发布后复用原定前缀、恢复、尾批和正式两轮训练门槛。在旧冻结运行时中，续跑客户端不要求
未启用的预取接口，关停则复用单末块重放已验证的服务入口。

2026-11-19，v0.0.155 作为 69 的 v0.0.151 备用控制器启动，等待原任务终态；等待期间不占卡。
109 项定向回归通过；用历史 159 块真实归档生成的新候选经独立封包审核，确认计划仅重放末块，
并以冻结运行时成功导入两个入口。此处证据范围为 CPU 验证，1.7B 尚未执行 NPU 恢复。
见 [归档预检](/srv/multimodal-jev-xingchao/operations/reference-resume-preflight-v0.0.154/preflight.json)
和 [备用控制器状态](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.155/successor/progress.json)。

如果原租期结束，可在同一主机的新授权窗口使用同一个模板接续。宏另外传入
`renewed_lease.deadline_unix` 和 `stopped_fallback`；后者固定备用控制器的
`config`、`driver`、`terminal` 的路径和 SHA-256。若备用控制器在派发前停止，
独立复核原打分归档。若备用容器已打分后中断，先绑定该次部署、原前缀收据、非零退出和
完整收集归档，再从备用容器的最新已提交块接回。若备用容器在恢复旧提交时即退出，只有重核
旧前缀以及两次归档的重叠部分后，才沿用旧前缀中已验证的更多提交。缺少备用终态或归档就停止。
新候选只用本次授权的截止时间，沿用原主机、资源锁、训练人口和模型身份。两条续租分支已通过
控制器及封包定向回归（37 项），尚未实际派发；现行 v151/v155 未修改。

## 视图已发布、正式训练尚未派发

在同一主机拿到新的授权窗口后，可用 [视图接续模板](../../configs/portable/operations/context-stage2-view-successor.json)
与 `python -B -m tools.reports.context_stage2_view_successor --config TEMPLATE --local MACROS --action validate`。
验证通过后用 `--action detach`。本地宏 `STAGE2_VIEW_SUCCESSOR` 绑定：

- `prior_pipeline` 和 `prior_terminal`：原流水线配置和退出记录，各提供绝对路径与 SHA-256。
  `prepared_inputs` 绑定原候选清单哈希、TRAIN 绑定配置和独立视图发布收据；后两项也提供路径与 SHA-256。
- `completed`：从 `prefix` 开始连续通过的验收阶段。每个阶段提供原操作配置与审核收据的路径和
  SHA-256，前缀与恢复另绑定原 NPU 轨迹回放收据。接续前重查原容器退出、完整归档、审核器和数值；
  只复用核验通过的阶段。
- `windows`：尚需执行的验收和 `formal`，逐项指定新的标签、封包目录与本地和远端运行目录。
  同时配置独立的 `output`、`trace_destination` 及 I/O 并行宽度。
- `lease`：本次授权的 Unix 硬截止、关停宽限、预留时间、各有限验收的最低时间和正式步时安全倍数。
  正式训练仍按冻结视图完整运行两轮，并在派发前复查剩余窗口。

`validate` 输出总步数、已复用与待执行阶段、剩余租期和各阶段最低预算。有限验收齐备后，
还会给出两轮训练含预留的预算，以及当前窗口是否能容纳该预算；验收未齐备时，正式预算留空。

可用 [接续装配模板](../../configs/portable/operations/context-stage2-view-continuation-input.json)
和同一入口的 `--action assemble` 生成上述本地宏。输入宏 `STAGE2_VIEW_CONTINUATION_INPUT`
提供 `prior_pipeline`、`prior_terminal`、`lease`、`io_workers`、`trace_destination`、`windows`、
新的 `output` 及同目录下的 `local_output`。装配器从原归档定位目标父包、绑定视图、
连续已审核的窗口及其对应的 NPU 回放；证据缺失或冲突即停止，完整重核后才写入宏并给出预算。
`windows` 可一次提供四个窗口的新位置；生成配置只保留待执行窗口。装配不派发任务，
生成的 `local_output` 用作视图接续模板的 `--local` 参数。

如果原正式容器已有派发尝试，先按原句柄收集并审核；此入口不恢复已开始的正式训练。
181 上已有的真实 NPU 归档已用于复核前缀、恢复和尾批的复用流程；复核未派发新 NPU 作业。
