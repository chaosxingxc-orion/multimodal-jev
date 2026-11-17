# Step64 selection审核中断与恢复验收

> 2026-11-15 16:48 UTC实际恢复确认：迁移已完成，52,007个文件与归档共
> 49,743,816,421字节，原路径通过别名保留全部内容。`/home`可用空间10.868%，
> 原存储门槛通过；step64原audit.json已补齐passed。修正本地依赖环境后的恢复
> PID1328893存活，step128容器`4b2df9ffef675a5860eaca9114c0f787428a5cd10d6172046ae485c5dd2f1b79`
> 已由docker inspect确认running/nonOOM，原训练与step64推理没有重跑。统一后继
> PID1330344仍在准确等待完整曲线。当前还没有step128完整指标或迁移任务成绩。
> 证据：`/srv/multimodal-jev-xingchao/operations/recovery-observation-20261115T1647.json`。
> 下方未迁移/未派发叙述是各时间点历史，后继接续见[新记录](../../../recovered-research-successors-v1.md)。

2026-11-15 16:10 UTC到期核验发现GRPO原driver已因step64审核失败退出，RLOO后继及
endpoint跟随停止。step64推理容器正常exit0、nonOOM，全部产物已由原collector收集。
原日志保留在`.local/decision-objective-research/context-rl-grpo-arithmetic-v2-experiment/step-0064-audit.log`。

失败字段为top1_macro_f1。本地Python3.11算得0.7557083105767279，运行时报告为
0.755708310576728；60类支持数、预测数、TP、逐类F1与混淆矩阵精确一致。固定镜像
52ba94e1中无NPU、无网络、只读CPU探针确认Python3.12.13的sum得到报告值。原因是
CPython3.12将浮点sum改为Neumaier补偿求和，3.11为此前累加路径。

`tools.reports.context_metric_python312`在单个审核进程中为context_event_metrics模块
提供与CPython v3.12.13 `Python/bltinmodule.c/builtin_sum_impl`一致的有限浮点求和。
仅覆盖模块级sum，整数计数仍用原整数sum，不改builtins、冻结源码、预测或阈值。
这不是给macro-F1增加容差；原比较器仍要求该字段精确相等。原NLL/Brier的8ULP
规则保持原样。混合类型和非有限值拒绝，退出作用域恢复原模块状态。

实际全产物重审通过：
`/var/tmp/jev-selection-sum-acceptance-otx554if/step0064-audit.json`。
该审核重新校验训练来源、模型/adapter、256条parity、9226条音频请求记录和评分。
step64 setF1=.7745140544837106，top1=.8066334272707566，macroF1=.755708310576728，
NLL=.04020769876077619，Brier=.006088686893940114。这是单个selection点，非完整
曲线、TEST、迁移收益或复杂训练准入。不能仅凭setF1提高而忽略其他指标退化。

新增显式恢复入口`tools.reports.resume_context_selection_audit`，通过本地宏模板
`configs/portable/operations/selection-audit-recovery.json`绑定原失败状态和接受审核hash。
它持有原owner锁，拒绝已有后续point/launch/attempt；只补缺失审核及原计划后续点，
不重跑训练、step0或step64推理。失败保留现场，不自动重试。

39项相关CPU测试通过，包含原严格比较器、兼容层范围、精度门槛、曲线审核及恢复
拒绝路径。恢复控制器尚未执行真实接续，不能称已恢复队列。

实际预检：身份与原计划通过，但`/home`所在盘仅8.55%可用（182785490944 bytes），
低于10%门槛，派发被拒绝。记录为同验收目录`recovery-preflight.json`。未降低门槛，
未删除研究资产。下一步需把后续可写实验目录迁往足量本地持久盘，并显式处理原路径
依赖/后继关系后接续。当前GRPO/RLOO/FSC/TEST/MINDS链不能继续称为存活运行中。

## 2026-11-15 16:32 UTC：本地迁移及接续已启动

新增固定入口 `tools.reports.relocate_terminal_collection` 及本地宏模板。仅迁移已经
完成且独立审核通过的 `training-collected` 和对应 tar；逐文件及归档 hash 必须匹配
原 transport/proof，在本地 ext4 上复制、fsync、重新校验后才切换原路径软链接并
移除原盘重复副本。全部 checkpoint、轨迹及归档内容保留，不修改任何冻结包。
源目录 owner.lock 全程独占；中断不自动重试，不把部分复制视为完成。

持久位置为 `/srv/multimodal-jev-xingchao`，已核验根盘 ext4、可用约 14 TB。
该专用目录由已有本地 CPU 镜像的无网络受限容器创建，未下载镜像，未改变宿主机
挂载或全局清理规则。没有使用受 30 天自动清理约束的 `/var/tmp` 保存长期训练资产。
迁移只改变收集目录的存储位置，实验根、回执目录及 owner 锁路径不变。

迁移 PID `1322189`，证据目录
`/srv/multimodal-jev-xingchao/operations/relocation-grpo-v1`；此次记录时进程实际存活，
尚无 COMPLETE，不宣称迁移成功。step64 接受审核另存到同一持久盘 operations 下。

恢复 v2 PID `1323582` 已实际启动，当前在原 owner.lock 上等待上述迁移进程。
通过显式迁移 intent hash、COMPLETE、原路径目标、proof/transport 身份和原存储
门槛后，才补 step64 审核并接原 step128…634 和完整曲线。源码/依赖 1089 pins
绑定，原 driver 的 1078 pins 全部未变；每次调用前复核。失败保留现场，无重训练、
无 step0/64 重新推理。恢复日志为
`/srv/multimodal-jev-xingchao/operations/selection-audit-recovery-v2-driver.log`。

51 项相关测试通过，包括迁移内容保持/拒绝路径/写失败保护、原严格指标比较器、
缺失审核与后续点的有限编排验收。工具和模板锁定在
`configs/releases/selection-audit-recovery-v2/`，本机宏保存在忽略的 `.local/config/`。
测试不是实际迁移完成或 NPU 评测成功证据。

下一次普通检查不早于 **16:47:07 UTC**。RLOO/FSC/TEST/MINDS 和 endpoint 的原后继
已确认 stopped，尚未接回；它们还需要版本化接续及审核侧 Python3.12 求和兼容，
不可直接重启旧脚本或修改冻结配置。研究总体目标仍未完成。
