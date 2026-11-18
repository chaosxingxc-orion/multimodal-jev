# 日常操作归并：固定入口与配置

遵循[工程基线规约](context-prob-v1-baseline.md#日常任务执行规约)。本页登记实际已实现的
能力，不把登记或设计视为整套清理、采样及编排已经统一。

2026-11-17 简单任务两阶段有限验收已接入固定后继：
`context_train_parity_operations` → `context_reference_job_operations` →
`context_cached_acceptance_sequence`。末项只按配置顺序调用现有prefix、真实尾批、
新进程恢复的prepare/launch/audit；准确pidfd等待、同一资源锁、失败停止下游，
不增加模型调用、轮询或正式训练派发。共享模板与证据边界见
[四卡基线](context-four-device-simple-baseline-v1.md#缓存grpo有限验收固定串接2026-11-17)。
本地CPU回归通过不等于NPU有限验收完成；执行状态须查准确进程及独立终态。

缓存GRPO成本汇总使用`tools.reports.context_cached_efficiency_cost`及
`configs/portable/operations/context-cached-efficiency-cost.json`，复用已有逐步效率汇总，
不启动训练、远程轮询或模型分析。报告把reference一次性建库与训练loop/容器成本分开，
36项本地CPU回归通过，真实100步报告仍待运行。配置字段和指标边界见
[缓存GRPO效率协议](../../experiments/context-cached-grpo-efficiency-v1.md)。

团队快捷入口见[配置说明](../../configs/README.md)。最近一次联合验收覆盖宏解析、
训练模板、两类报告、缓存清理、全量日程与loader、编排、轨迹导出及运行环境计划，共96项通过。
11份共享执行模板均可用团队宏示例解析，且不含当前用户的绝对根路径。
这是本地工程测试与模板检查，不包括新NPU训练、跨机器执行或完整学术收益验收。

新任务统一通过[本地宏入口](local-config-macros-v1.md)生成配置；可共享模板位于
`configs/portable/operations/`，地址及本地产物绑定集中在`.local/config/macros.json`。
下方step512配置/命令为已完成工程验收的历史记录，不作为团队迁移时需要改写的模板。

## 已归并：已提交checkpoint的TRAIN信号统计

入口：`tools/reports/committed_signal_report.py`。配置样例：
`configs/operations/committed-signal-step512-v1.json`。后续相同协议的checkpoint只更换
配置中的step、审核证明/hash、归档位置与输出目录，不再复制Python脚本。

在仓库根目录执行（本地已有核验依赖，无需安装或下载）：

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:. python -m tools.reports.committed_signal_report --config configs/operations/committed-signal-step512-v1.json
```

流程固定为：验证审核证明与四rank更新记录 → 构造本地observation → 复用现有reward
support及实际动作重放分析器 → 核验来源 → 发布结果/hash清单。不调用模型、SSH或训练。
这一路径仅适用于现有SLURP冻结数据/日程、DP4、全真实样本global128、Final60/G16协议；
末批含零权重padding、端点概率等不支持输入会拒绝，不能声称任意数据集通用。

输出包含`observation.json`、`reward-support.json`、`reward-groups.json`、`summary.json`、
`COMPLETE.json`及`status.json`。指标口径沿用原分析器，汇总保留训练耗时、奖励组及概率
质量；不把同批TRAIN改善解释为selection收益、校准结论或复杂任务准入。

入口复用本地存储检查，文件和目录提交保留fsync。持有单实例锁；相同配置已成功时
核验输入与输出身份后返回`already_complete`，不重新计算。失败记录并保留现场，重入拒绝
自动重试；需排查后使用新版本配置及目录。不得接管现有后台owner或覆盖旧产物。

## 本次验收

- `tests/test_committed_signal_report.py`：8项通过，覆盖单实例、幂等、输入/输出变更、
  计算中来源变更、失败不发布/不重试、存储拒绝及证明绑定。
- 真实归档step512在本地完成运行；两份报告的全部非来源字段与既有归档逐项相等。
- 第二次执行返回`already_complete`，未重复计算。
- 实际发布清单：`.local/operations/committed-signal-step512-v1/COMPLETE.json`。

## 已归并：概率错误分层

入口：`tools/reports/probability_error_report.py`；配置：
`configs/operations/probability-errors-step512-v1.json`。只消费上一环节的`COMPLETE.json`，
配置绑定其SHA256，验证全部来源及输出后计算，不读取未发布的部分结果。
复用上一入口的存储检查、持久提交、校验及单实例机制。

```bash
PYTHONPATH=src:. python -m tools.reports.probability_error_report --config configs/operations/probability-errors-step512-v1.json
```

固定输出四类结果（精确集合、空集、包含gold的多选、错误非空）的NLL贡献、概率饱和
计数和reference→behavior转移。保持原Final60、TRAIN128、阈值0.5口径，端点概率明确
拒绝，不静默裁剪。输出为`probability-errors.json`及独立`COMPLETE.json`；失败不自动
重试，成功重入只校验并返回`already_complete`。

11项测试通过，覆盖四类计算、分母/贡献、非法概率、数据绑定及幂等与来源变更拒绝。
真实step512全部非来源字段与原分层报告完全一致，第二次执行返回`already_complete`。
发布清单为`.local/operations/probability-errors-step512-v1/COMPLETE.json`。这是旧结果
的工程回归验证，不是新增训练效果结论；未运行新的checkpoint分析或模型调用。

## 已归并：可再生成的本地缓存清理

入口`python -m tools.cache_cleanup --config configs/portable/operations/cache-cleanup.json`
直接使用统一宏加载器，`--local`或`JEV_LOCAL_CONFIG`选择本地绑定。默认只生成候选清单；
配置中另设新的输出版本后加`--apply`执行删除，不需要每轮模型挑选文件。

固定策略仅选择有源码的`__pycache__/*.pyc`及`.pytest_cache`内普通文件；配置控制目录、
最小年龄、文件数/总字节预算、保护路径、源码pins清单及共享清理锁。本机策略限定
src/tools/tests、48小时、最多10000文件/1GiB，并读取原队列六组及终点分析的pins。
被pin的源码对应字节码也保留。团队新增运行/排队任务时必须更新该保护登记；工具不会
自动发现所有任务或保证其他进程配合共享锁。不要在有不协作写入者的目录使用本工具。

跳过符号链接、硬链接、共享模型/数据目录、虚拟环境和.local等运行状态目录；只允许
本地文件系统，不在故障主机上修改Ceph。先持久化plan，执行前重新检查配置、依赖清单、
文件身份/hash及源码，逐级无符号链接打开父目录，记录删除intent和结果，保留本地fsync。
不递归删目录；部分失败保留现场，不自动重试。不把日志中的候选字节数称为已回收空间。

9项隔离测试通过，包含实际删除、幂等、失败留档、不删除checkpoint/源码、依赖保护、
文件替换、父目录软链接替换、文件年龄和清理预算。真实工作区检查以
`.local/operations/cache-cleanup-plan-v1/COMPLETE.json`为完成依据；该版本仅生成清单，
实际已正常完成：候选0、删除0。没有符合48小时门槛且不受保护的缓存，不放宽规则凑清理量。

## 已归并：全量训练数据日程

全量训练数据组织新增固定入口`tools.prepare_training_schedule`，共享配置为
`configs/portable/operations/training-schedule.json`，使用本地宏中的release、ledger及输出
地址。复用原`prepare_context_training_schedule`和`build_epoch`，不重写采样算法。
v1显式锁定seed317、两轮、global128、DP4、micro8、全量无放回、类别分散、batch内长度
分组和零权重padding；其他参数拒绝而非静默忽略。此处类别分散仅改善类别出现时序，
不把原本不均衡的类别样本数变成均匀分布，不以重复采样改变训练人口。

```bash
PYTHONPATH=src:tools/jev:. python -m tools.prepare_training_schedule \
  --config configs/portable/operations/training-schedule.json
```

入口校验数据与processor账本、生成日程，再用现有loader核对全量人口和rank分配，
发布源码/输入/输出身份。成功重复执行只校验；失败留档不自动重试。不启动模型/训练，
产物`training_dispatch_allowed=false`。20项入口与既有schedule/loader测试通过。
当前只覆盖此固定SLURP训练协议；任意数据集抽样、其他batch或seed需要独立版本验收。

真实重建完成：每轮40513条、两轮634更新、81026真实曝光、126零权重补齐。
`epoch-0.json`、`epoch-1.json`、`MANIFEST.json`与原冻结日程逐字节hash相等，重复执行
返回`already_complete`。发布及一致性证据位于
`.local/operations/full-training-schedule-v1/{COMPLETE,equivalence}.json`。

## 已归并：已有终态收集的轨迹导出

`tools.export_training_trajectory`通过共享模板
`configs/portable/operations/training-trajectory.json`读取本地宏绑定的collection及transport。
复用原终态收集器产物，不重新SSH收集，也不接管原owner。校验transport身份、配置及
四rank的每步记录，要求相同全局rollout，按(actor_update, request_id)导出JSONL，
另存各rank耗时、metrics和实际real_queries；记录出现次数包含padding，不冒充真实曝光。

```bash
PYTHONPATH=src:. python -m tools.export_training_trajectory \
  --config configs/portable/operations/training-trajectory.json
```

原始query完整保留，包括概率、动作seed/hash和输入身份；没有完整动作位数组的记录
不伪造成已导出数组，不重采样或重算reward。更新缺口由contiguous显式记录，不默认
认为收集了完整训练。终态失败记录也可以导出，输出不认证训练成功或收益。

7项测试通过；真实两步旧工程验收归档导出256次query，全部逐条等于rank0原记录且
四rank副本一致，重入already_complete。证据
`.local/operations/trajectory-prefix-v1/COMPLETE.json`，该prefix不是全量训练结果。
该工具已加入编排kind `training_trajectory`，共享流程模板为
`configs/portable/operations/trajectory-pipeline.json`；远程采集仍由现有固定collector负责。

编排及轨迹相关23项测试通过。编排现在逐项验证子发布清单的预期输出hash和完整输出
集合，并将子清单声明的sources纳入整个流程的复用校验，避免只对当前文件计算hash就
采信成功。此处“完整”指清单声明的集合，不声称覆盖Python的全部传递依赖。

实际轨迹编排完成且重复执行already_complete，JSONL与此前256次query导出逐字节相等。
本次/home可用比例9.96%，低于规约10%；未降低门槛，改用经检查的/var/tmp所在ext4盘
（可用约95%）验收。绑定副本及发布证据位于
`/var/tmp/jev-trajectory-acceptance-_gj5iijf/{macros.json,workflow/COMPLETE.json}`。
这只是本次本地验收路径，不是长期归档或训练目录；原本地配置写入在存储检查时被拒绝，
因此没有为默认macros.json添加TRAJECTORY_PIPELINE_OUTPUT。使用时需通过--local指定
已配置且满足存储要求的绑定文件。原v2编排源码已留档在旧报告流程目录。

## 本地流程编排及剩余边界

本地流程编排已提供`tools.run_operations`，共享示例为
`configs/portable/operations/prepare-training-configs.json`：

```bash
PYTHONPATH=src:. python -m tools.run_operations \
  --config configs/portable/operations/prepare-training-configs.json
```

配置声明步骤id、已登记kind、子配置及depends_on；执行前拒绝重复id、缺依赖、环和
缺失宏，按依赖顺序调用已有工具。统一保存本地宏快照、配置/入口hash、步骤结果及完成
清单，单实例锁保护同一流程。失败停止下游并留档，不自动重试；成功重入验证已记录
产物后返回，不重复调用工具。没有任意shell/模型/远程训练执行入口，也不自动调度。

登记kind为materialize、training_schedule、cache_plan、signal_report、probability_errors、training_trajectory。
缓存只生成清单，删除仍用独立清理入口。子工具自己的身份、来源、输出及锁校验保留。
v1只支持已存在的子模板与宏绑定。v2增加概率错误步骤的`signal_from`，只能引用
depends_on中声明的signal_report步骤；验证其完成清单、来源及输出后，自动生成该步骤
专用的SIGNAL_PUBLICATION和SIGNAL_PUBLICATION_SHA256宏快照，不修改全局本地宏文件。
完整实验队列还未迁入。各步骤输出应使用独立目录，不互相覆盖。

8项编排测试通过，包含真实配置生成、依赖失败阻断、成功不重复执行、输入/产物变更
拒绝。真实actor→service配置生成完成，两份解析结果与原包相同（actor派发false）；
重复运行返回already_complete。证据位于
`.local/operations/prepare-training-configs-v1/COMPLETE.json`。目前实际编排验收仅覆盖
materialize分支，其他登记分支复用各自工具，但尚未完成组合流程验收。

报告串接模板为`configs/portable/operations/report-pipeline.json`，使用同一
`tools.run_operations --config ...`入口。本机独立输出目录为
`.local/operations/report-pipeline-step512-v1`；只消费归档step512，不做新采集。
编排测试现为11项通过，新增覆盖动态绑定、旧宏覆盖、上游产物变动拒绝及依赖声明。
此前8项materialize验收仍为原v1代码的历史证据，已保存执行源码快照到原验收目录；
当前代码更新后不把旧运行的源码hash视为仍匹配。

v2真实signal→probability_errors已完成，reward-support、reward-groups和probability-errors
三份报告的全部非来源字段均与原归档一致；下游绑定的SHA与本次上游完成清单一致。
重复执行返回already_complete，发布及比对证据为该目录的COMPLETE.json和equivalence.json。
这证明本地报告动态衔接可运行，不是新训练收益或完整实验编排已完成。

现有900秒监听及完整曲线终点分析继续由原固定脚本承担。本入口尚未接定时调度，
没有新建后台进程。checkpoint/研究中间产物的归档保留策略、轨迹收集、采样及完整任务编排仍需
逐项归并；复用现有实现，优先用配置接入，避免再建第二套工程。当前运行和排队任务的
冻结依赖保持不动，旧逐step脚本及证据暂保留，不在这次代码归并中删除。
