# 恢复后的研究任务接续

2026-11-15 16:41 UTC。原 GRPO 已训练完毕，selection 因宿主机 Python3.11 与镜像
Python3.12 浮点求和差异停止。原 RLOO/FSC/TEST/MINDS 后继及 endpoint 均停止。
本文记录实际部署的接续，不把排队或工具测试称为完整实验结果。

## 保留原计算与实验配置

`tools.reports.context_compatible_pipeline` 仅在宿主进程中临时替换两个原 driver 的
`invoke` 路由：RLOO 的 selection/curve 审核，以及 FSC 的源策略准备。后者会重新
审核 GRPO 完整曲线，也需要同样的指标求和兼容。完整曲线比较与 endpoint 使用相同
作用域。模块退出后恢复函数对象；不改磁盘上的原 driver、不改 builtins/subprocess、
不改冻结 NPU 包、不改概率和容差、不改原配置或候选选择规则。

新的 `tools.reports.continue_recovered_research` 从共享宏模板读取原 spec 路径、原
失败状态 hash、协作锁、禁止已存在的输出、依赖 pins、原始运行阶段与验收结果字段。
只允许明确登记的入口，运行中每阶段复核 pins 与存储门槛。持有原五个后继 owner 锁，
使用 pidfd 等待恢复进程退出，并校验 PID 启动 ticks 和 cmdline hash，避免 PID 复用。
准确进程退出后还必须检查完整 GRPO 曲线完成状态；失败则停止，无自动重启。

原 RLOO/FSC/TEST/MINDS 输出尚未创建。它们在真正执行前才通过原路径软链接指向
`/srv/multimodal-jev-xingchao/experiments` 下的独立持久目录。原配置不改，原 launcher
的 launch/attempt 回执仍位于原工作目录。只允许不存在的输出；等待期间若出现竞争
输出也拒绝。新的控制器状态是权威接续状态，旧 queue-status 保留 stopped 现场。

## 启动环境纠正

首次本地预检发现缺少 `tools/jev` 导入路径，补齐后又发现缺少 `safetensors`。项目
已有 `.local/verification-python` 提供验收依赖，无需下载。恢复 PID1323582 经实际
`/proc` 核实仍阻塞在迁移锁、尚未创建恢复输出目录，因此仅终止该等待进程，保留
其 launch/log；没有中止迁移、NPU 或重复派发评测。

修正后环境显式设置为仓库内 `src:.local/verification-python:tools/jev:.` 的绝对路径，
恢复新 PID **1328893**，后继 PID **1330344**。这属于恢复入口环境修正，运行包未改。
原 recovery v2 源码锁保持；新本地 v3 宏还绑定已有 verification-python 文件。

后继共八个阶段：RLOO 全量训练与 11 点评测、完整 GRPO/RLOO 比较、FSC 冻结迁移、
FSC 学术参照比较、SLURP clean TEST、MINDS、两条固定 step634 endpoint 报告。
源策略按原 GRPO selection 选择；FSC/TEST/MINDS 不用于调参或重新选策略。

## 验收与实际状态

- 原 RLOO 包及计划预检通过，完整 steps 为 0,64,…,576,634。
- 原 FSC、SLURP TEST、MINDS spec 与 pins 预检通过；未改原文件。
- 61 项相关测试通过：兼容作用域与恢复、原调用保留、拒绝未登记入口、pidfd 身份、
  原失败记录保留、失败不重试，以及原完整曲线与比较回归。
- 1232 个依赖 pins 绑定；新源码及模板快照位于 `configs/releases/research-successors-v1`。
- PID1330344 实际存活，内核 poll 等待 PID1328893；此时恢复进程也实际存活。
  后继状态为 `waiting_exact_recovery_process`，不声称已执行 RLOO 或取得新的任务指标。

持久证据根：`/srv/multimodal-jev-xingchao/operations`。恢复日志为
`selection-audit-recovery-v3-driver.log`，后继日志为 `research-successors-v1-driver.log`，
后继权威状态为 `research-successors-v1/status.json`。本机绑定在忽略的
`.local/config/research-successors-v1.json`，共享模板不嵌入机器地址。

此次只核验新启动句柄及环境故障，不提前轮询 NPU；普通实验进度检查仍不早于
16:47:07 UTC，并按 900 秒间隔执行。总体目标仍待完整曲线、学术基线与迁移结果证明。
