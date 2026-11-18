# Q7资源等待与原预算接续

2026-11-12 UTC。Q7训练容器实际退出0，原控制器持续等待独立suite首次创建。
11:58资源快照发现privileged容器运行固定tar日志归档脚本，未声明NPU设备。
脚本仅读取指定runtime/queue及前驱完成凭证并向stdout输出归档；未执行模型、
外部子进程或设备打开。审计快照见[intent-resource-archive-inspect-v1.json](../../../intent-resource-archive-inspect-v1.json)。

资源识别新增该确切镜像摘要、Python脚本SHA256、两个固定路径及原host PID、
挂载、自动清理配置。变更脚本、参数、路径、设备声明等仍拒绝。实际主机
设备FD扫描继续执行，不能以无设备声明或容器名称放行。它只解决已审计
脚本的保守预留误判，不涵盖其他未知privileged程序。

新增resume_intent_matched_pilots.py：在原控制器精确退出后，重新认证所有已完成
训练/suite，再从“Q7训练完成、suite未认领”接续。新控制器拒绝原进程仍活着、
Q7训练未退出0、suite已有创建记录、后续任务被认领或协议改变。原Q1–Q6结果
独立重算，Q7训练不重跑；剩余Q/P仍使用原初始化、日程、奖励、预算和快照。

42项本地及固定镜像CPU测试通过，实际CPU容器退出0，覆盖上述拒绝场景和
已识别观察器仍不能绕过实际FD检查。测试不代表NPU推理或学习收益。

迁移在本项目4–7卡创建锁内，以完整argv与pidfd精确停止16个监督进程；其中
Q/P只等待Q7资源，其他15个仅等待前驱。旧目录写superseded记录，不伪造成功
终态。冻结新源码和新配置后启动替代进程；已独立确认全部16个精确argv存活。
迁移中曾遇到控制器自引用source-manifest无法提前映射；保留已启动进程，
确认未启动的utility替代目录只有source后，从确切中断处完成剩余六个控制器。
原迁移脚本与修复接续脚本均归档，没有重跑任何训练或重复启动控制器。

Q/P新入口为intent-qp-d12-controller-v2，PID3677691；低学习率入口v2，
utility完整曲线入口v2，FSC矩阵及外部报告入口v4。下游依赖和FSC来源解析
配置一并改指新控制器；七策略选择规则、全量人口及评分配置未变。
16个源码快照差异均只包含resource_observers.py及新增接续入口，所有配置
pins重新检查通过。完整新旧映射与PID见共享迁移目录completed.json。

截至12:09 UTC启动验收，Q1–Q6已重新认证，新控制器仍等待Q7首次验收创建。
它报告另一个未知容器5db1cd7…；随后精确docker inspect返回no such object。
因此不能声称资源等待已完全解除，也未豁免未审计容器或停止其他工程任务。
后续普通观察继续至少600秒间隔。新suite未完成前，不更新已接受学习曲线。

证据根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-resource-archive-migration-v1/`。
[测试及实际接续凭证](../../../intent-resource-archive-handoff-v1-verified.json)。
