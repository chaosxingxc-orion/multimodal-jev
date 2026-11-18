# Stage1 评测启动前 CPU 审核诊断

## 核实结果

2026-11-18 03:00 UTC左右，用户要求优先锁定Stage1训练与评测。正式Stage1为InfoNCE
区分性监督学习，训练318/318更新、81026曝光已经独立审核；完整selection尚未验收任何点。
训练容器921b51a96963现场inspect为exited/exit0/nonOOM，结束于2026-11-17 18:56:31 UTC。
完整正式效果仍未证明，不能用训练终态或reference缓存审核替代selection成绩。

冻结v0.0.49 driver PID2368231、start ticks354628961现场身份匹配且存活。
终点318、初始0两点已经创建各自finalization/config.json并进入启动前validate；
尚未到ReadyQueue.slot，events为空，没有该版本NPU attempt。03:00 UTC远端检查时
没有本项目评测容器，物理4–7卡没有进程。以上是现场快照，不是持续状态承诺。

## 直接证据

使用py-spy0.4.1对同一准确PID只读堆栈采样30秒、10Hz，332 samples、0 errors。
工具独立安装在本地diagnostics，不修改实验Python环境、镜像、runtime或冻结源码。

- 原始speedscope：`/srv/multimodal-jev-xingchao/diagnostics/stage1-admission-v49-20261118.json`
- 确定性汇总：同目录`stage1-admission-v49-20261118-summary.json`
- 汇总脚本：`tools/reports/summarize_cpu_audit_profile.py`
- 宏模板：`configs/portable/operations/cpu-audit-profile-summary.json`
- 本地配置：`.local/config/stage1-cpu-admission-profile.json`

汇总只做规则统计，不调用模型。5项测试通过，覆盖递归frame不重复计权、leaf/inclusive
区别、权重/索引/单位及长度错误。重复执行同输入时保持幂等，输出不同则拒绝覆盖。

| 活动线程 | 采样权重（秒） | 栈上实际工作 |
| --- | ---: | --- |
| independent-branches_1 | 20.0 | Stage2有限验收→reference准入→parity→来源重建→原始check_batch |
| infonce_0 | 6.8 | 终点318启动validate→selection包重建→来源/FP32 master/Adam/BF16检查 |
| infonce_1 | 6.4 | 初始0启动validate→selection包重建→同类来源检查 |

权重是活动线程采样时间，并发可重叠，不能把它当独占墙钟比例。两次栈快照中评测线程
从master/optimizer校验推进到下一次来源transport哈希，未见死锁证据。

## 原因与下一步

当前调度器“CPU线程并行”仍把Python密集来源重建放在同一进程，争用GIL；Stage2
尚未接入共享来源引擎。Stage1只复用batch证明，完整checkpoint、optimizer及transport
仍因prepare/validate/runtime前驱递归重复。这解释了NPU尚未ready的实际CPU开销；
本次未观测到模型加载失败，也不能归因于vLLM或NPU计算慢。

短期保留已在推进的准确controller，不重启、不重复派发终点评测。原900秒普通监控
间隔继续适用；本次是用户要求收口Stage1后的有界故障诊断，不启用高频模型轮询。
Stage2正式训练在现有控制器中依赖完整Stage1曲线，不会以本次诊断替代准入。

后续工程改动应是独立候选：CPU分支使用隔离进程；Stage1先就绪先提交NPU，
Stage2重建不争用Stage1的Python解释器；共同来源证明使用完整输入内容键合并。
保留全部checkpoint/transport数值检查，不能为赶评测跳过准入。候选需验证完整来源
与启动准入等价、异常传播、设备租约以及不重复派发，再在任务边界切换。
v0.0.50/51只解决reference输出/输入叶节点复用，尚不能宣称以上整链问题已经修复。
