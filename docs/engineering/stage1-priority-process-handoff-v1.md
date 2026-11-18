# Stage1 优先与审核进程隔离接管

## 用户要求与范围

用户明确要求：修复经确认后直接替换旧工程脚本，不等待旧前置审核链全部执行完。
本次只改变本地审核和调度：Stage1全量11点完成后继续既有Stage2效率、正式GRPO、
源曲线与迁移流程；不改模型、loss、数据、阈值、容器镜像或已运行包。

`context_isolated_source_audit.IsolatedSourceAuditEngine` 在一个持久spawn进程中执行
原共享来源引擎，最多4个batch审核子进程、各torch线程1；调度进程等待Future，
不与CPU密集来源重建争用GIL。请求串行有界，异常传播并封闭引擎，不自动重试。
共享batch证明仍重新校验所有输入内容；其他checkpoint/master/Adam检查全部保留。

新控制器 `tools.reports.context_stage1_priority_pipeline` 复用原完整流程，调整Stage1
与Stage2准备顺序；Stage1两个点评测的CPU准备、NPU执行和结果收集仍可交错。
每阶段普通远程观察900秒，精确docker wait不改成轮询。

## 接管协议

切换先校验新候选完整来源和真实selection包的差分证明以及全部源码pins。
旧driver通过PID、start ticks、完整argv和pidfd绑定，SIGSTOP防止新增派发。
仅允许已识别multiprocessing resource tracker及准确容器的只读docker-wait子进程。
未知子进程、未知NPU attempt或已有未登记收集状态使切换拒绝，并恢复旧driver。

若旧任务未派发，持有设备租约并检查四卡空闲后结束旧CPU控制器。
若终点318已启动，必须显式绑定其operations、attempt、launch的SHA256，现场inspect
确认原container/image；不停止、不重启远端容器。终止旧本地driver及其只读wait观察者，
新控制器取得同一设备租约后对同一容器wait，收集到新独立目录，再执行原完整审核。
初始点0及其余点在新版本命名空间执行；旧包和证据保留。

完整source/selection证明通过只是本地主机审核能力验收，不等于NPU效果验收。
接管后的终点成绩必须来自原容器全量9226条预测、同策略HF256面板、transport及评分审核。

## 验证与证据

进程隔离测试：真实spawn、PID隔离、重复调用同一进程、退出回收、异常失败不重试。
控制器测试：Stage1→效率→正式→GRPO曲线→迁移顺序；终点接管跳过prepare/launch；
wait→inspect→collect→runtime验证→独立审核顺序。合计11项通过。

首轮v0.0.52的两份来源证明和selection包证明均等价，但验证期间新增源码改变内容键，
最终“一次batch计算”条件未通过。保留失败记录，不将其报告为验收成功。
修订验收v0.0.54冻结源码后重跑，证据目录：
`/srv/multimodal-jev-xingchao/operations/context-isolated-source-audit-v0.0.54/`。

共享模板 `configs/portable/operations/context-stage1-priority-pipeline.json`，本地宏
`STAGE1_PRIORITY_PIPELINE_CONFIG`；所有地址、产物身份、资源绑定从本地宏读取。
发布以实际report、retirement-complete、driver和远端container现场核验为准。

## 已完成验收与接管（2026-11-18 03:24 UTC）

v0.0.54全量验收exit0：两份完整来源证明和真实selection包证明均与原始证明逐字段相等。
完整来源首次93.367秒、第二次45.523秒，selection包重建125.173秒；batch计算1次、
复用3次，工作进程退出已回收。相关进程、顺序、接管和来源链20项测试通过。
这是进程隔离正确性和完整审核等价性验收，不是整体训练提速结论。

v0.0.53控制器已通过实际detach执行接管，未等待旧前置审核全部完成：

- 新driver PID2388164，start ticks354889321，准确argv/身份现场匹配。
- 旧driver2368231已退出，原证据保留。
- 原终点容器`dd41c20680940139e88f1450a366b181b594a21cd47f9f759c0c611a9b29103e`
  现场inspect仍running/nonOOM，StartedAt仍为03:12:30 UTC；没有远端停止/重启操作。
- 新driver准确docker-wait子进程2388175等待原容器；独立审核进程2388172已启动。
- 新队列已记录`adopted-endpoint318`取得设备租约。

本地绑定`.local/config/context-stage1-priority-v0.0.53.json`；运行根
`/srv/multimodal-jev-xingchao/operations/context-stage1-priority-v0.0.53`；接管证据
`retirement-intent.json`、`retirement-complete.json`；新操作根
`/srv/multimodal-jev-xingchao/operations/context-stage1-priority-jobs-v0.0.53`。
原终点结果收集到新操作根的`adopted-infonce-0318`，沿用原launch和原冻结配置，
独立审核后计入完整11点曲线。此时尚无全量评分结果，不宣称Stage1评测完成。

不得再启动旧v0.0.49或未补齐adoption的草案；未来重接管同样要核验实际容器和收集边界。
