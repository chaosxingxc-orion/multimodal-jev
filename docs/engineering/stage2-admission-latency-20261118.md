# Stage2 前置审核时延诊断（2026-11-18 08:00 UTC）

本次为用户主动要求的性能诊断。只读检查和20秒非阻塞采样，没有改动运行包、
没有停止审核或触发新训练，也没有调用模型做日常审核。

## 现场数据

07:58附近，v0.0.77控制器PID2461825已运行28分18秒，主进程CPU时间17分52秒，
专属来源审核进程PID2461852 CPU时间7分47秒。v0.0.79正式后继在pidfd等待。
当时尚无prefix容器launch记录；控制器已持有四卡租约，调用栈处在锁内第二次validate。

两进程累计rchar约15.1GB、92.5GB，合计107.7GB；read_bytes均为0。
这些是累计read系统调用返回字节，包含重复读取和管道，不是新增数据下载，
也不能用它们推算物理磁盘吞吐。采样及计数不支持将本次主要瓶颈归因于磁盘等待。
用户slice CPU quota=-1；观察时主机load约3.6，没有发现该cgroup CPU配额节流证据。

20秒、25Hz非阻塞py-spy采样：

| 进程 | 有效样本 | 观察 |
| --- | ---: | --- |
| Stage2主进程2461825 | 437 | 95.9%包含reference_input_evidence.load调用链；JSON encoder单一叶节点21.5%，其余大量dataclasses.asdict/deepcopy相关帧 |
| Stage2来源审核2461852 | 0 | 采样窗口无活跃样本，不代表此前没有工作；现场wait为pipe_read |
| Stage1曲线来源审核2461268 | 103 | file_hash及其读取lambda合计约84.5%的有效叶节点样本 |

主进程采样62次读取失败，非阻塞样本有偏差；以上是短窗口活跃样本比例，
不是整段28分钟的wall-time占比。不能据此承诺具体剩余时间或加速倍数。

## 根因

1. 全TRAIN reference输入审核逐条compile_candidates：40513×60=2430780个候选对象。
   ContextDecisionRequest.__post_init__对两种保留token各执行一次asdict+json.dumps，
   随后model_input再asdict并计算请求hash。相同候选本体反复构造和深拷贝，单线程执行。
2. 独立封包重建、reference准入、parity准入嵌套调用同一个Stage1来源审核；
   目前只对batch重放做内容绑定复用，完整source/transport/checkpoint/hash仍重新核验。
3. shared.launch在锁外和锁内各调用完整validate；锁内第二轮重做全量来源及reference，
   导致已占用逻辑设备租约但NPU尚未派发。
4. cpu_workers=4仅作用于内层batch重放。IsolatedSourceAuditEngine的外层池max_workers=1，
   用锁串行提交；reference输入审核直接运行在主进程。现有4路配置不能加速当前热点。
5. Stage1曲线和Stage2控制器分别持有独立SourceAuditEngine，两者不共享完整来源证明。
   曲线的每点评测包审核又回到训练来源，进一步重复读取和哈希历史产物。

## 优化顺序

先消除依赖图重复计算，再扩大真正可并行的计算：

- 为完整Stage1来源、reference输入、reference输出各建立内容身份绑定的审核节点；
  同一控制器单次计算并复用，不以未经核验的PASS字符串代替证明。
- 将重审核移到取得NPU租约之前；锁内只做冻结身份、依赖变更、资源和派发原子性检查。
  复用必须有不可变资产/变更检测和失效机制，不能简单删除第二次validate。
- 并行分片核验reference输入，复用不变候选结构；逐条请求hash、全量覆盖、候选顺序、
  音频/张量一致性及异常拒绝与原实现作差分测试，保证结果完全一致。
- 每个审核节点记录调用次数、首次计算、命中、hash字节、CPU/wall时间；用实际NPU开始
  前的总时延验收，不以新增线程数或某个局部函数变快作为成功。

此次未更改运行中的冻结源码；优化需独立版本验收后切换，不能原位热修。

## 证据

原始采样及固定脚本摘要位于：
`/srv/multimodal-jev-xingchao/diagnostics/stage2-admission-v77-{2461825,2461852,2461268}.json`
以及同前缀`-summary.json`。

摘要入口：`tools.reports.summarize_cpu_audit_profile`，宏位于
`.local/config/stage2-admission-profile-{pid}.json`。
主要源码：`context_reference_input_evidence.py:37`、`context_decision_contract.py:42`、
`context_replica_training_operations.py:217`、`context_shared_source_audit.py:64`、
`context_isolated_source_audit.py:54`、`audit_context_infonce_training.py:244`。

## 本次诊断结束时的实际进展

prefix容器`0b312949d83452ca5865dc6cf5706a49764fcb9750be26a77bceb984f390e4c2`
于2026-11-18 07:59:37 UTC启动，docker inspect确认running/nonOOM、固定镜像一致、
物理设备4–7。v77从07:29:44启动到实际容器开始约29分53秒。
这是两步NPU验收，尚不是正式634步训练；未核验到完成的optimizer更新。
