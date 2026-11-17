# 真实模型卸载重放已接入队尾

2026-11-12 UTC。使用`intent-residency-replay-prepared-v2`，保留v1准备产物，
新版本在独立HF/vLLM验收后增加真实FP32 master/Adam与原D第2步的对照。
普通ACCEPTED不足以接受该实验，还必须有完整RESIDENCY_ACCEPTED及准确
训练/suite容器实际退出0。四rank、同一来源与监督、BF16 adapter身份和
全部证据哈希分别检查。不是新的学习曲线点或常驻vLLM共存证据。

控制器PID3628148，实际/proc命令匹配、等待绑定首步控制器PID3611061。
顺序保持全部既有训练、迁移评测、绑定与优化对照，在最后追加此工程重放。
前序绑定实验及已排队小模型NPU卸载探针必须实际通过。轮询600秒，设备
检查与四卡启动锁沿用现有方式，不自动重试或重建不明确的容器。

控制器v1误将虚拟环境Python路径resolve为系统Python；启动检查发现后，
确认所有训练/suite均未创建，停止准确PID3628045，并启动使用原虚拟环境
路径的v2。09:25确认v1不再存活、v2实际命令一致。原启动收据和替换原因
保留，不改变冻结计划、学习因素或先后顺序。

12项配置/队列测试通过，覆盖十分钟等待、前序失败阻断、失踪PID拒绝、
实际容器未结束时不接受以及普通验收不能掩盖重放失败。固定镜像另完成
真实D第2步四rank shard的比较器验证：每rank原状态自比较通过，16份
主参数/Adam单元素篡改副本全部被拒绝，包含BF16值未变的FP32改动。
该控制只验证实际布局与区分能力，不是尚未执行的卸载重放结果。
证据见`intent-residency-real-shard-controls-cpu-v1-verified.json`。

启动记录见同名launch JSON。共享根：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-residency-replay-controller-v2`。
