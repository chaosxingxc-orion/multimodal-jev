# 双信号 RL 新进程恢复验证

恢复验证使用同一有限 trainer 和 service，恢复第一步已提交的 FP32 LoRA/head
master 与 Adam，在新容器执行第二步，与原连续两步结果核对。冻结主模型始终
只读引用；恢复包不复制模型、adapter 或 checkpoint。

`tools.reports.prepare_context_dual_replay` 通过宏配置调用完整原 prefix audit，
重新核验归档后才生成恢复包。它保持 source 树和 RUNTIME-SOURCES.json 字节不变，
固定参考策略、目标函数及运行时身份不变，只修改 supervisor 的 start1→stop2
游标和外部已提交断点绑定。四卡分片、policy、adapter 全部加外部 hash pin。
新 reference cache 从空目录开始，恢复不依赖旧进程内存缓存。

同一个 `context_dual_training_operations` 入口支持 initial/replay 两种有限模式。
replay 额外只读挂载原 run，重新执行原 prefix audit，核对原容器身份和保存文件，
沿用原设备锁、真实空闲检查、唯一容器 handle、原 collector 和 runtime inspect。

同一个 `audit_context_dual_training` 对恢复运行核验：

- 原 prefix 必须是 start0 的独立通过记录，不接受另一 replay 作为父运行。
- 新容器 ID 不同，四 rank 均报告 exact master/Adam 恢复至 step1。
- 第二步仍通过真实日程、双信用独立重算、parity、LoRA/head 保存和激活检查。
- 第二步 groups（含动作/奖励/优势）、execution、四 rank master/Adam checkpoint，
  与原连续训练逐张量精确一致；batch metadata 和最终 policy ID 一致。

共享入口分别为 `context-dual-replay-prepare.json`、
`context-dual-replay-operations.json`、`context-dual-replay-audit.json`，均在
`configs/portable/operations/`。初始两步和恢复继续使用同一工程，不引入第二个训练器。
启动仍默认禁止派发，NPU 实测尚未发生；真实恢复包只有父运行实际验收后才能生成。

19 项本地测试通过，覆盖源码/参考保持、完整分片绑定、只读父挂载、错误 contract/
证据拒绝，以及原操作入口的锁和超时路径。测试使用合成父归档，只证明封包/约束
逻辑；此前真实 CPU DP4 更新恢复的证据另见既有 dp-resume 报告。固定镜像本轮
19 项全部通过、零跳过，容器
`08bab7f868d7f5774f14ba75790feb91114fc9ab22d13d863d5fa367a312d177` 已准确 wait0
并独立收集，非 OOM、417 文件。结果见 `context-dual-replay-fixed-image-cpu-v1.json`，
不能称为真实 NPU 恢复通过。

后续固定镜像 v2 将 master→BF16 导出重建、独立双信用组件核验和原恢复/启动约束
合并检查，44 项通过、零失败、零跳过。容器
`600cd1a17f1d3d1e22bd1b7b2b97f2bfa9245cf8c0f2e969b1ad9e6fee1160f0`
准确 wait0，独立收集确认正常退出，源码快照 420 文件。配置入口为
`configs/portable/operations/context-dual-replay-cpu-tests-v2.json`，报告为
`context-dual-replay-fixed-image-cpu-v2.json`。这些是 CPU 合成夹具检查，仍不代表
完整真实归档组合审计、新进程 NPU 恢复或训练提速通过。旧训练和 NPU 封包未修改。

仍需有限实际音频 probe、连续更新和恢复；当前两个性能对照包没有实际收益报告。
资源占用不通过终止其他服务来绕过。用户已授权在确定效率收益后切换旧训练，
恢复检查提供必要证据，但不单独证明持续吞吐或模型效果。
