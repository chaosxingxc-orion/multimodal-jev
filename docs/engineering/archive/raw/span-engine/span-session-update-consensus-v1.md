# 常驻训练会话：更新后跨 rank 确认

2026-11-12。检查 actor 更新到策略发布的交接时发现，原会话在本 rank 的
update_policy 返回后立即推进游标。若另一 rank 在优化器已修改参数后报错，
本地成功 rank 仍可能处于 awaiting_publication，无法共同确认此次更新失败。

现在每个返回的 rank 都先进入 update_result 集体确认。任一 rank 的普通异常、
身份/阶段差异或该确认自身失败，使所有参与者进入 failed；不推进已验收游标，
不发布、不重试。优化器可能已更新，保留旧游标不等于回滚；恢复仍需重新加载
最后验收的模型和Adam。进程被杀或阻塞在后端collective中不由此机制恢复，
仍依赖分布式超时和外部控制器处理。

18项本地单元测试通过。固定镜像实际CPU容器退出0，四Gloo rank运行原生verl：

- resident、resumed、paused三个分支各8次正常更新，恢复与暂停分支保持逐值一致。
- 相对独立参考最大梯度误差1.49e-8、参数误差2.98e-8。
- 第9次真实优化器更新完成后，仅rank2抛出注入异常。
- 四rank的Adam步数实际为9，但会话均为failed、游标保持8、策略身份保持policy8。
- 四rank均拒绝发布policy9和再次调用更新；既有旧策略拒绝、跨rank验收证据差异、
  单rank重载失败负对照仍通过。

详见[独立验收记录](../../../span-session-update-consensus-cpu-v1-verified.json)和
[启动记录](../../../span-session-update-consensus-cpu-v1-launch.json)。
这是合成FP32 CPU路径，尚无真实音频/NPU的此项验收；没有改变任何冻结队列快照。
该会话修复将用于后续连续actor集成，现有一步一容器流程不冒称已接入常驻会话。
