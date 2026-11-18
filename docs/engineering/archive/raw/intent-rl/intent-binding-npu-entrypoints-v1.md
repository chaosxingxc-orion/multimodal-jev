# 重绑定四卡训练与更新后验收入口

2026-11-12 UTC。新增三个独立入口，未改冻结D/Q/P训练、原178输入验证器或
正在执行的队列，尚未派发重绑定NPU训练。

- `train_intent_binding_npu.py`：复用现有已执行的四卡FSDP FULL_SHARD、BF16
  LoRA、FP32 master/Adam及原生verl更新逻辑。改用binding配置、712输入前置
  验收和恢复谱系；每步8来源×4视图、每rank2来源×4视图。结果分别登记source、
  view和gate IDs，不能将32视图计作32条独立录音。
- `verify_intent_binding_npu.py`：重新生成来源日程与重映射标签；核对实际外部
  supervision文件、更新前行为概率及actor概率、rank分配与全局权重、112个LoRA
  执行dtype、FP32 master/Adam张量及适用的精确恢复。独立HF/vLLM检查完整712
  输入；本批更新后FSDP输出须与HF导出重载逐项相同。返回binding专用接受schema。
- `run_intent_binding_acceptance.py`：先检查binding配置和712人口，再运行独立
  HF/vLLM gate，最后调用状态验收器；任一步失败不发布ACCEPTED。

更新后状态验收器检查固定镜像与卡位登记，但不在验收容器内部查询Docker。
派发控制器仍负责观察训练容器及独立suite容器实际退出0，再允许下一轮。
既有freeze工具可提供不可变产物冻结机制；新协议/配置与排队入口仍需准备和
验证，不能直接套用原178输入controller。

## 已执行验证

固定镜像CPU容器
`dbf7a9da37d0dcfa8094782c41f882444dbe3ad427cfe5e79446c667394417c6`
实际退出0，无NPU挂载，55项测试通过、0跳过。集成夹具生成四rank的真实CPU
张量文件及首步/恢复步状态，但dataset和完整策略gate函数被mock；验证的是
新验收器的文件/张量判定，不是模型训练。负例包括标签/过程权重被改、更新前
概率错误、更新后概率错误、view/source/gate IDs错位、精度错误、Adam二阶矩
非法及恢复master不一致等。另5项本地测试覆盖独立后端→状态检查→接受发布
的顺序与失败不发布行为。证据见同名CPU verified和acceptance-control JSON。

训练入口已做语法与对原生入口的差异检查；尚未运行实际重绑定NPU模型更新，
也未验证新入口的端到端显存和耗时。前置712输入的真实策略gate仍在原队列中。
08:25 UTC普通观察确认D/QP控制器实际存活，D第11步suite容器
`7d7895e9bf990c8f5ca002b86f89568677906bd970cc35438668b43ec6ecba5e`
实际运行中；D接受进度仍10/12。没有把运行中验收视作失败或重启。
下一次普通观察不早于08:35 UTC。

后续首步D训练与独立suite已冻结并通过实际CPU容器挂载/导入检查，见
[首步计划](intent-binding-first-step-plan-v1.md)。尚未创建NPU任务或新队列，
真实712输入前置策略验收和原队列顺序继续有效。
