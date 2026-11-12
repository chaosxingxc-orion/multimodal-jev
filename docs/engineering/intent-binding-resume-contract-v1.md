# 重绑定训练的来源预算与恢复谱系

2026-11-12 UTC。新增`intent_binding_resume.py`，与原生意图恢复入口分离，
不放宽原178输入验收器。该模块还未接入NPU训练派发。

## 固定约束

新schema要求每步8个来源、每来源4个视图、全局32请求、4个rank。四视图保持
native/shift1/shift30/permutation317顺序，执行精度BF16、短音频v2 profile。
恢复时数据发布及manifest、过程奖励臂D/Q/P、seed、来源预算、LoRA结构、镜像、
运行配置和Adam参数均须相同；更新游标只前进1。第一步新建optimizer，后续
必须指向已接受的前驱adapter与suite，不允许静默重置或改用其他checkpoint。

前驱接受产物必须是新的`intent-binding-update-verified-v1`，明确712个gate
请求、8次来源使用与32次视图使用，并保有4个rank完整状态及输入/输出证据。
从实际数据重新生成前驱来源日程，与保存的selection逐项核对，再检查全部文件
哈希。最后重新执行完整712输入策略文件验收，不只接受一个passed字段。
实际Docker退出仍由派发控制器核验；FP32 master/Adam张量精确恢复仍由rank执行。

## CPU验证

52项配置、恢复及原生入口回归测试通过，0失败、0跳过。负例包括32独立来源
混入8来源视图任务、候选版本/arm/精度/Adam改变、178输入冒充712、缺失rank3
状态、分片被改、前驱来源顺序被改、接受阶段失败及完整gate失败。

实际发布的D/Q/P均重新加载并核对12步日程：每臂96次来源使用、384次视图使用、
89个独立来源；来源顺序与原D/Q/P一致，每四视图权重之和等于原录音权重。
该复核使用假设的未来路径检验配置谱系，没有创建训练队列或接受未来产物。
见`intent-binding-resume-contract-tests-v1.json`与
`intent-binding-resume-source-audit-v1.json`。

另在固定镜像CPU容器
`713cedfc88b71d5d5f0b9c1bee877385ba9007d8d8fdca6d9c6f81e084e481d2`
执行真实四Gloo-rank原生verl更新/恢复探针。每步32视图使用实际发布P臂的重映射
标签、过程权重及来源权重，特征与策略概率为合成FP32。四rank均证明：序列化
恢复后第二步参数及Adam状态与连续更新逐项相等；先模拟BF16舍入损失，再恢复
FP32 master也得到精确结果。故意重置Adam的负对照参数最大误差为0.0163651。
容器实际退出0，镜像、源码、外部监督与结果哈希复核通过，无NPU挂载；临时
状态张量在结果归档后删除。见`intent-binding-resume-dp4-cpu-v1-verified.json`。
这验证实际标签/权重下的CPU恢复，不证明真实语音特征、NPU FSDP或BF16执行。

仍需实现专用NPU训练入口及更新后验证器，由其实际产生上述新schema的接受
产物；不能用此处的测试夹具代替训练完成。完整真实模型收益、跨任务迁移及
连续verl trainer均未由本模块证明。
