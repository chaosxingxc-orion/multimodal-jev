# 常驻行为概率与训练状态的联合认证

2026-11-12 UTC。开发版将已验收的resident行为绑定接入intent恢复认证，
不修改冻结D/Q/P、utility或其他已排队训练快照。

原配置将acceptance_root与previous_suite绑定为同一目录，因此独立重新
推理得到的resident行为即使验收通过，也不能用于原优化器状态的继续更新。
现在显式resident_behavior绑定可以使用不同的acceptance_root；原previous_root、
previous_suite、上一轮adapter路径和完整状态验收仍然保留。

恢复认证先检查原训练协议、目标、优化器配置、恰好一步的游标连续性、
完整178输入验收、BF16导出重载和四rank状态分片哈希。随后用原验收的
base_id与adapter_id认证resident凭证，拒绝“状态属于策略A、行为属于策略B”。
只通过validate_lineage的结构检查不代表恢复已获准；必须通过authenticate。

宿主绑定工具对继续训练配置先验收原恢复状态，再重新查询真实Docker完成
并复算resident suite，最后验收联合配置。两套证据缺一不可。工具不派发训练，
不替换状态验收目录，也不修改原配置。首次训练仍使用原输入绑定流程。

47项本地测试通过，包括原恢复与完整resident凭证测试、新增同策略联合恢复、
状态分片损坏拒绝、两套有效凭证策略不一致拒绝、原状态验收失败拒绝。
测试使用受控Docker与模型身份替身；不能据此声称真实NPU消费成功。
同一47项测试在固定镜像无设备CPU容器中通过，实际退出0，源码哈希与
冻结测试快照一致。[独立镜像凭证](intent-resident-resume-cpu-v1-verified.json)。

当前实际限制：尚无真实多策略resident suite完成凭证，尚未完成新导出策略→
独立HF与resident vLLM→宿主验收→下一次常驻actor更新的在线闭环。
本次连接的是已接受训练状态与已接受推理证据，不是替代这些实际执行步骤。

运行观察归档：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-runtime-observed-1157.json`。
Q7训练实际退出0，原Q/P控制器完整argv存活，仍等待独立suite资源检查。
共享梯度CPU诊断实际运行中，已到首条来源的过程梯度反传，尚无完成结果。
资源快照显示临时privileged归档容器未声明NPU设备；未停止其他工程容器，
未修改原控制器资源判定，也未重复派发Q7。
