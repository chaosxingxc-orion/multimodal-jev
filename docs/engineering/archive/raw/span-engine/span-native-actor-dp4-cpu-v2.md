# 原生 typed actor 四进程更新验证与 NPU 接续

2026-11-12 UTC。固定镜像中四个真实 Gloo/DDP rank 已完成原生 verl actor 验证。每步8个独立合成请求、64个联合动作，各 rank 接收两个完整请求组；真实 DataProto 经对象通信分发，按 microbatch=8与16分别连续更新两步。

四 rank 的梯度、参数和 Adam 状态均与同数据单卡全量计算一致：最大梯度误差1.49e-8、参数误差2.98e-8、Adam 一阶误差1.86e-9、二阶误差3.64e-12。两步分别使用新行为概率和新策略身份。旧概率未被 actor 重算值替代。四 rank 均拒绝单 rank mask 异常、重复请求、不同 rank 权重和被改动的 old logprob；拒绝发生在模型通信之前。

证据见 `span-native-actor-dp4-cpu-v2-verified.json`。本次是合成 float32 CPU 数据，未验证音频、BF16、FSDP、NPU、Ray 或 vLLM 生命周期；只使用真实 DDP 归约证明原生 actor 的完整请求组全局权重与连续两步计算。

真实音频接续已冻结于 `span-native-actor-npu4-prepared-v1.json`，尚未派发。沿用已验证四卡探针的相同初始化、8录音、每请求32动作、随机种子、学习率和全候选概率，替换更新入口为原生 actor。保持 BF16 LoRA 执行、FP32 master/Adam、FSDP FULL_SHARD，只映射物理4–7卡。启动顺序在既有恢复、独立验收和原生任务44请求后端验收之后。

后置验收工具 `verify_span_native_actor_npu.py` 先调用既有48输入 HF/vLLM 验收，再比较固定的显式 loss FSDP 参考：输入游标、master、Adam 及 BF16 导出。master/Adam 使用预定 atol=2e-7、rtol=2e-5，BF16 adapter 要求逐元素相同。该新验收工具尚无真实 NPU 输出可消费，不能称 NPU 已通过。

此比较只验证两个工程更新入口等价，不是 RL/SFT 效果比较。按用户最新要求，后续任务效果主线是固定 RL 起点与连续检查点的指标曲线，不新增 SFT 对照实验。Omni 作为任务能力参照，跨任务泛化仍单独验证。
