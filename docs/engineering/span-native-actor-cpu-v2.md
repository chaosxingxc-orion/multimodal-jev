# typed Span 接入真实 verl actor：CPU 验收

2026-11-12 UTC。固定镜像中的真实 `DataParallelPPOActor.compute_log_prob/update_policy` 已通过合成 CPU 验证。运行回执与独立验证分别见同目录 `span-native-actor-cpu-v2-launch.json`、`span-native-actor-cpu-v2-verified.json`。

`span_native_actor.py` 用真实 DataProto 传递完整请求组。一个 response 位置承载整个 Span 联合动作；完整音频输入、候选和 Span 位置保存在 multi_modal_inputs 中。传输占位 token 不进入模型。GRPO advantage 在完整组上计算，再由原生 actor 切分 microbatch、累积梯度并执行优化器。只替换 typed forward，并限制每次一个完整 minibatch、一个 epoch；每次更新后必须重新导出、同步和验收策略。

三个合成请求分别覆盖60分类、六个布尔 Span、3类与2类组合；每请求8个联合动作。microbatch=8与24都与显式全量 verl joint loss 对比。前者梯度、参数和 Adam 状态误差均为0；后者梯度最大误差1.86e-9，参数误差0，Adam 一阶/二阶状态最大误差1.82e-10/4.22e-15。梯度范数0.32758，实际发生非零更新。

DataProto 经 pickle 往返后仍通过原生 select/split 与更新。旧策略身份、被改动的 old logprob、打乱的组内顺序、不完整采样组均被拒绝。行为概率保持不变，未使用 actor 重算值替代。固定 actor 源码哈希为 `207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce`。

本次只证明 float32 合成 CPU 模型的真实 actor 循环；不证明真实音频、BF16 LoRA、多 rank FSDP、Ray 传输、恢复和 vLLM 生命周期。payload 身份在本地序列化后成立，分布式传输仍需验证。各 rank 须持有等量完整请求组，后续另验全局权重与单卡参考一致性。此结果不能计入 SLURP 学术成绩。

后续顺序保持：现有评估收尾 → 四卡原生恢复与独立48输入验收 → 60类/55类型44请求后端验收 → 真实 typed 原生 actor 的分布式连续更新与重启对照。按用户最新修订，全本体监督准入独立完成后，跟踪 RL 起点与后续检查点的任务提升，Omni 保留为能力参照；不新增 SFT 对照实验。
