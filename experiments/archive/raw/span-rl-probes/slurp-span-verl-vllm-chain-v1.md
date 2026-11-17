# SLURP BF16 连续更新工程验收 v1

本实验验证多个新策略版本能否连续传递真实 Adam 状态、均衡样本游标及
训推一致性验收；不用于模型选择，不读取测试反馈，不声明泛化收益。

使用固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`、
本地 Qwen3-ASR-0.6B、原生音频+完整三 Span 问题模板、BF16 未合并 LoRA，
物理 NPU4 顺序运行 HF actor 与 TP1 vLLM。FP32 master/Adam 与损失归一化。
这是单卡训练链；此前四卡 FSDP 单步是另一条 optimizer 时间线，不混算。

从 `slurp-span-verl-vllm-resume-v1/run/policy-1` 的累计第 2 次更新恢复，
只执行接下来的两个互补样本对（batch_index 2、3），预期累计至第 4 次。
每次两条录音、三个任务均正负各一条，每条采样 32 份完整 Span 动作。
奖励为三个已审核并列标签的平均正确性，使用 verl 联合 GRPO/PPO loss；
当前 date/time/timeofday 不构成有依赖的决策 CoT，不增加虚构过程奖励。

每个版本必须先完成真实 actor 更新、独立重载及当前输入的 vLLM 验收，
再在固定 48 条音频上独立重载 HF 并验证全部候选/联合概率。
候选 logp 误差上限 0.02、完整动作概率比 [0.98,1.02] 保持不变。
只有完整验收后才能发布 accepted version 并进入下一步。
失败停止，新建目录恢复上一个已验收 checkpoint；不重放不确定更新。

源码与输入均固定哈希，容器离线、两小时运行上限。独立验收工具
`tools/jev/verify_span_training_chain.py` 将检查真实 checkpoint/Adam tensor、
游标、版本传递、独立 HF 参考及概率；READY 标记本身不足以证明成功。

启动记录见 `docs/engineering/slurp-span-verl-vllm-chain-v1-launch.json`。
启动时未完成；最终结论须以终态和独立 verified 文件为准。
