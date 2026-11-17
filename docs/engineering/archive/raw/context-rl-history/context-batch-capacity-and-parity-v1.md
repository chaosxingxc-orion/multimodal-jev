# 新概率头的真实批处理容量与一致性

2026-11-14，同一已更新联合策略67260b78，两条真实音频（2834/3104 tokens），
完整60决策，BF16执行，无额外optimizer更新。固定镜像不变。

## 已完成结果

HF-order对齐容器7e4d2b20已exit0，HF/vLLM全部hidden及60概率逐元素完全一致，
head参数核对相同。见`context-aligned-hidden-gap-v1.json`。只覆盖两输入，
不自动准入完整数据、批处理、更新后策略切换或正式RL。

eager批处理探针35ec0f2a exit1，框架明确报告device OOM：active59.03GiB，
再分配590MiB失败，发生于语言eager attention的前向。完整收集中有已完成的
`hf-serial-batch.pt`，因此失败在后续两请求合批阶段。不能将nonOOM容器
标记误读为没有设备OOM。见`context-real-audio-batch-npu7-v1.json`。

复用已有固定投影+变长融合attention后，edce0b65及b6cd301f均成功完成真实
前向/反向，所有128个训练参数梯度有限。仅在probe中选择新路径。

| v2模式 | 峰值allocated | 测量秒数 | 对serial最大概率差 | 梯度相对L2差 |
| --- | ---: | ---: | ---: | ---: |
| 两请求serial | 6.75GiB | 2.489 | 0 | 0 |
| 同一backbone合批、head合批 | 11.29GiB | 1.137 | 0.00024313 | 0.004209 |
| 同一backbone合批、head逐请求去尾padding | 11.28GiB | 0.869 | 0 | 0.002663 |

时间包含诊断工作和快照持久化，只有一次且次序固定，不能作为稳定吞吐加速比。
梯度来自诊断线性泛函，不是任务loss/收益。不同批处理反向归约有差异，不能
声称梯度或多步参数轨迹完全一致，也不能仅凭前向通过自动准入多卡更新。
证据`context-packed-audio-batch-npu7-v2.json`，完整张量留作独立分析。

新可选`context_row_head.install_row_head`只对head逐请求消费完整有效前缀，
backbone仍一次合批调用，不增加文本生成/解码或跨问题答案反馈。
9项本地相关测试通过，覆盖实际共享head的概率/梯度、padding拒绝和旧音频collate。
当前默认模型与rollout未自动启用。

## 正在进行

959fc458（175卡7）运行融合attention+固定投影+逐请求head的HF/vLLM真实对照，
同权重且无optimizer。新候选HF参考另存`hf-packed.json`，原eager参考保留；
不得把候选与原eager概率变化混入同后端数值误差。
后续需要更广输入上的同checkpoint概率验证、实际多卡head/LoRA更新恢复、
冻结任务reward与训练日程。原完整Stage1审核98629d12与selection自动链继续。

959fc458后续已exit0，融合候选HF/vLLM两音频完整hidden和60prob全部0差，
见`context-packed-hidden-gap-v1.json`。候选HF与原eager有约.00946/.00877概率差，
这是执行profile变化，不属于同profile跨后端误差，原参考保留。

256条学习音频（60类最短/最长，再补长度分位、按长度排列）对照92963435已启动，
含HF单条、micro8批处理与vLLM同策略输出，不看标签选预测、无权重更新。
人口范围2807–3104 tokens，原始输入列表及选择规则在.local对应manifest。

批处理helper改由`model(...,return_logits=True)`进入Module调用，避免绕过未来
DDP/FSDP forward hook；默认model输出仍prob。20项本地模型/头/更新相关测试通过，
包含封装层实际调用与反向，不是实际多卡验证。

首轮任务/实验方案已登记`configs/experiments/slurp-context-event-rl-v1.json`，
仍dispatch=false。集合F1候选奖励与辅助目标共11项本地测试通过；固定镜像组合
loss6210c732 exit0、GRPO组合梯度误差7.45e-9、RLOO0，通过独立公式检查。
v1逐位相等断言失败保留。真实多卡原生verl更新、生产token/runtime身份、策略
切换与任务效果仍未完成。
