# 当前策略的真实音频微批与 verl 损失诊断

> 用户纠正任务契约后暂不派发此旧生成式诊断；优先实现一次前向多 Span typed 判决。参见[当前契约](../docs/research/12-single-pass-span-cot-contract.md)。

状态：工具已准备，尚未派发；先读取实时 HF 音频后端三臂概率结果再决定。不得把历史 CPU 扰动 fixture 的 native actor 测试当作本实验结果。

输入采用当前策略的真实 vLLM 采样，固定两个 learning 来源、六提示、每提示两响应；不根据奖励或正确性挑样。使用同源 BF16 基座+LoRA、真实音频与任务定义，经过 `verl_audio.make_batch` 的左侧提示补齐、右侧响应补齐及音频帧补齐。比较微批大小 1、2、3、6；三响应微批会拆开部分 GRPO 组，但优势只在完整批上计算。

只读前向，不反传、不构造优化器、不热更新采样器。逐响应原 logprob 不替换，阈值仍为有效响应 token 最大绝对差 0.02；掩码之外的补齐 token 不计。完整保存每个批大小的概率、原概率、有效 mask、example ID 和 replicate。

奖励只使用已审核原子判断的实际 grpo 奖励；UID 按完整 prompt 分组，不将同音频不同角色混组。报告非零优势 token 数；若全部为零，损失比较仍可运行，但不构成有效学习信号或策略更新验证。

同一批概率输入下，比较逐微批 vanilla policy loss 按有效响应 token 数加权之和与整批 verl loss，绝对差阈值 1e-6。该比较验证归约，不要求不同微批的模型数值完全一样。分别报告概率门槛与归约门槛，不用外层 `diagnostic_complete` 代替通过。

工具：[audit_audio_microbatches.py](../tools/jev/audit_audio_microbatches.py)。已通过语法检查，尚无 NPU 运行证据。仍缺实际反传/优化器、原生 verl worker、分布式与更新后策略采样验证；此准备不代表这些要求已完成。
