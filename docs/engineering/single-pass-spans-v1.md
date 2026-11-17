# 一次前向音频多 Span 实现

对照[任务契约](../research/12-single-pass-span-cot-contract.md)和[Intern-Decision 源码调研](../research/13-jev-multistep-evidence.md)，实现直接音频及完整问题模板的一次骨干前向，使用原有 LM head 读出多个 typed Span。无 generate、KV 解码、外部循环及答案回填。

`SpanRequest` 包含音频、完整问题定义、思考要求和最终位置 ID，不含 targets。Qwen3-ASR 的 stock template 丢弃 user 文本，因此音频放 user，完整问题及占位符骨架放已给定的 assistant context，思考要求放 system。编码器检查全部文本保留、音频展开后占位符唯一且有序、所有音频 token 在首个读出之前。

`NativeAudioSpanModel` 使用原有音频/语言骨干和预训练 LM head。占位符在 p 时读 p−1，多个位置同次取得全词表 logits，候选单 token 符号另做归一化。全词表 masked CE 与候选政策概率是两个不同的归一化空间。现有词表组成的多 token 标记是本项目适配；没有复现 Intern-Decision 训练过的专用 decision token。

早期位置的隐藏表示能通过后续层影响 Final，离散预测答案不会回填。共享输入不等于候选随机采样存在自回归条件依赖。缺失监督用 mask，unknown 则为另一个合法语义类别。并列时间角色不称为 CoT 链。

早期随机分类头 `AudioSpanModel` 保留为实验候选，当前主入口使用 native head。NGA 局部代码检查的[参考凭证](nga-span-readout-reference-v1.json)不能支持整个 Jev 生态不存在多 Span 实现；已通过 Intern-Decision 调研纠正。

## 真实运行证据

v1 位于 `/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-single-pass-spans-smoke-v1`，终态 1：user 文本中的 marker 被模板丢弃。历史快照保持不变。

v2 位于 `/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-single-pass-spans-smoke-v2`，终态 0：本地 Qwen3-ASR-0.6B-hf，BF16，物理卡 4，原始录音 243 个特征帧，600 输入 token，读出位置 286/447/589。骨干 hook 确认调用一次；输出 `[1,3,151936]`，全部有限且三组候选概率归一化。零优化器更新。见[启动凭证](slurp-single-pass-spans-smoke-v2-launch.json)和[验收凭证](slurp-single-pass-spans-smoke-v2-verified.json)。耗时仅单次冷启动前向，不作为稳定性能基准。

相关 CPU 测试 33 项通过，覆盖模型接线、模板丢失、因果读出、审核标签绑定、缺标屏蔽、类型严格匹配及发布文件变更拒绝；不能代替真实训练验收。

## 训练与后续

`span_training.py` 从已审核 release 合并同一录音的决策，核验发布哈希、决策准入和音频身份，输出独立 targets/mask。当前 48 条学习录音合并为 48 个三 Span 请求，142 项有效监督、2 项屏蔽。完整输入中的其他元数据不进入模型模板。

首轮 native Span SFT 协议见[实验记录](../../experiments/archive/raw/span-rl-probes/slurp-native-spans-sft-v1.md)。已完成 24 次真实更新并保存适配器；学习集 72/142，静音 71/142，尚不能证明有效语音判决。48 条录音保存后重载的候选 logprob 最大绝对误差为 0，保存/加载验收通过。之后需要独立发布有真实前置—最终关系的样本、同 Final 对照、跨任务划分，以及 verl/vLLM 多位置推理和政策概率接口。旧生成接口的后端概率误差仍未解决，不能以新 HF 前向成功替代一致性验收。
