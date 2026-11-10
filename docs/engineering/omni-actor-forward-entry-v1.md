# Omni padded actor 概率重算入口

`omni_actor_forward.compute_log_probs` 补齐已构造 DataProto 到 verl
`compute_log_prob` 的调用参数，不加载模型、不创建优化器或派发训练。
调用方仍须验证模型、源码、采样和数据身份，并使用 `omni_actor_batch.make_batch`
构造完整学习组。这个入口不是接收任意外部 batch 的认证边界。

支持的路径固定为 padded、无 fused kernel、无 prefix grouping、无 dynamic batch、
sequence parallel size=1。每次重算固定 temperature=1，设置显式微批大小和 pad ID。
验证 response 的右 padding、input_ids 后缀、attention mask 和有效 old logprob。
传给 actor 的张量仅包含 responses/input_ids/attention_mask/position_ids，
保留按样本的 multi_modal_inputs；old_log_probs、奖励和分组数据不传入重算接口。
创建独立 metadata，不改原 batch 的温度或微批设置。

输出要求形状与 response mask 一致，有效 token 概率有限且非正；padding 清零。
返回值独立交给调用方比较，绝不写回或替换原始 rollout old_log_probs。
原 batch 的输入张量可能与 actor 输入共享内存；这是受信任 verl actor 的调用契约，
不是防止任意 actor 原地篡改输入的隔离措施。

## 验证范围

固定镜像中的 CPU torch 测试使用模拟 actor 和 DataProto select 替身：验证参数、
字段、原 metadata/old logprob 不变、padding NaN 排除，以及不支持的路径和
有效位置 NaN 被拒绝。测试不证明实际 verl wrapper、真实 Omni 微批或 NPU 前向。
首两次测试容器因快照包导入不完整在收集阶段失败，分别为缺少 __init__ 和其依赖；
第三次提供完整本地包，不更改被测实现或测试。失败记录保留。

真实 pretrained Thinker 六条 CPU 音频前向仍在独立快照运行，尚未调用本入口。
待该探针完成且独立验收后，才执行实际 actor wrapper 对同音频/token 的比较。
已知 CPU device/logprob 分派需要显式适配；本入口不自动安装补丁，也不隐式回退。
