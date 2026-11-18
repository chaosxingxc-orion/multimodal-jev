# Omni 文本注意力与 verl 响应对齐 CPU 诊断

目的：在真实 30B actor 前向之前，检查已固定镜像的 Omni 文本注意力中，
纯音频场景使用的累计位置在二维、三轴、四轴表示下是否保持有效 token 输出一致。
本探针只使用随机初始化的两层小型 MoE 文本模块（hidden=24，4 experts），
不加载任何预训练权重，不运行音频 encoder，不读取数据集或启动权重训练。
它不是实际语音模型的性能、HF/vLLM 概率一致性或 NPU 可行性证据。

工具为 `tools/jev/probe_omni_text_positions.py`。固定 seed=17，CPU eager attention，
两条不同长度的序列含 prompt 左 padding 和 response 右 padding。
FP32 下比较三个位置表示的有效 token logits，并对照逐条去 padding 的前向。
修改未来 token 后，检查所有有效前缀 token 输出不变。

响应检查执行固定镜像 `DataParallelPPOActor._forward_micro_batch` 的 AST 原方法体，
使用真实小型 Omni 文本模块加线性词表头，保持 padded、无 fused kernels、
无 prefix grouping、temperature=1。对照独立逐位置 log-softmax/gather，
检查有效响应中的首 token、后续 token 和变长 response mask。
actor 使用 CPU BF16 autocast；事先设定绝对误差阈值 0.04，只用于此合成诊断，
不是未来 BF16 NPU/HF/vLLM 一致性阈值。

## 保留的失败和诊断调整

1. v1 容器在因果性检查失败：最初把左侧全 masked padding 的输出也纳入比较。
   192 个不同元素正好对应三处 padding 的 64 维输出；有效位置和去 padding
   对照已通过。v2 将断言限定于有效 token，不改变输入或模型。
2. v2 通过上述检查后，调用真实 verl `logprobs_from_logits` 失败。
   固定源码按 `NPU_CROSS_ENTROPY_LOSS_AVAILABLE` 选择 NPU cross-entropy，
   没有检查传入 tensor 所在设备，导致 CPU 输入调用 NPU 算子。
3. v3 记录原分派失败，并仅在 AST 诊断的局部 globals 中显式绑定真实
   `logprobs_from_logits_v2` 通用实现。不修改镜像、生产 actor 或全局后端标志。
   因而不能报告“原样 verl actor CPU 路径通过”。

各轮容器、脚本、启动参数和日志保留在共享 experiments 下的
`omni-text-position-probe-v1/v2/v3`，验收记录另存 JSON。
从先前已退出容器读取的 actor、FSDP engine、monkey patch 和概率函数源码位于
`omni-verl-actor-forward-source-v1`。三个初始文件及追加概率函数均记录哈希。

## 已验收结果

v3 精确容器退出 0，前两轮退出 1 且均已保留。验收确认三轮均无 NPU 设备、
无网络、非 privileged，CPU 限制 2 核、内存 6 GiB。
二维与三／四轴的有效位置 logits 最大差异均为 0；逐条去 padding 和有效前缀
因果性检查通过。五个有效响应 token 的 BF16 通用概率结果相对 FP32 参考
最大绝对误差为 0.0128143，低于预设 0.04。原生 CPU 分派错误仍然存在。
详见 [逐轮容器与数值记录](../../../omni-text-position-probe-v1-result.json)。

## 后续接入要求

实际 `compute_log_prob` 还要求 DataProto 的 micro_batch_size、temperature、
use_dynamic_bsz 等 meta_info；现有音频 batch 构造器只提供契约元信息，尚不能
不经 worker 适配直接调用。未来 worker 必须显式检查 padded、SP=1、temperature=1，
保留原始 rollout logprob，确认音频特征迁移到实际 actor device。
本探针没有调用完整 compute_log_prob、FSDP engine 或任何分布式操作。

下一步仍需真实预训练 thinker 的单/双音频前向、与已验收 vLLM 采样的逐 token
概率比较及内存测量；然后才能讨论 LoRA、反向传播和更新。二维位置在本诊断下
成立也不能推广到视觉输入、remove-padding、序列并行或其他注意力后端。
