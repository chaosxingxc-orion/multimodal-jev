# Omni 分片加载前的位置编码精度核查

结论：后续采用 BF16 权重的冻结概率验证时，不能照搬旧 verl worker 的
整模型 `actor_module.to(torch_dtype)`。当 `torch_dtype=BF16` 时，它会将
原始 FP32 位置编码 buffer 一并舍入；之后 FSDP 的 FP32 buffer cast 无法
恢复原值。文本 RoPE 的原 FP32 路径与此前九次真实预训练 Omni 前向保存的
位置编码逐元素相同，BF16 往返路径在九次调用的有效位置均不相同。

这是具体加载路径的数值差异证据，不是完整 NPU、任务质量或训练失败结论。
旧 worker 默认 actor model dtype 可能为 FP32；本发现针对显式 BF16 加载
方案，不声称所有 verl 配置都会执行 BF16 往返。FP32 权重方案另有容量问题，
不能为了避免该舍入就默认其可在当前两张 64 GiB 卡上完成前向与训练。

## 实际探针

固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
源码 `81c3f6828b609ad1aca7d56e14a7591447a75c36`，容器
`9cfb0a24ff47cd8f27d7f66f14ea8555fb215680fd6d04d9cb47e2243c209044`。
容器 2 CPU / 8 GiB、无网络、无设备，退出 0、无 OOM。探针执行 0.365 秒，
仅挂载原始完整 config，不挂载权重。没有新模型下载或完整模型前向。

`probe_omni_buffer_precision.py` 使用原始 Thinker 配置，通过
`accelerate.init_empty_weights(include_buffers=False)` 和 HF no-init 构造
31,719,205,488 个 meta 参数，保留真实 CPU buffer。这里直接构造模型，
没有执行 `from_pretrained`，也没有验证分布式 rank 初始化。

检查固定 HF/verl 源码哈希后，实际执行整模型 BF16 cast，再调用实际 PyTorch
FSDP `_cast_buffers_to_dtype_and_device` 的 CPU 路径恢复 FP32。没有初始化
FSDP 实例或通信组。四个非持久化 buffer 的结果为：

| Buffer | 元素数 | 变化元素数 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 音频 sinusoidal positional embedding | 1,920,000 | 1,918,343 | 0.001953125 |
| 视觉 RoPE inv_freq | 18 | 17 | 0.001827955 |
| 文本 RoPE inv_freq | 64 | 63 | 0.001154721 |
| 文本 RoPE original_inv_freq | 64 | 63 | 0.001154721 |

这些 buffer 不在 checkpoint state_dict 中；仅校验 checkpoint 参数哈希，
不足以确认它们的运行时精度。图像分支的结果仅作为完整模型 buffer 清单，
当前音频任务不由此得出视觉任务结论。

实际文本 rotary 模块在显式位置 `0..T-1` 上产生的 BF16 cos/sin：

| 序列长度 | cos 最大差异 | sin 最大差异 |
| --- | ---: | ---: |
| 1230 | 1.30078125 | 1.30078125 |
| 4096 | 2.0 | 2.0 |

这是 RoPE 数值的比较，不是 logprob 差异，也没有设置任务通过阈值。逆频率
的微小误差会随位置放大相位差，因此仅检查 buffer dtype 恢复为 FP32 不够。

## 与真实模型历史证据对照

独立核验保存的全部 buffer 和 rotary 张量，重新计算误差，并用乘法及
sin/cos 公式重建合成位置输出，逐元素相同。随后读取已认证的
`omni-actor-prefix-train6-v1/run` 全部九份原始输入证据：六次 batch1、三次
batch2，行顺序 `[2,0,5,1,3,4]`。逐文件校验 SHA，再使用保存的 position_ids
和本次原始/往返 inv_freq 重建 cos/sin。

- 原始 FP32 inv_freq：九次调用的完整张量及有效位置均与历史记录精确相同。
- BF16→FP32 inv_freq：九次调用的有效位置均有差异，最大差异 1.30078125。
- 未重新运行历史 actor、未读取 gold、未以新的推理替代历史证据。

容器资源、124 份源码与 Git 对象、全部保存张量的独立核验及逐调用统计见
[核验记录](../../../omni-buffer-precision-v1-verified.json)。外部核验脚本位于本地
`.local/audits/verify_omni_buffer_precision_v1.py`；原始张量在实验根目录保存。

## 对后续实现的约束

1. 显式 Thinker 加载应在读取 checkpoint 参数时选择 BF16，同时保持原始
   buffer 精度；避免对整个已加载模型再次统一降精度。不能只在降精度后
   `.float()` 恢复 buffer dtype，并声称已恢复数值。
2. 在分片前、分片/同步后分别验收全部四个 buffer 的身份、dtype 和数值；
   包括非持久化 RoPE 原频率副本。rank 间同步一致不等于与原基座一致。
3. 完整 NPU 分片加载仍待实现和实际验证。单层 NPU 探针消费历史保存的
   position_embeddings，不经过这里的 buffer 构造路径；它的结果不能覆盖
   完整模型的 buffer 加载验收。
4. 未改动安装包、现有评测、原 checkpoint 或排队中的单层探针。暂不把
   BF16 往返配置投入完整 actor；真实采样概率、优化器、导出/恢复和跨任务
   效果仍需分别验证。
