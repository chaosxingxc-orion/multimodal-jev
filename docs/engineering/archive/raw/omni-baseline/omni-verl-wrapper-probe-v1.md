# 真实 verl compute_log_prob 与 Omni batch 入口验证

固定镜像 CPU 容器退出 0，实际记录见
[result](../../../omni-verl-wrapper-probe-v1-result.json)。使用源码 `ecac155`，
2 CPU、8 GiB、无网络、无 NPU，初始化单进程 Gloo 仅满足 actor rank 查询。
没有 FSDP、Ray、梯度或优化器。

本次使用真实 DataParallelPPOActor 构造器和完整 compute_log_prob 方法、
真实 DataProto select/split/to、实际 extract_multi_modal_inputs。
入口为 `omni_actor_forward.compute_log_probs`，设置静态微批、SP=1、无 fused、
无 prefix grouping、temperature=1、CPU BF16 autocast。
显式作用域 CPU logprob 与 device 适配器在退出后恢复，不修改镜像或原生生产 worker。

模型为随机初始化的小型 Omni 文本 MoE 加线性输出头，输入含左右 padding。
三段合成音频特征按双音频样本和单音频样本分组；模型包装器记录其张量顺序，
不运行音频 encoder，也不让这些特征影响文本 logits。
因此该检查只证明调用/拆分/传递正确，不能用于音频语义或真实模型概率结论。

## 数值与数据传递

- micro_batch_size=1 时两次前向接收音频标记 `[1,2]`、`[3]`，有效帧 `[5,6]`、`[7]`。
- micro_batch_size=2 时一次前向接收 `[1,2,3]`，帧 `[5,6,7]`。
- 两种微批方式的有效响应概率最大差异为 0，低于预设 atol=0.04。
- 原 rollout old_log_probs 和原 batch metadata 保持不变；padding 输出清零。
- 所有参数梯度保持 None，weight_updates=0。
- actor 源码 SHA256 与先前镜像审计一致。

该结果比模拟 actor 单元测试多验证了真实 verl 包装器调用，但仍没有验证完整
pretrained Thinker 的微批等价、音频 encoder、多 rank 或 HF/vLLM 概率一致。
独立运行中的 pretrained TRAIN6 音频前向仍需全部结束和独立验收。
后续在同一音频/固定 token 上接入该真实 wrapper，再测量 BF16 微批差异，
不得把本轮随机小模型的零误差推广到 30B 模型。
