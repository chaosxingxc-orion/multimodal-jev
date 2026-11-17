# Omni padded actor batch：真实音频 CPU 验证

`src/multimodal_jev/omni_actor_batch.py` 将 SFC-Jev 请求、真实 processor 输入与
已验证的采样记录连接到 verl DataProto。输入保持 waveform + 任务定义，
没有转写中间步骤。接口暂不接入优化器或派发权重训练。

## 数据契约

`prepare(processor, requests)` 校验原始 bundle 和每段音频的文件哈希；按请求处理
有序单/双音频，保留 supplied history 和动态函数定义。每个决策使用真实模板展开，
记录 prompt 与展开 token 的哈希、音频特征及 mask。相同请求的所有决策须产生
完全相同的音频张量。检查帧覆盖与音频占位段数，不截断音频或文本。

不能将不同请求的原始波形随意合并后重新计算特征：现有 processor 的批量 padding
可改变较短音频尾部帧。这里先按原请求处理，再对已计算的特征及 mask 补齐。

`make_batch` 调用现有 `sfc_reward_batch.compose`，复现完整 TRAIN 学习划分，
要求选中会话的所有回合、决策和 K 个预登记采样都齐全。奖励只从学习集目标计算。
同 prompt 的 uid 绑定原请求、策略、指令、实际提示和展开 token；禁止用不同决策
或不同回合凑 GRPO 组。响应文本必须由实际响应 token 解码得到。

prompt 左填充，response 右填充，reward 放在最后一个有效响应 token，
old logprob 按采样顺序逐 token 保存。位置按 attention mask 累积，padding 处为 1。
音频使用 `input_features` 与 `feature_attention_mask`，逐样本保留音频数量和顺序。
上下文按实际 padded batch 宽度检查，超限拒绝，调用者须重新划分微批。

当前 API 的 prepared 张量为 CPU 内存对象；调用者不得在准备与构批之间修改它们。
render 实现、模型 processor 资产和策略身份由不可变运行快照绑定。尚未提供跨进程
序列化或音频特征缓存的完整性协议，不应将该对象视为可接受任意外部输入的安全边界。

## 实际验收

源码提交 `98e9c64`，固定镜像及精确容器见
[验收记录](../../../omni-actor-batch-train6-v1-result.json)。CPU 2 核、内存上限 12 GiB、
无网络、无 NPU 设备；实际探针用时 124.89 秒，容器退出 0。

- 既定四个完整学习会话的 6 个 TRAIN 回合、78 个决策。
- 使用真实本地 Qwen3-Omni processor 和真实录音；响应由 `false` 与
  `invalid output` 两种合成文本编码产生，各带 EOS 和明确的合成 logprob。
- 打乱到达顺序后，156 条响应构成 `[156, 1507]` token 张量，78 个 prompt 组。
- 检查 prompt、response、mask、reward、old logprob 和记录身份逐条一致；
  独立检查 causal shift：首个响应由 prompt 末 token 位置预测。
- 用真实 verl `select_idxs` 重排双音频与单音频微批，再经
  `extract_multi_modal_inputs` 拼接；三段音频的有效特征及 mask 与逐请求结果严格相等。
- 六个负例均拒绝：缺失采样、NaN logprob、解码文本不符、提示哈希不符、
  padded context 超限、prepared 输入 token 被修改。
- 79 项既有奖励、采样身份和任务接口测试通过。未加载模型权重。

仅解析上述学习回合的目标；全 TRAIN 目标文件为绑定版本而读取原始字节计算哈希，
其他目标行跳过解析。未读取 TEST 目标。

## 尚未成立的能力

这不是模型推理质量、训练吞吐或强化学习提升结果。未验证完整 thinker 前向、
HF/vLLM logprob 一致性、音频 encoder 反向传播、LoRA 更新与导出、分布式 actor、
remove-padding 或 sequence parallel。元信息中的路径限制是调用契约，尚无 worker
入口负责强制执行；接入时必须检查 padded、sequence-parallel-size-1 设置。

当前 probe 的 reward 张量检查覆盖默认 independent 模式；function-conjunction
奖励已在独立数值探针中验证，但尚未与本次真实 processor 流程组合验证。
下一步优先核对实际 rollout prompt/token 与 actor processor 的一致性，再进行
保持权重冻结的 thinker 前向与内存测量，不能据本次 CPU 结果直接启动训练。
