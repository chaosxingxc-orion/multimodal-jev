# Omni rollout 到 actor 的概率契约

## 当前证据缺口

已排队的 `infer_sfc_jev_omni_probe.py` 使用 greedy、n=1，检查每次
`response.prompt_token_ids` 与 HF 展开输入完全相等，并保存输入哈希和输出 token。
但 SamplingParams 没有设置 logprobs，记录不含生成 token 概率，也没有 K 个采样。
因此它是冻结任务评测，不能转换成已验证的 RL old_log_probs。
不能补零、用新策略重算后冒充旧策略，或将同一 greedy 输出复制成采样组。
当前运行及排队快照保持原协议。

## 固定源码确认

从固定镜像读取 vLLM `config/model.py`、`sampling_params.py`、
`v1/sample/sampler.py`、`outputs.py`，哈希及原路径见
[源码记录](../../../omni-rollout-logprob-source-v1.json)。

- 默认 `logprobs_mode='raw_logprobs'`，含义是温度、惩罚及截断处理之前的概率。
- `processed_logprobs` 是处理后的概率；不能不区分两者就用于 PPO 概率比。
- `logprobs=0` 仍返回实际采样 token 的概率，只不额外返回 top-k 候选。
- prompt_logprobs 不经过采样处理，不能用它代替生成时行为分布的审查。

这是上游固定源码分析。尚未验证 vLLM-Ascend 实际 sampler 的输出与 HF 前向等价，
也未运行新的 NPU 采样。

## 初始采样约定

`src/multimodal_jev/omni_rollout_records.py:sampling_kwargs` 明确固定：
temperature=1、top_p=1、top_k=0、min_p=0，presence/frequency penalty=0，
repetition_penalty=1，min_tokens=0，保留 EOS 停止行为，不自定义停止字符串。
这样 raw model logprob 与未截断的行为策略概率具有相同定义，actor 也必须使用温度 1。

调用者还必须显式设置 engine `logprobs_mode='raw_logprobs'`、
`generation_config='vllm'`，避免模型 generation_config 覆盖缺省值；
禁止额外 logits processor、grammar、allowed-token mask 和未登记 adapter。
这些条件不包含在 RequestOutput 中，转换函数不能从输出对象反向证明它们。
真正的运行器必须记录并校验最终生效的 engine 和 sampling 配置。

K>=2，replicate 是 engine CompletionOutput.index，独立于完成顺序和奖励。
seed 在派发前登记，不能按成功/失败挑选随机种子。重复文本可以是合法采样结果，
不得因重复而重试或去重。输出预算不得超过当前 1024，实际 actor batch 不得截断。

## 已实现与验证

`capture` 要求完整 K 输出、完整展开 prompt token 相同、已准备请求身份一致、
每个响应 token 均有有限非正 logprob，且 tokenizer 解码与 engine text 完全相等。
严格保留格式失败、length 截止输出和 EOS token；不存在补写或自动修复。
返回字段与 `sfc_reward_batch.compose` 一致，可连接到 `omni_actor_batch.make_batch`。

43 项采样转换及既有分组测试通过，包括反转输出到达顺序后仍维持 replicate 身份，
转换后的记录通过真实奖励组合函数，以及 16 类错位/缺失/无效输入被拒绝。
测试使用合成 RequestOutput 形状对象，未调用真实 vLLM 引擎，也不证明当前版本
的 SamplingParams 构造或 Ascend sampler 运行成功。

## 后续验收

1. 独立采样运行器固定镜像、模型、指令、学习会话及最终生效配置，并在派发前保存意图。
   收集完整 78 决策 × K 的初始工程样本；不修改原有 78 条 greedy 质量评测。
2. 实际引擎返回输入 token 须与 actor processor 一致。逐条记录原始生成概率与终止状态；
   请求状态不明时保留证据，不自动重放。
3. 冻结 thinker 对同一完整 prompt+response 做 teacher-forced 前向，检查有效响应
   token 的概率误差和峰值内存；正确性阈值在运行前登记，不事后随结果放宽。
4. padded/sequence-parallel-size-1 前向及导出映射成立后，才能验证梯度与权重更新。

以上仍是端到端音频任务。这里的 teacher-forced 指固定响应 token 的概率计算，
不是语音转写后调用文本模型，也不是远程提示词教师调用。
