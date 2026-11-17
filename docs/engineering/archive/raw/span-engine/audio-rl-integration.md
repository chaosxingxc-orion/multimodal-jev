# 音频 RL 接入审计与当前组件

## 当前状态（以此节及链接结果为准）

2026-11-11 收尾：Omni 原始单层 NPU 探针九次前向已完成并独立核验，见[审计](../../../omni-npu-layer-final-audit-v1.json)。物理卡4，零权重更新；micro1/micro2 有效位置最大绝对差0.0078125，未预设数值通过阈值，不判为概率等价。分配器峰值约1.71GB不代表完整模型容量。停止扩展Omni验证，后续优先clean SLURP/Qwen3-ASR；此探针不证明完整actor/FSDP/verl链路。

下文保留早期接入审计，部分“尚未实测”描述代表当时状态。当前已经完成真实
CPU vLLM 采样、HF actor 的 verl DataProto/GRPO/PPO 更新、模型导出及第二轮
重新采样与优化器继续更新，见[两轮结果](../../../vllm-two-step-chain-v3-result.json)。
[变长批次结果](../../../vllm-mixed-actor-v1-result.json)还验证了两条录音的完整投影更新。
这些是工程闭环，不能推导出算法收益；现有结果尚未证明跨任务泛化提升。

当前执行器仍由本项目串联采样、actor 与导出进程。使用 verl 的数据与损失函数
不等于完成原生 Ray/FSDP worker 接入。固定镜像的原生 worker 已重新做只读源码
审计，见[接入差异](../../../verl-worker-integration-audit.md)。默认音频 server、分布式
worker、NPU 训练和交互式 Agentic Speech 仍需实现并实际验证。

新增可选预算控制支持 SFT/RL 的整组预留、实际记账与预算停止，见
[协议](../../../../../experiments/archive/raw/early-rl-engineering/token-budget-comparison-v1.md)。45 份配置仅完成静态准备；
真实预算验证与正式有效性实验必须分别报告，不能把配置数量当作已完成实验量。

本轮源码审计基于镜像 `52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。

## 已实现组件

`audio_policy.py` 为 HF actor 与 vLLM request 使用同一个序列化 schema：
放在 Qwen 官方模板保留的 system turn，user 只承载音频。监督 target、参考转写
和 source metadata 不进入模型输入。vLLM 接收原始 prompt 和 `(waveform, 16000)`，
让自身 processor 展开一次音频占位符；不能把 HF 已展开的 IDs 再次提交展开。

Teacher forcing 手动追加 JSON 答案及 `<|im_end|>`，prompt labels 全部 -100，
response log-prob 使用前一个位置预测下一个 token，包括结束 token。此实现为单样本，
尚未宣称 batching、padding、packing 或分布式 actor 支持。

`pair_reward.py` 实现完整 source_group 的四种成对奖励：候选排列、定义重绑定、
候选移除和正反 Noul。仅两端均语义正确才给成对奖励，权重 .2；
每个样本按参与边取平均，避免 native 因参与三条边而自动得到三倍奖励。
缺视图、重复视图、跨 audio/split、无效定义映射立即报错。
`paired_rollout.py` 已将它接入 `rl_update.update_actor` 的 verl advantage 前处理；
尚未完成真实 vLLM rollout 到更新的端到端验收，也没有 pair-GRPO 效果结论。

## verl 尚缺的连接

| 路径 | 当前固定版本行为 | 需要实现与验收 |
| --- | --- | --- |
| `utils/dataset/rl_dataset.py` | image/video；向 agent loop 返回 raw_prompt | 自定义 audio 数据集，长度过滤与无监督信息泄漏检查 |
| `experimental/agent_loop/single_turn_agent_loop.py` | process_vision_info | 音频 agent loop、真实 waveform 与 metadata 传递 |
| `experimental/agent_loop/agent_loop.py` | server manager 只转 image/video；默认 multimodal position 调 get_rope_index | 音频 server 转发、actor 音频输入和 Qwen3 文本 position 对齐 |
| `workers/rollout/vllm_rollout/vllm_async_server.py` | generate 只接 image/video | 显式 audio request；不能只在客户端加参数 |
| actor/rollout checkpoint | HF 转换与 vLLM thinker 配置不同 | 同 checkpoint 加载映射、LoRA 同步、逐 token log-prob 对照 |
| reward manager | 普通独立 rollout reward | 保存 group/variant/replicate 对齐；完整成对奖励再做 GRPO advantage |

vLLM 的 Qwen3-ASR 类有 HF 权重名称 mapper，但构造器使用其自带 thinker config；
类注册和 request 构造成功均不能证明 NPU 加载/rollout 成功。

## CPU 优化器边界

torch-npu 2.10.0.post7 修改 AdamW 自动 foreach 检测，在纯 CPU 参数更新时
也可能调用 `npu.current_device()`，导致无设备容器初始化失败。CPU 对照明确使用
`foreach=False, fused=False`。不通过挂载忙碌 NPU 修复 CPU 检查。
首次失败保留在 `experiments/multimodal-jev/audio-policy-smoke-v1`；修复另开 v2。

## 实测边界

[CPU LoRA 检查记录](../../../audio-policy-smoke.json)：真实官方 0.6B 模型与 FSC 音频，
146 个 prompt token、6 个 response token；只更新 decoder q/v projection LoRA，
573,440 个可训练参数，56 个参数张量在一次 AdamW 更新后变化。
不同 schema 的 token IDs 不同、音频特征相同；prompt loss mask 和 causal CE/log-prob
一致性通过。vLLM 目前仅验证请求结构，未进行实际引擎推理。

这证明音频条件训练在 CPU 上具备一次真实前后向闭环；不证明模型已学会任务，
不证明 NPU 支持，不证明 GRPO 或成对奖励有效。

## verl DataProto 与 GRPO batch 适配

`verl_audio.py` 使用固定镜像内真实 `verl.DataProto`、
`compute_grpo_outcome_advantage` 和 `compute_policy_loss_vanilla`。

- `uid` 是同一个 example/schema 的身份；不同候选视图仅共享 `source_group`。
  禁止用录音 group 代替 GRPO prompt group，否则会把不同问题的难度差当优势。
- 同一 uid 必须带完全一致的模型输入和 waveform，且有完整的重复采样组。
- HF prompt 左 padding、response 右 padding；reward 放在最后有效 response token。
  位置编码从 attention mask 累计生成，padding 不参与目标。
- 音频帧 pad 到当前 batch 最长，保留 `input_features_mask`，使 verl 的
  `extract_multi_modal_inputs` 可以沿 batch 维拼接。
- 仅支持不移除 padding、无 sequence parallel 的初版 actor 路径；默认视觉
  agent loop/server 的音频扩展、vLLM 实测和权重同步仍待完成。

CPU 集成 smoke 的响应是受控 fixture，用于检查奖励方向、梯度、变长音频和
padding。它不能替代 on-policy rollout，更不能作为 RL 有效性实验。

[成对奖励数值检查](../../../verl-pair-reward-smoke.json) 在同一固定镜像内通过：
6 个真实 FSC train schema、每题 2 个受控响应；真实 verl DataProto、GRPO
与 PPO 梯度。仅改变一侧 rebind 答案时，相同 native 单题分数通过配对 bonus
产生相反优势方向；其他等分组优势仍为零；全 malformed 的 bonus、优势为零；
padding 梯度为零。这里的答案与旧 log-prob 是测试 fixture，不是模型采样。

[实际 actor 对照](../../../verl-actor-parity.json) 使用官方 Qwen3-ASR 0.6B HF 权重，
两种长度的真实音频、不同长度的 schema 和答案，4 条 batched teacher-forcing
与逐条 unpadded 前向比较。有效 response token 的最大 log-prob 误差
5.722e-5，通过预设 atol 3e-4 / rtol 1e-5。这确认当前音频 padding 与 mask
在该真实模型上的数值一致性；还没有比较 vLLM 产生的 token log-prob。

[verl 数值检查](../../../verl-audio-smoke.json) 已通过，覆盖两个 prompt、每个两次受控
response；真实 DataProto、音频拼接、GRPO advantage、PPO loss/backward，
以及不完整 group/同 uid 不同 schema 的拒绝。正优势对应增加 log-prob 的梯度方向，
padding 梯度为零。响应仍为受控 fixture，不冒充采样或算法效果。
