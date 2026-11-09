# 音频 RL 接入审计与当前组件

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
这是待接入 verl 的奖励函数，不是已训练或已验证有提升的 pair-GRPO 算法。

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

[CPU LoRA 检查记录](audio-policy-smoke.json)：真实官方 0.6B 模型与 FSC 音频，
146 个 prompt token、6 个 response token；只更新 decoder q/v projection LoRA，
573,440 个可训练参数，56 个参数张量在一次 AdamW 更新后变化。
不同 schema 的 token IDs 不同、音频特征相同；prompt loss mask 和 causal CE/log-prob
一致性通过。vLLM 目前仅验证请求结构，未进行实际引擎推理。

这证明音频条件训练在 CPU 上具备一次真实前后向闭环；不证明模型已学会任务，
不证明 NPU 支持，不证明 GRPO 或成对奖励有效。
