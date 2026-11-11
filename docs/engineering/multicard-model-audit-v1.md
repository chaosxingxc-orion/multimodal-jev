# 当前训练模型与多卡路线核查

2026-11-11 用户询问多卡训练及是否使用 Qwen3-30B-A3B。本次核对实际运行配置、
本地模型 config/index、现有 Span 源码与 175 服务器设备状态。

## 实际已运行

训练模型为 `/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf`，
`Qwen3ASRForConditionalGeneration`，文本骨干 Qwen3 dense，28 层、hidden_size=1024。
已验证 BF16 q/v LoRA（rank4/alpha8）、单物理 NPU4、verl GRPO/PPO 联合 Span 损失，
vLLM TP1 pooling。已完成累计两次真实 RL 更新及各版本 48 条跨后端验收。
这些结果不属于 30B Omni，不证明多卡训练。

证据：`slurp-span-verl-vllm-resume-v1-launch.json`、
`slurp-span-verl-vllm-resume-v1-verified.json`、`slurp-bf16-resume-suite-v1-verified.json`。

## 本地 30B 资产

| 目录（位于共享 models） | 模型类型 | index 参数量 | BF16 index 字节数 |
| --- | --- | ---: | ---: |
| qwen3-omni-30b-a3b-instruct | qwen3_omni_moe | 35,259,818,545 | 70,519,637,090 |
| qwen3-omni-30b-a3b-thinking | qwen3_omni_moe | 31,719,205,488 | 63,438,410,976 |

两版 Thinker 文本配置均为 48 层、hidden_size=2048、32 attention heads、4 KV heads、
128 个专家、每 token 选择 8 个专家。Instruct 包含 Talker/code2wav；
当前只需音频输入和 typed decisions，可使用 Thinker 路径，保留原始 checkpoint 不裁剪覆盖。
Thinking 是 checkpoint 版本名称，不代表已实现本项目一次填充式决策 CoT。
普通 Qwen3-30B-A3B 与 Qwen3-Omni-30B-A3B 应区分；本地此次核对的是 Omni。

A3B 是激活参数规模，不是全部驻留权重大小。
已核验的完整 Thinker 参数共 31,719,205,488，BF16 约 59.08 GiB；
四卡理想均分仅权重约 14.77 GiB/卡，不含 all-gather、激活、通信、LoRA、优化器和工作区。
容量是否满足须真实测量，不能由参数除以卡数保证。

## 多卡候选与缺口

本次 `npu-smi info`：175 为 910B3，每卡 65,536 MiB HBM；4–7 卡无进程，
显示约 3.44 GiB 基础占用；0–3 有其他服务。该状态是当次快照，派发前重查。

建议先在授权 4–7 卡顺序运行 actor 与 rollout：

- actor：显式 Omni Thinker 加载，四 rank FSDP FULL_SHARD，BF16 前向，
  FP32 梯度归约/概率/损失和 optimizer state；先仅 attention q/v LoRA，冻结基座与路由器。
- rollout：四卡 vLLM TP4、相同未合并 BF16 LoRA，音频与完整问题模板一次 Span pooling。
- 交接：actor 释放后 rollout 加载不可变 adapter 版本，验收完成再继续更新。
  初期不将 actor 和 rollout 同时驻留这四张卡。

这里是候选设计，尚未执行/验收。不能把当前 TP1 配置的数字改成 4 就宣布支持：

1. 现有 `NativeAudioSpanModel` 和 vLLM Span 子类针对 ASR；需要适配 Omni Thinker 输入、
   hidden state 与 LM head，保持直接音频及单次 typed Span 读出。
2. 固定 verl 的默认 AutoModel 选择与 Omni 类不匹配，默认 FSDP wrap 名称还含失效层名；
   既有探针只验证显式加载及 wrap 结构，完整 NPU FSDP 初始化/反向仍未验收。
3. Omni MoE 专家 kernel、路由和 TP reduction 的数值路径需要独立检查；
   ASR dense 模型的 HF-order 对齐 profile 不能直接移植当作保证。
4. LoRA 在分片参数中的 BF16 执行、FP32 master 管理、聚合导出及 vLLM TP 切片需逐项验证；
   不把当前本地优化器 bridge 直接叠到 FSDP master 管理上。
5. 多卡完整候选与联合动作概率 gate、一次更新后的版本交接、保存恢复都需实测。

已有可复用证据：`omni-fsdp-layout-v1.md`（真实 meta/wrap）、
`omni-fsdp-loading-v1.md`（完整 CPU 权重及 FP32 buffer 精确加载）、
`omni-fsdp-initialization-plan-v1.md`（两卡初始化探针准备）、
`omni-fsdp-forward-storage-v1.md`（CPU FSDP/verl 合成集成）。
这些不构成 Omni 30B 多卡训练成功。

先完成模型版本选择，再派发对应多卡模型任务。当前推荐 Instruct 为主、ASR 0.6B 保留工程对照；
本次仅提出建议，未将历史 ASR 结果重新标记为 Omni，也未启动 30B 权重更新。


## 后续四卡基础验证

`fsdp-lora-npu4-v2` 已在物理4–7完成合成四卡 HCCL/FSDP/PEFT/verl 损失三步更新，
BF16 实际 LoRA 计算、FP32 主参数/Adam、固定拓扑恢复等价与跨 rank BF16 导出全部通过。
发现并修复了嵌套 FSDP 未转换 PEFT 输入的问题，需显式 `cast_forward_inputs=True`。
该结果不改变上述实际训练模型，也不证明 Omni 30B 或 vLLM TP4 准入。
详见 `fsdp-lora-npu4-v1.md`。


后续 `span-fsdp-audio-npu4-v2` 已将相同四卡路径推进到真实 ASR 0.6B：
八条真实语音的一次 FSDP/verl LoRA 更新通过，修复共享 embedding/head 的分片归属；
独立 HF 重载与八条训练输出误差0，48条TP1 vLLM验收最大误差1.1920929e-07。
这是 ASR 四卡训练/导出/TP1推理证据，不代表 Omni 30B、TP4 rollout 或任务收益。
详见 `span-fsdp-audio-npu4-v1.md`。

后续原生 vLLM TP4 48条诊断已完成：仅1条通过原定概率gate，最大logp误差0.2460，
8个Span首选候选变化，不能准入RL。这一原生profile尚未使用TP1的HF-order算子对齐，
不能把全部差异归因于TP归约。详见 `span-tp4-native-suite-v1.md`。
