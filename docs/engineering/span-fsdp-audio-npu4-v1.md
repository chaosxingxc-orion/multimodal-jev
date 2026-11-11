# 真实语音四卡 Span FSDP 探针

从已验收的 ASR 0.6B BF16 adapter 出发，四个 rank 各消费一对互补标签录音，
batch_index=2,3,4,5，共8条。使用清理审核的 temporal 学习发布；每 rank 的三项标签各一正一负。
原始音频直接输入模型，一次填充三项 Span，当前三项仍为平行分类。

## 固定契约

- 真实 torch-npu/HCCL 四卡 FSDP FULL_SHARD；q/v PEFT LoRA 主参数 FP32、计算 BF16，
  base BF16 冻结，buffer FP32，显式 cast_forward_inputs=True。
- 初始 adapter 来自 `slurp-span-verl-vllm-resume-v1/run/policy-1/adapter`，
  行为分布来自相同策略的 `slurp-bf16-resume-suite-v1/run/vllm/result.json`，固定全部文件哈希。
  先核对音频处理/特征/token/位置/候选/策略的完整 request identity，再复算全部动作概率 gate。
- 四 rank 全部完成前向准入后才 backward；每条录音32个完整动作，真实 verl GRPO/PPO loss。
  每 rank 两条请求的 mean loss 经 FSDP 平均，对应八条请求等权；每个完整组不会跨 rank 拆开。
- 此实验新建 FP32 AdamW，学习率1e-4，global norm clip1；不是单卡 optimizer 续训。
- 每 rank 保存本地 LoRA master 与优化器状态，仅作固定拓扑诊断；
  所有 rank 聚合完整 BF16 LoRA，核对112个键和形状与原 PEFT adapter 一致、跨 rank 哈希相同。
- 导出的 adapter 需要单独完整 HF 重载，再与训练中的八条结果比较；
  同时需对全部48条独立HF/vLLM验收。导出不等于已准入新版本。

本轮使用已验证 TP1 vLLM 分布验证 FSDP actor；尚未使用 TP4，也没有完整 Ray/verl worker 生命周期。
仅通过真实四卡数值检查才可称为该环节已验证；仍不报告学术收益或跨任务泛化。

## v1 失败及 v2 修订

v1 首次 embedding 前向失败，四 rank 均未更新。真实模型 tie_word_embeddings=True，
LM head 和 token embedding 是同一 Parameter；只包装 Linear 把共享参数归给 head 单元，
embedding 则读到了未 all-gather 的一维分片。v2 使用 `unshared_linear_modules` 排除任何
跨模块共享参数的 Linear，让共享权重留在共同根单元；不复制/解除权重绑定，不改变数学模型。
新增共享 embedding/head 与非共享 Linear 区分测试通过。真实 NPU 结果独立记录。


## v2 真实更新与独立验收通过

四卡容器退出0、无OOM。八条更新前全部通过概率gate，最大误差1.1920929e-07；
每rank记录112个实际BF16 LoRA A/B计算。真实verl更新global gradient norm=3.0731935501，
112×4=448个LoRA本地分片发生变化，冻结base未变化，四rank导出的adapter身份相同：
`6df66622568c868b2ab06efffdfd56fcfb3e512727a764953865ce0b68a1e986`。

`span-fsdp-audio-suite-v1`独立HF/vLLM验收也退出0/READY：
训练中的八条更新后概率与独立HF重载逐位一致（max error=0），
全部48条候选logp最大误差1.1920929e-07、完整动作概率比
[0.9999998807907176,1.0000001192092967]。本次使用的仍是TP1 vLLM。

独立脚本 `verify_span_fsdp_audio.py`核验source/config/pins、四rank实际optimizer分片、
FP32 master/Adam、真实导出身份、八条精确重载及48条完整概率gate。
证据 `span-fsdp-audio-npu4-v2-verified.json`。相关本地回归24项通过。
这是新的FSDP optimizer单步试验，不能接到此前单卡Adam时间线上声称累计第三步续训。
FSDP原生FP32梯度归约与单卡BF16执行梯度桥接并非已证明相同的更新轨迹，
仍需等全局batch的更新对照、真实四卡恢复、TP4 rollout和完整verl worker生命周期。
