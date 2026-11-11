# BF16 Span LoRA 恢复训练接口

当前入口是 `tools/jev/train_span_rl_step.py`，使用真实 verl GRPO/PPO 损失和
vLLM pooling LoRA 分布，模型直接读取音频及完整问题模板、一次填充所有 Span。
它是固定镜像内的单卡生命周期入口，尚未接入 Ray/FSDP 分布式 trainer。

## 恢复配置

沿用已验收运行的模型、学习发布、问题顺序、seed、音频处理和 LoRA 配置，新增或替换：

```json
{
  "initial_adapter": "/path/accepted-run/policy-1/adapter",
  "initial_optimizer_checkpoint": "/path/accepted-run/policy-1",
  "batch_index": 1,
  "runtime_profile": "qwen3-asr-bf16-hf-order-v1",
  "runtime_acceptance": "/path/accepted-version-suite/vllm/result.json"
}
```

`pins` 必须包含已验收的 suite、checkpoint 的 CHECKPOINT.json、optimizer.pt、adapter 文件、
本次 source snapshot 和迁移证据（如需）。模型与数据只读挂载，固定本地镜像与模型，不下载。
完整部署参数见 `slurp-span-verl-vllm-resume-v1-launch.json`。

`batch_index` 从零开始，必须等于前一 checkpoint 中的 `training_state.next_batch_index`。
它指向固定均衡 epoch 的下一对互补标签录音；轮末拒绝自动回绕。
第一步历史 checkpoint 没有游标，唯一兼容入口是 `legacy_selection_evidence`，必须绑定原 optimizer 哈希
及实际历史输入/均衡顺序，不依据 loss 猜测训练位置。后续 checkpoint 自带游标，无须再次迁移。

恢复加载完整 FP32 master、Adam 一二阶矩和 step，BF16 执行参数必须与 master 舍入结果一致。
Adam 参数组（含学习率）来自 checkpoint；当前不提供恢复时修改学习率的调度接口。
本地单步目录始终叫 `policy-1`，累计更新次数以 CHECKPOINT.json 的 `completed_updates` 为准。
每个运行目录必须全新；已 claim 或失败的目录不能重复启动。

## 验收和继续训练

1. 独立 HF 前向生成当前两条输入的参考概率。
2. vLLM 分别加载相同 adapter，逐候选及所有完整动作概率通过门槛。
3. actor 重新认证输入与行为概率，恢复优化器，通过 gate 后执行一次真实 verl 更新。
4. 原子发布 BF16 adapter 与完整优化器，独立 HF 重载的概率误差必须为零。
5. vLLM 加载新版本并再次验证这两条输入，全部成功才写 READY。
6. 用 `check_span_policy_version.py` 生成新版本独立 HF 参考并完成全部 48 条顺序验收，
   然后才将其作为下一次训练的初始 adapter 和 runtime_acceptance。

门槛为候选最大 logprob 差 ≤0.02，全部完整动作概率比在 [0.98,1.02]。
行为概率不得用 actor 重算值覆盖。READY 与 terminal 需同时检查，Docker 退出码另行保存。
观察采用单个 `docker wait`，或至少十分钟间隔检查；等待工具返回超时不意味着运行失败。

当前手动独立运行每个生命周期，保留两条前后 gate 和 48 条验收。
后续连续运行器需要接管版本交接、终态恢复和验收依赖，不能仅循环调用 optimizer.step。
真实 NPU 恢复成功仅证明状态加载及下一步生命周期；不等于与不中断训练逐位等价。
当前每步固定采样 seed，未承诺跨步骤 RNG 序列恢复；随机音频增强关闭，保留 raw。

## 结果解释

48 条录音来自审核的学习诊断发布，均衡训练使用其中 40 条、八种联合标签各五条，
两条缺少完整监督的录音不会进入互补训练对。审核基于参考文本的语义复核，未完成人工听审。
三个时间标签是平行分类，奖励为正确率均值，不构造依赖式 CoT 过程奖励。
恢复或一致性验收均不作为 held-out 效果、算法收益或跨任务泛化证据。


## 已验证结果

`slurp-span-verl-vllm-resume-v1` 终态 0/READY，累计真实 verl 更新 1→2，
112 个参数完整 Adam 状态均恢复并推进至 step 2，独立 HF 重载误差 0。
新版本 `slurp-bf16-resume-suite-v1` 全部 48 条通过，最大候选 logp 差 1.1920929e-07，
独立枚举 384 种完整动作的概率比为 [0.9999998808,1.0000001788]。
本轮相关本地测试 26 项通过；固定镜像 NPU4 实验验证了真实后端路径。
凭证为同目录 `slurp-span-verl-vllm-resume-v1-verified.json` 与
`slurp-bf16-resume-suite-v1-verified.json`。
