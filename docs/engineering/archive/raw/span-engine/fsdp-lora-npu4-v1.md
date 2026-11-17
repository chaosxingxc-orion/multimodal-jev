# 四卡 NPU FSDP / LoRA 工程探针

目标：在固定镜像、授权物理 NPU4–7 上验证 HCCL、真实 PEFT LoRA、FSDP FULL_SHARD、
真实 verl GRPO/PPO 损失及固定拓扑 checkpoint 恢复。使用合成小线性模型，未加载语音或 Omni 权重。
入口 `tools/jev/probe_fsdp_lora_npu.py`，torchrun 四进程；固定执行上限 600 秒，独立容器和运行目录。

## 协议

- 各 rank 不同输入及采样种子，32 个完整三 Span 动作，三个平行分类的平均正确率奖励。
- BF16 冻结 base，FP32 持久 LoRA 主参数，FSDP MixedPrecision 将 A/B 前向转换为 BF16。
  用实际 Linear 前向 hook 检查输入/权重 dtype，FP32 reduce 和 Adam 状态。
- 分别包装 frozen BF16 base Linear 和 trainable FP32 A/B Linear，避免混合 dtype 的 flat parameter。
  `use_orig_params=True`，FULL_SHARD，`sync_module_states=True`。
- 先执行 rank-specific HCCL all_reduce，要求求和结果为 10。
- 更新两步后，各 rank 保存本地参数分片和 Adam 状态，原模型不中断再执行第三步。
- 新建第二个真实 FSDP 模型，加载本 rank checkpoint，要求恢复前向与第二步完全相同；
  用相同第三步采样种子更新，要求输出、FP32 参数和全部 Adam 状态逐位等于不中断路径。
- 冻结基座必须未改动，LoRA 必须确有变化；收集完整 BF16 LoRA 导出，所有 rank 哈希相同。

## 边界

此处测试的是 FSDP 原生 FP32 主参数→BF16 计算路径，未使用单卡 BF16MasterAdamW bridge。
本地分片 checkpoint 仅支持相同 world size、rank、参数顺序及 wrap 拓扑；不提供跨拓扑恢复保证。
导出是探针 tensor 字典，尚非可供 vLLM 消费的完整 PEFT 发布。
尚未包含单卡与多卡梯度等价性、真实音频/Omni MoE、vLLM TP4、完整 verl worker/Ray trainer
或任务质量验收。不能因合成探针通过而宣称 30B 多卡训练可用。

运行状态与结果另存 launch/verified/failed 凭证，不根据已创建容器推定成功。


## v1 失败与 v2 修订

v1 四个 rank 的 HCCL all_reduce 全部通过，但首次 PEFT LoRA 前向的 BF16 检查失败，
0 次参数更新。初始 MixedPrecision 只有根输入转换，嵌套 A/B FSDP 未显式设置
`cast_forward_inputs=True`；PEFT 可在进入 A 前按尚未 all-gather 的 FP32 主参数 dtype 转换输入。
v2 为每个 FSDP 单元启用输入转换，并在失败时记录实际输入/权重 dtype，保持 FP32 主参数与 BF16 计算契约。
不能仅凭该修改推定修复，需完整真实四卡更新与恢复验收。


## v2 实测结果

固定镜像容器 `ab37b0406fe010049fd62f32029fa0e876b8edc28763cac5856163cabd9fd6d6`
退出 0、无 OOM，四个 rank 均完成全部检查。启用嵌套输入转换后，所有 A/B 的实际计算权重和输入
均为 BF16；本地持续 LoRA 参数及 Adam moments 为 FP32。
每 rank 持有 base 24 元素、A 16 元素、B 6 元素，是完整 96/64/24 元素的四分之一。
三个真实 verl 更新后，两组 LoRA 的全部八个分片均有变化，冻结 base 未变化。
第二步恢复后第三步与不中断路径的输出、master 和 Adam 逐位相同。
两个 BF16 导出张量在全部 rank 上哈希完全相同。

每 rank 实测峰值 allocated 为 33,576,960 字节，探针内部耗时约 3.2 秒；
这不包含全部进程启动时间，不代表 30B 模型显存或吞吐。
`verify_fsdp_lora_probe.py` 独立核对 source snapshot、全部 rank 终态、
实际 checkpoint 张量/Adam 状态与 BF16 导出；结果 `fsdp-lora-npu4-v2-verified.json`。

此结果表明四卡 HCCL/FSDP/PEFT BF16 计算及固定拓扑恢复路径可用，
但尚未证明单卡全局 batch 与四卡梯度/更新等价，也没有实例化完整 verl worker。
下一步需将 typed Span actor 接入分布式训练并验证真实音频及 vLLM TP4；
若选择 Omni 30B，还需原始 Thinker/MoE 权重和实际 wrap/内存测试。
