# 连续两次 RL 更新验收

在物理 NPU4、固定 BF16 镜像和 Qwen3-ASR-0.6B 上完成两次连续更新，
从累计两步的 checkpoint 恢复，最终累计四步。容器及控制器均退出 0。
独立 verifier 检查了父子策略身份、数据游标、112 个实际 Adam 状态、
BF16 adapter 导出重载及每版完整 48 条音频的 HF/vLLM 概率。

| 新版本 | Adam step | batch cursor | 独立 HF 重载误差 | HF/vLLM 最大候选 logp 误差 |
| --- | ---: | --- | ---: | ---: |
| update 3 | 3 | 2 → 3 | 0 | 1.1920928955078125e-7 |
| update 4 | 4 | 3 → 4 | 0 | 1.1920928955078125e-7 |

每个更新使用一个三项标签均衡的录音对，32 个完整动作样本/请求，真实 verl
GRPO/PPO 联合动作损失，vLLM TP1 行为分布。Adam/master 为 FP32，执行与导出
均 BF16。两个版本的全部 48 条请求都通过既有概率阈值。

最终 adapter ID：`b86f4b9e6633231a50dd2d7852eb19bf401f6ac031d036e8bdb07fa5a567ee1e`。
完整结果见同名前缀 `-verified.json`，原始运行位于共享 experiments 同名目录。

这证明有限单卡连续恢复和版本验收链路；不证明四卡恢复、完整 Ray trainer、
不间断/恢复轨迹等价或学术性能提升。四卡 FSDP 的单次 fresh-optimizer 实验是
另一条优化器谱系，不能计入此处四次更新。当前 temporal 三项为平行分类，
不能以其更新成功宣称规则组合或决策 CoT 有效。
