# 四卡 vLLM Span 原生推理诊断

## 实际结果

物理 NPU 4–7、固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
Qwen3-ASR-0.6B、BF16 基座与未合并 q/v LoRA。使用四卡 FSDP 导出的同一 adapter：
`6df66622568c868b2ab06efffdfd56fcfb3e512727a764953865ce0b68a1e986`。

48 条真实音频均完成推理，容器退出0。独立复算结果：

| 检查 | 结果 |
| --- | --- |
| 逐请求通过原定候选/联合概率阈值 | 1 / 48 |
| 最大候选 log-prob 误差 | 0.24601364135742188 |
| actor / rollout 完整动作概率比范围 | 0.6835159660716852–1.4110209244964416 |
| 首选候选发生变化的 Span 数 | 8 / 144 |
| RL 准入 | 否 |

这验证了原生四卡 pooling、真实音频、native LoRA 和词表聚合可运行，
同时证伪了“同为 BF16 就能直接用于当前 actor”的假设。
不是精度提升或 held-out 评测；不把退出0当成一致性通过。

## 入口与验收边界

独立架构 `Qwen3ASRForSpanDecisionsTP4Diagnostic` 只接受 TP4/PP1、完整无缓存 prefill。
保留 `Qwen3ASRForSpanDecisions` 的 TP1 限制及已通过的训练准入 profile。
四卡词表投影显式使用真实 logits all-gather，使每个 pooling worker 可读取完整词表；
不复制 HF 激活、不合并 adapter、不替换为参考输出。

诊断 profile `qwen3-asr-bf16-native-tp4-diagnostic-v1` 只关闭 internal format，
其余保持原生 vLLM/Ascend 算子。未安装 TP1 的 HF-order norm/QKV/MLP/RoPE/attention。
因此与既有 TP1 已对齐结果的差异同时包含算子路径与 TP 分片归约，不能全部归因于 TP。

`summarize_span_tp4_diagnostic.py` 独立验证配置和文件 pins、完整48条请求身份、
原 adapter/base 身份、四 worker 的 BF16 head/LoRA/adapter状态，重新计算全部候选与
完整动作概率 gate。不生成 READY，不放宽训练阈值。

本轮拓扑准入与原有 profile 测试12项通过；策略版本、概率、请求身份回归7项通过。
源快照在实验根目录，发布后未修改；复核脚本位于工程 tools/jev。
证据见 `span-tp4-native-suite-v1-launch.json`、`span-tp4-native-suite-v1-verified.json`。

## 下一步

以独立实验 profile 逐项验证 HF-order 的局部分片运算，保留原生 LoRA 切片和通信，
测量残余误差再定位 row-parallel 投影与归约。保持已验收的四卡 actor + TP1 rollout 可用。
真实四卡 optimizer 恢复、等全局 batch 更新对照、完整 verl worker 生命周期、
学术基线和泛化实验仍未全部完成。当前结论不适用于 Omni 30B。
