# 原生意图有限候选 RL 四卡首步准备

2026-11-12 UTC。`train_intent_finite_npu.py`将已审核D/Q/P数据与精确有限候选信用接到原生verl actor。预定首轮为D、seed317、全局8个来源、每rank2请求、microbatch1、一次optimizer更新、lr1e-4。Q/P作为后续匹配对照，不在D尚未验收前并行派发。

模型为本地Qwen3-ASR-0.6B，初始adapter沿用已固定的temporal初始化并明确保留来源；LoRA rank4/alpha8，BF16执行/导出、FP32 master/Adam，FSDP FULL_SHARD DP4。rollout来自TP1独立HF/vLLM验收中同policy、同请求的完整候选概率。没有转写分支、文本解码或答案回填。

rank0校验数据与行为策略、生成全局候选积分权重后，将真实DataProto按完整行分给四卡。使用 `span_finite_actor` 已通过合成CPU四rank验证的原生compute/update。模型前向逐次检查全候选概率，优化器只在完整梯度累积后更新一次。每rank保存实际master/Adam、执行dtype和更新前后概率；导出要求四rank逐tensor一致。

首步入口显式拒绝step>0和任何续训参数，避免把fresh Adam误作连续RL。后续仍需接入经过验收的精确FSDP恢复、更新后的新行为策略及固定评测曲线，不能把本首步声称为完整trainer。

`verify_intent_finite_npu.py`已实现独立审计：重新验证源/监督/样本日程，检查真实FP32分片、Adam步数及二阶矩，核对全局有效权重，要求训练8输入的FSDP输出与独立HF重载完全一致，并复算全部178个D/Q请求的HF/vLLM候选一致性。该新训练及验收入口尚未在NPU执行，编译/配置测试不是后端成功证明。

冻结产物位于同名prepared JSON。只保存config-template；等待现有原生actor验收、178输入D/Q/P gate完成后才能绑定最终config及行为结果哈希，检查容器全部pin可见和卡占用再派发。当前没有实际训练容器。

`intent_policy_metrics.py`与 `score_intent_learning_policy.py`分别计算Final准确率、观测类macro recall/F1、组件准确率、兼容率、正确类别期望奖励与加权期望奖励。缺少gold覆盖的4类列出，不把56类macro冒充60类全覆盖；学习集分数不作为独立泛化证据。
