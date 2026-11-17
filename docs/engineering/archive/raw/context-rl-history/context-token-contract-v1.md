# Jev 决策 token 与请求契约验收

2026-11-14。本协议限定端到端音频决策 pooling：一次消费完整输入，输出全部候选的
独立sigmoid概率；不进入生成或强制对齐接口。

## 已验证资产

固定镜像的Qwen3-ASR配置声明timestamp_token_id=151705，processor的
decode_forced_alignment按此ID筛选时间戳位置。普通音频骨干前向不读取该配置项。
源码位置/hash见`.local/decision-objective-research/context-timestamp-source-review-v1.json`。

`context_token_contract.validate_processor_extension`实际验证原模型与当前policy processor：

- tokenizer的model、normalizer和其他非added_tokens字段完全一致；
- 原有added_tokens全部属性逐项保留，无重复ID；
- 仅新增151705保留占位`<|jev_probe_reserved_id_151705|>`与151706 `<|jev_decision|>`；
- 两者均special且normalized=false，其余AddedToken属性固定；
- 151706不与model config任何token ID保留项冲突，且在原embedding容量内。

决策行独立从原模型safetensors切片复算：原token A/B的ID32/33，FP32均值再转BF16，
与已保存policy中的decision_token_row逐元素精确一致。原151705行另记录hash，
初始化不写这行。证据`context-token-assets-v1.json`；没有改原模型/processor资产。

## 输入隔离

保留位不得作为语音决策问题、历史或其他上下文的一部分。新增三处检查：
ContextDecisionRequest拒绝字面注入；processor编译后拒绝实际保留ID；HF模型及
固定vLLM owner在消费预编码输入时再次拒绝151705。由此缓存输入不能绕过编译检查。
该约束保持已有正常请求不变，不改变骨干/head方程。

19项本地token结构、请求编译、缓存身份测试通过，含改normalizer、词表、重复ID、
特殊token属性、config冲突、embedding容量以及注入保留位的拒绝测试。
本地没有vLLM安装，尚不能把这些测试称为新增owner分支的真实vLLM执行验收。

## 正式契约仍需完成的步骤

现有初始policy仍是test_only。其request_version是通用
`jev-audio-context-applicability-v1`，实际SLURP请求是
`jev-context-exclusive_event-v1`；初始policy的runtime_profile也还是CPU诊断名称。
两者必须在新的不可变policy中绑定实际任务版本和命名BF16运行profile；不能原位改
历史manifest或复用旧policy ID。训练采用的基座/tokenizer/初始化张量保持不变，
新身份仍需导出/重载、实际请求和跨后端验证。

本文与token-assets证据只完成资产和边界审查，不自行将test_only改为admitted。
全人口输入ledger、DP4更新恢复和256同策略HF/vLLM证据已存在，需在新的正式契约
中明确绑定；缓存只绑定输入资产，policy元数据变化本身不应使缓存失效。
