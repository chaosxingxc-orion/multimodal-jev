# Stage1 初始化策略的运行契约准入

2026-11-14，修正请求契约后的RL更新态b01da14a已exit0并独立收集：256实际learning
音频×60候选HF/vLLM概率差0、阈值翻转0，128完整BF16张量与RL导出精确一致。
证据`context-engine-rl-updated-npu7-v1.json`，该策略ID
69bad9a9c414e714935461370ae5b5a1c9b589142c99f8f9f3eabf9641b4d763。
它证明协议执行，不作为正式Stage1的研究初始化。

`context_policy_admission.admit_initial_policy`绑定如下既有独立证据：
实际tokenizer扩展/冻结embedding、全日程DP4前缀、跨进程恢复重放、修正契约的256
HF/vLLM对照。逐项核对基座、head架构、LoRA语义、processor文件hash及冻结marker相同。
缺失人口、非零概率差、阈值翻转、恢复失败或marker不匹配均拒绝发布。

新的初始化只改变manifest里的runtime_profile、request_version与token状态，
全部原初始化execution.pt和processor文件逐字节复制，不使用RL诊断训练权重。
新契约为：

- runtime_profile: qwen3-asr-context-bf16-packed-row-v1；
- request_version: jev-context-exclusive_event-v1；
- token: 151706，冻结，admitted；151705仍为禁止输入的timestamp保留位；
- 输出: 全候选独立sigmoid，完整因果上下文浅层Transformer head。

固定CPU镜像容器6f79e1720c905f331bccfb9f45dc5a83cb6fd5b9a6483065d58d10dadb6cf70b
exit0，独立collector验证全部资产hash和原初始化字节完全不变，证据路径/hash正确。
新初始化policy ID：
816ed39c8e2684653467558cd212686902b93fd283c527f594b1d8978e11e2d1。
远程路径：
`/data/xingchao/multimodal-jev/simple-final-v1/context-initial-policy-admission-cpu-v1/run/policy`。
证据`context-initial-policy-admission-cpu-v1.json`。

21项准入/token/checkpoint相关本地tests过，其中8项检查证据绑定与初始化保留；
合成测试替代tokenizer扩展审核用于隔离测试，真实资产另有完整检查，不能混淆。
正式trainer现在同时要求admitted token、实际exclusive_event请求版本和命名运行profile。

这授予Stage1输入与执行协议资格，不保证每个未来权重的跨后端概率都相同。
每次RL行为仍须按当前policy实际校验。全量缓存、正式训练新增保存路径、连续joint-policy
激活、完整selection与迁移收益仍未完成。研究配置training_dispatch_allowed尚未打开。

## 完整评测入口

新增`evaluate_context_selection_vllm.py`：完整9226 selection、全部60候选、单固定policy，
输入线程只拿id/audio/hash，engine不接gold；逐条保存预测与实际输入receipt，全部完成
才生成完整predictions和指标。使用单请求vLLM已验证模式，后台CPU预取；缓存热路径
要求本地存储。记录评测入口会重构完整split并核对policy/request ID。
当前仅代码编译及已有14项指标测试，未派发完整selection，也不宣称此新入口真实NPU
验收完成。输入receipt自洽不足以独立证明processor正确，最终仍要独立人口/输入核验。
