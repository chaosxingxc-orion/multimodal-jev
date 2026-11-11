# BF16 Span runtime profile v1

`qwen3-asr-bf16-hf-order-v1` 对齐本地 Qwen3-ASR-0.6B-hf 的 HF/PEFT actor 与 vLLM pooling rollout。它继续使用真实音频、一次 Span 前向、原生 vLLM LoRA；不合并 adapter、不使用历史激活替代实时计算。

## 适用范围

固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，torch-npu 2.10、transformers 5.14.1、vLLM 0.30、verl 0.8dev、PEFT 0.21.2。

当前仅 TP1、单个完整无缓存请求、eager、无量化，物理 NPU 4。不得直接用于 packed batch、chunked prefill、prefix cache、decode 或 TP>1。LoRA 仅语言层 q/v、rank4、alpha8、dropout0；基座/adapter/输出投影 BF16，归一化概率与 optimizer master/state FP32。吞吐和峰值内存尚未单独验收；显式 eager attention 的开销需要后续测量。

## 数值路径

| 路径 | 运行行为 |
| --- | --- |
| actor | HF eager；显式关闭 NPU internal format；PEFT 禁止自动升精度 |
| 音频 encoder | vLLM 原生卷积/QKV/投影；关闭 internal format；按音频窗口采用 HF eager attention |
| 语言 norm | HF RMSNorm 顺序及 BF16 residual addition |
| 语言 QKV | 分别执行三个基础 GEMM，保留原生 vLLM LoRA add |
| 语言 MLP | gate/up 分离 GEMM，BF16 SiLU 后乘法，原生 down projection |
| rotary/attention | 默认 full-head NeoX 的 HF-order rotary；单序列 causal eager attention |
| Span readout | 相同位置、词表候选、BF16 LM head、FP32 candidate log-softmax |

## 证据与验收

单录音 `slurp-vllm-spans-bf16-language-rotary-v2`：100个非stride trace点逐位相等，最大候选logp差5.9604645e-08；8种完整动作actor/behavior概率比均约1.0000001192。与原生vLLM的max误差0.140574相比，定位到了实际运算路径差异；单录音证据不代替多请求及更新后验收。

48条现有SLURP审核学习录音的顺序验收入口为 `tools/jev/check_span_runtime_suite.py`：使用独立保存的HF概率、严格核验音频/预处理/展开token/位置，单engine运行全部录音，逐条保留全部候选及请求身份。它是工程诊断，不提供held-out收益结论。

`tools/jev/train_span_rl_step.py` 的 `runtime_profile` 参数必须同时提供 `runtime_acceptance` 并将其加入 `pins`。入口核验验收文件hash、基座与初始adapter身份、48条不同请求、八种完整监督组合，并从保存概率重新计算gate。之后每个更新前请求仍单独过gate；一次真实verl完整动作PPO/GRPO更新后，导出不可变adapter/optimizer快照，独立HF重载和新vLLM版本检查通过才写READY。任意失败都不会伪造READY；更新后失败则保留新版本供诊断，不继续采样。

任意新adapter的48条独立验收入口是 `tools/jev/check_span_policy_version.py`，顺序调用 `build_span_hf_reference.py` 与 `check_span_runtime_suite.py`，两后端不同时驻留NPU。配置包含 `model`、`adapter`、`identity`（已锁定的输入manifest）、`runtime_profile` 和 `pins`；source/模型/输入来自固定镜像与只读挂载，不允许下载。输出目录必须全新。

```bash
PYTHONPATH=src python tools/jev/check_span_policy_version.py --config /path/config.json --output /path/new-run
```

命令必须在指定NPU镜像和物理4卡隔离环境内执行。`READY.json`记录新adapter身份和全量结果hash；失败仅保存终态，不进入后续政策版本。

固定门槛：每个候选max |Δlogp|≤0.02，所有完整动作的actor/behavior概率比∈[0.98,1.02]。不以argmax相同、单一准确率或profile名称代替检查。保存重载目标为HF候选概率完全一致。

## 尚未完成

48条顺序验收已全部通过，max logp差1.1920929e-07，完整动作概率比范围[0.9999999851,1.0000001788]；所有384种动作比率已独立枚举复核。step-v3完成一次真实verl更新，但更新后重载失败，新版本未准入；step-v4正在重验。批量/多卡、完整Ray/verl trainer与跨任务效果仍待验证。当前时间类三问属于并列分类，平均正确性奖励不被称为依赖式CoT过程奖励。学术结论仍需要更大的清洗发布、独立划分及等预算SFT/RL对照。

## verl 导入副作用隔离

固定镜像的 `/opt/verl/verl/__init__.py` 在检测到NPU时导入 `models.transformers.npu_patch`，后者全局替换HF Qwen3 RMSNorm.forward、MLP.forward与apply_rotary_pos_emb。若在首次loss计算时才导入，更新前actor与更新后的前向会使用不同实现；这会破坏保存重载检查，甚至改变后续rollout政策。

Span更新进程在构建模型前调用 `initialize_verl_preserving_qwen3()`：保留三个原始HF函数对象，导入真实verl core，再恢复这些对象。遇到已被verl替换的进程或未知来源的替换即拒绝；不修改共享镜像文件。loss继续调用固定verl实现。记录 `verl-runtime-initialization.json`，并依靠更新前gate、独立重载与新版本gate验收实际效果。step-v3的失败checkpoint不用于后续训练。
