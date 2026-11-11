# BF16 基础投影打包误差定位

目的：在前层输入相同处区分基础投影与 LoRA 路径误差。固定同一 BF16 checkpoint，使用 rotary-v1 中已保存的第一层归一化激活，仅用于算子诊断；保存激活不进入任何正式推理、奖励或训练。

568/600 个位置的 HF/vLLM 第一层归一化输入逐位相等。在这些位置，Q/K/V 输出分别有 91/25/27 个元素不同；K 没有 LoRA，不能把全部差异归因 adapter。

`tools/jev/audit_span_projection.py` 在物理 NPU4 同一进程内读取真实输入和原始 safetensors，比较 separate 2D、separate 3D、packed 2D 基础 GEMM，统一使用 BF16 PEFT 顺序计算 q/v LoRA；交叉 ND/NZ 权重布局与内部格式开关，共 12 种组合，每种重复两次。

v1 因 torch-npu 的 allow_internal_format 仅提供 setter，读取属性失败，在运算对照前终止。v2 使用固定运行时 option getter 读取配置，终态0。结果：所有 separate 计算逐位复现完整 HF 投影；所有 packed 计算在 568 个相同输入位置逐位复现 vLLM 投影；所有重复计算误差0。内部格式开关、权重 ND/NZ 与 2D/3D 输入均不改变结果。这在本样本第一层确认了基础 QKV 打包 GEMM 舍入差异；未证明所有层、音频塔或 LoRA 内部计算完全一致。

凭证：`docs/engineering/slurp-bf16-projection-audit-v2-verified.json`。原始对照结果与 source/config pins 位于 `slurp-bf16-projection-audit-v2`，没有新增训练或模型下载。

后续单因素整模型探针 `slurp-vllm-spans-bf16-qkv-v1`：以 norm-v1 为对照，仅将语言层 QKV 基础投影拆成三个 F.linear，保留 vLLM 原生 LoRA add、attention、rotary、音频塔。限制 TP1、BF16、三片 QKV wrapper。它是诊断，不进入正式训练配置；需核验首层同输入位置以及最终完整动作 gate。

QKV整模型探针终态0：568个相同输入位置Q/K/V输出全部逐位一致，原生LoRA路径保留。最终max logp差0.140574，完整动作比率[0.839265,1.165205]，Span0 argmax变化，未过gate。局部修复不能冒充端到端验收。

后续mlp-v1仅在QKV+norm设置上拆分gate/up投影和SiLU→BF16→乘法，保留原生down projection、attention、rotary和音频塔；新增第一层MLP输出hook。该干预同时覆盖MLP投影打包和激活融合，不能单独归因其中一个因素，仍为诊断。
