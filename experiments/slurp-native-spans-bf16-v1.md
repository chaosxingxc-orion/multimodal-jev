# BF16 LoRA 执行与训推一致性

用户指定 BF16 后停止旧 FP32 LoRA v5 探针（docker wait=143，未完成概率评测），保留其不可变快照；v4 则在 worker callable 序列化阶段失败，不能用作 head_dtype 修正的概率证据。审计接口改为具名 worker extension RPC，不开启 pickle fallback。

新实验 `slurp-native-spans-bf16-v1`：固定镜像、本地 Qwen3-ASR-0.6B、物理 NPU4；同一 reviewed SLURP 发布、40 条均衡录音、20 次更新、rank4/alpha8、q/v、dropout0、lr1e-4、candidate CE、raw 音频。完整 48 条/142 标签只作学习诊断，包含训练样本，不能作泛化结果。

精度政策：基座、LoRA A/B 执行参数、LM head 投影均 BF16；PEFT autocast_adapter_dtype=False 并校验/转换 adapter 参数。候选 log-softmax 与损失 FP32。AdamW 在独立 FP32 master 上累积更新，每步将 master 舍入到 BF16 执行参数；导出的 adapter 正是该 BF16 副本。当前 master bridge 仅为本地单卡 pilot，不是已经完成的分布式 verl 集成，不提供续训恢复。正式续训必须同时保存 master 和 optimizer state。

vLLM 使用同一未合并 BF16 adapter，显式 lora_dtype=bfloat16、head_dtype=model、语言 qkv 精确目标路径；拒绝 checkpoint dtype 不匹配的静默转换。初期 TP1/eager、禁用 prefix cache/chunked prefill。worker 审计实际投影 dtype、包装模块与 adapter ID。

验收顺序：BF16 SFT 有限 loss/非零梯度和 precision.json → 独立重新加载后全部48条候选概率复现 → 同一 BF16 checkpoint 的 HF/vLLM 全候选比较 → 多录音/长度/组合及批量变化 → 同步版本更新验证 → verl actor 更新。保留 max |Δlogp|≤0.02 与完整动作比率 [0.98,1.02] 的初始 gate，不因 BF16 放宽门槛。单样本通过不能代表全局通过。

奖励与精度解耦：本次不改变奖励，也不执行 RL。并列 temporal 标签不包装为过程链；未来独立 Final 使用 Final-only、可靠辅助 CE/奖励对照，缺标严格 mask。BF16 一致性只能由实际候选概率验证，不能从 dtype 配置或任务准确率推出。

主机测试：61 项相关测试通过，包括小于 BF16 分辨率的更新在 FP32 master 中保留、导出执行副本舍入一致、非有限梯度拒绝。项目 .venv 缺 torch；另用具备 torch 的 host Python 执行上述验证。NPU 实验及重载结果见下文，跨后端验证另记，不能预先声称通过。

训练终态0：20次更新完成；precision.json确认112个BF16执行参数、FP32 master/AdamW状态。独立读取safetensors文件头确认112个保存tensor均为BF16。均衡schedule及逐样本混淆矩阵独立重算通过。学习诊断before70/142 → after93/142，静音80/142；三问全对分别5/46、10/46、7/46。after balanced accuracy：date0.541667、time0.757576、timeofday0.666667。它不能证明BF16优于旧FP32运行（旧candidate CE为92/142），也不代表泛化或RL收益。保存重载验证已另行启动。

独立重载终态0：全部48条音频、全部候选logprob最大差0（门槛1e-5）；重算学习/静音指标一致。仅证明HF保存重载一致。`slurp-vllm-spans-bf16-v1`已启动同checkpoint跨后端单条探针，worker实际LoRA buffer dtype与配置逐层核验。

跨后端探针终态0但概率gate失败：最大候选logprob差0.1405742168，完整动作比率[0.8114645496,1.1992668623]，第0个Span的argmax翻转。独立重算gate一致。worker确证28个语言qkv wrapper的实际A/B缓冲均BF16、head_projection_dtype=BF16、adapter ID从空变为[1]；base→adapter最大logit变化2.5625，返回base差0。未执行RL更新。不能因与旧FP32探针的误差不同就宣称BF16导致退化：两次使用的训练adapter不同。

当前结论：BF16训练/导出/重载与原生LoRA serving链路可运行，HF/vLLM数值一致性尚未达标。下一步在同一BF16 checkpoint上分层对比音频投影、语言各层、LoRA分支和head输入；先寻找最早偏离位置，再研究统一计算实现。每次变更复验完整请求/输出概率，不通过奖励或阈值改动掩盖误差。通过单条后再扩展长度、标签组合、预处理、batch以及更新同步覆盖；这些工作尚未完成。
