# BF16 基础投影打包误差定位

目的：在前层输入相同处区分基础投影与 LoRA 路径误差。固定同一 BF16 checkpoint，使用 rotary-v1 中已保存的第一层归一化激活，仅用于算子诊断；保存激活不进入任何正式推理、奖励或训练。

568/600 个位置的 HF/vLLM 第一层归一化输入逐位相等。在这些位置，Q/K/V 输出分别有 91/25/27 个元素不同；K 没有 LoRA，不能把全部差异归因 adapter。

`tools/jev/audit_span_projection.py` 在物理 NPU4 同一进程内读取真实输入和原始 safetensors，比较 separate 2D、separate 3D、packed 2D 基础 GEMM，统一使用 BF16 PEFT 顺序计算 q/v LoRA；交叉 ND/NZ 权重布局与内部格式开关，共 12 种组合，每种重复两次。

v1 因 torch-npu 的 allow_internal_format 仅提供 setter，读取属性失败，在运算对照前终止。v2 使用固定运行时 option getter 读取配置，终态0。结果：所有 separate 计算逐位复现完整 HF 投影；所有 packed 计算在 568 个相同输入位置逐位复现 vLLM 投影；所有重复计算误差0。内部格式开关、权重 ND/NZ 与 2D/3D 输入均不改变结果。这在本样本第一层确认了基础 QKV 打包 GEMM 舍入差异；未证明所有层、音频塔或 LoRA 内部计算完全一致。

凭证：`docs/engineering/slurp-bf16-projection-audit-v2-verified.json`。原始对照结果与 source/config pins 位于 `slurp-bf16-projection-audit-v2`，没有新增训练或模型下载。

后续单因素整模型探针 `slurp-vllm-spans-bf16-qkv-v1`：以 norm-v1 为对照，仅将语言层 QKV 基础投影拆成三个 F.linear，保留 vLLM 原生 LoRA add、attention、rotary、音频塔。限制 TP1、BF16、三片 QKV wrapper。它是诊断，不进入正式训练配置；需核验首层同输入位置以及最终完整动作 gate。

QKV整模型探针终态0：568个相同输入位置Q/K/V输出全部逐位一致，原生LoRA路径保留。最终max logp差0.140574，完整动作比率[0.839265,1.165205]，Span0 argmax变化，未过gate。局部修复不能冒充端到端验收。

后续mlp-v1仅在QKV+norm设置上拆分gate/up投影和SiLU→BF16→乘法，保留原生down projection、attention、rotary和音频塔；新增第一层MLP输出hook。该干预同时覆盖MLP投影打包和激活融合，不能单独归因其中一个因素，仍为诊断。

MLP探针终态0：max logp差0.109080，完整动作比率[0.866957,1.130725]，本样本argmax不再变化但gate仍失败。相对QKV-only有所变化，不能据一个样本判定泛化或已满足训推一致。新增MLP hook未独立做无hook对照，因此仅记录本配置的候选概率，不把变化全部归因某一个算子。

## 音频首卷积复算

projection-audit-v3在前述GEMM对照之后补充首卷积同进程对照，使用两个trace中逐位一致的真实input/weight/bias。显式交叉输入stride（HF/vLLM）、ALLOW_INTERNAL_FORMAT（disable/enable）以及偏置融合/分离。全部重复计算误差0。融合偏置时，disable逐位复现HF、enable逐位复现vLLM，与stride无关；两者只差8个元素，max0.0009765625。分离偏置明显不同，不作为修复。源码核查发现vllm-ascend model_runner_v1在模块导入时显式开启allow_internal_format，而独立HF进程默认disable；这是有因果复算支持的具体运行时差异。

internal-format-v1在MLP+QKV+norm配置上，仅在worker加载完成后显式关闭内部格式，保留原生LoRA、attention、rotary；将验证首卷积、完整音频塔和完整动作gate。不能把首卷积对齐等同整模型已一致。

internal-format-v1终态0：三层conv、conv_out、首层音频input_norm以及Q/K/V全部逐位一致。首个剩余音频差异出现在首层attention输出（max0.015625、RMS0.00074114）。audio_projected RMS0.000218393；最终max logp差0.101054、联合比率[0.873943,1.122165]，gate仍失败。已锁定并消除此前最早音频前端差异，不能宣称全音频塔或整模型完成对齐。

audio-attention-v1以此配置为基础，仅将音频非因果attention改为HF eager顺序、按cu_seqlens逐窗计算；保留原生audio QKV和输出投影。CPU独立逐head参考、跨窗口隔离及非因果语义测试通过，NPU整体结果待验。

音频 attention 探针终态0：全部37个非stride音频trace点（18层音频encoder及projector）逐位一致；600个语言输入位置及首层Q/K/V亦全部一致。首个剩余trace差异在语言首层post_attention_norm。最终max logp差0.101054、完整动作actor/behavior比率[0.850760,1.162852]，仍未过gate、0次RL更新；已独立枚举8种动作复核。下一步在同配置上逐项加入语言eager attention与HF-order rotary。凭证见 `docs/engineering/slurp-bf16-audio-attention-v1-verified.json`。

language-attention-v2在全部已验证局部修复上，新增语言HF-order eager attention，保留原生rotary。终态0：max logp差0.0431666、完整动作比率[0.957752,1.035577]，argmax无变化但gate仍失败。音频projector保持逐位一致，语言首层post_attention_norm仍有差异。后续language-rotary-v2仅增加HF-order位置编码；不得因本样本误差缩小提前准入RL。

language-rotary-v2终态0且首次通过完整动作gate：全部100个非stride trace点逐位相等，含最终语言norm；max logp差5.9604645e-08、actor/behavior完整动作比率[1.0000001192,1.0000001192]。这是单条600-token真实语音、非零BF16 LoRA、无生成的证据，不代表批量/多卡/新adapter已验收。后续使用命名profile `qwen3-asr-bf16-hf-order-v1` 在同一engine顺序执行48条学习录音。
