# 当前 SLURP 策略的 NPU 采样概率诊断

固定于结果读取前。使用已审核、已训练的 temporal pilot，不读取 selection 或 TEST，不运行优化器。六个提示来自按录音路径字典序选择的两个完整三任务来源；选择不依赖预测正确性。每提示 n=2、temperature=1、top_p=1、top_k=-1、max_tokens=32、seed=17。max_num_seqs=1，最大上下文 2048，BF16，actor_features 输入，既有 vLLM 适配要求展开后的 prompt IDs 完全一致。

采样成功后独立进程对同一份 12 条响应进行三种审计：

1. 合并权重 HF 完整前向：观察执行后端差异。
2. 同合并权重 HF KV-cache 前向：观察预填充/解码形状影响。
3. 原始基座 + SFT LoRA 完整前向：贴近实际训练参数形式，同时暴露合并和低精度误差。

同一采样记录不替换旧 logprob，阈值固定为有效响应 token 的最大绝对 logprob 误差 0.02；完整保存每个 token 的两侧数值。单臂未达阈值仍完成另外两臂诊断。外层 `diagnostic_complete`/exit 0 只表示流程收尾，不表示三臂都通过。采样失败则终止，不用 HF 替代采样。

物理卡仅 4，固定镜像 `52ba94e1...`，8 CPU/64 GiB、无网络，整容器超时 900 秒。运行位置：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-temporal-rl-parity-v1`。

通过只说明这两条录音/六提示的单响应概率误差受控；仍需实际 padded microbatch、verl 概率/损失、原始基座加适配器更新和更新后新策略采样验证。不能把单响应结果当分布式 RL 已完成，也不能将工程阈值通过当算法收益。

若仍失败，依据三臂误差定位下一层。历史 cached replay 也失败过，因此不能先假定切换 KV-cache 就能解决；音频塔与语言模型中间状态审计需保存真实执行证据。历史音频分层探针失败在 callable 序列化、没有完整比较结果，不能把旧文件夹存在当作已诊断根因。
