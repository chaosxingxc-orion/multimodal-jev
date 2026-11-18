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

源码检查已重新从旧固定镜像的已停止容器提取 vLLM `qwen3_omni_moe_thinker.py`，SHA256 为 `0f44c586731ec7a589af4e58b1998f49ef17abf2907e70a66c0b338913e5fb81`，与本地历史副本一致。Qwen3-ASR 复用其中的音频编码器。HF 按固定窗长补齐卷积输入，vLLM 按当前 chunk 的最长实际长度补齐；短于一窗的音频可能走不同边界计算。当前六提示是否触发此差异尚未实测，不能将源码差别直接称为本次误差根因。

测试：项目 `.venv` 的相关回归 38 passed、1 skipped（缺 torch）；用宿主已安装 torch 单独执行 cached-response 位置与缓存用例，1 passed。未将跳过的用例算作通过，新增 adapter 审计分支仍须以真实运行结果验证。

## 已完成结果

既有 `docker wait` 返回 0，外层终态为 `diagnostic_complete`。采样得到 12 条响应、72 个响应 token；三个概率审计臂全部失败，阈值保持 0.02：

| 审计臂 | 最大绝对 logprob 误差 |
| --- | ---: |
| merged_full | 0.13464361429214478 |
| merged_cached | 0.13292181491851807 |
| adapter_full | 0.1329839825630188 |

独立读取各臂全部 `response-*.json`，逐条核对 example_id、replicate、token_ids、保存的 vLLM 概率与采样原记录一致；检查概率有限及长度匹配，重新计算 72 个 token 的误差，均与结果文件一致。采样文件 SHA256 为 `05423064aecaee8d4d264d0ab564a0fafd25b88a09dc6c5de744a584da0f503f`。

运行快照的 adapter_full `scope` 沿用旧的 same-merged 文案；实际 `base`、`adapter` 字段和命令为原基座加 LoRA，该文案已在后续源码修正，不改写历史结果。此次没有优化器更新。缓存前向与原适配器前向均未使误差通过门槛，尚不能定位为音频塔或某一算子的根因；当前不准入 RL 更新。工程概率问题与数据/语义质量问题分别处理，修复任一项不自动证明另一项通过。

已另行核验容器退出码、无 OOM、固定镜像和只挂载物理 4 号卡，并重新认证配置 pins；逐 token 重算与文件哈希见[独立核验凭证](../../../../docs/engineering/slurp-temporal-rl-parity-v1-verified.json)。后续[音频中间状态探针](slurp-temporal-audio-stages-v1.md)已派发，尚不据此认定根因。
