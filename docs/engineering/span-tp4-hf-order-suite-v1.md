# TP4 局部 HF 运算顺序对齐诊断

## 结论

48 条真实音频全部完成，四卡容器退出0。候选概率一致性仍未通过，不能准入 RL。
这轮不是算法收益实验，也不是 Omni 30B 验证。

| 同一 adapter / 同一48条输入 | 原生 TP4 | 局部 HF-order TP4 |
| --- | ---: | ---: |
| 通过候选与联合概率阈值的请求 | 1 / 48 | 3 / 48 |
| 最大候选 log-prob 误差 | 0.2460136414 | 0.3011400700 |
| actor / rollout 联合动作概率比下界 | 0.6835159661 | 0.7277374899 |
| actor / rollout 联合动作概率比上界 | 1.4110209245 | 1.4339686211 |
| 首选候选变化的 Span | 8 / 144 | 5 / 144 |

最大误差恶化，部分汇总量改善，不能宣称整体对齐改善。
原阈值仍为逐候选 logp 误差不超过0.02、所有完整动作概率比处于[0.98,1.02]。
基座及adapter身份、音频/特征/token/问题/位置/候选均保持相同。

## 本轮实现与实际范围

固定镜像及物理 NPU4–7 与原生实验一致。使用 Qwen3-ASR-0.6B，
BF16 未合并 LoRA identity
`6df66622568c868b2ab06efffdfd56fcfb3e512727a764953865ce0b68a1e986`。

独立 profile `qwen3-asr-bf16-hf-order-tp4-diagnostic-v1`：

- 关闭 internal format；音频 windowed eager attention；语言 HF-order RMSNorm/residual。
- 按实际本地 Q/K/V 输出宽度分别 BF16 GEMM，保留 native LoRA 加法和 adapter 切片。
- 本地 gate/up 分别计算，显式 BF16 SiLU及乘法；默认 full-head NeoX RoPE。
- 单序列完整 causal eager attention，保留 native 输出投影、MLP down projection 和通信。
- 独立 TP4 架构及 worker RPC；现有 TP1 默认检查与训练准入 profile 继续拒绝 TP4。

四 worker 均返回28层 QKV/MLP/attention 对齐回执。
音频塔18层的实际head partition均为1，即本轮音频注意力在各worker复制执行，
并未按四卡切分head；四卡head分片测试仅覆盖可支持的语义，不冒称实际采用该分片。
独立复核脚本检查配置与源快照pins、48条完整请求身份、真实BF16 head/LoRA、
完整候选与所有联合动作概率误差。不会输出训练 READY。
实际运行快照发布后未修改；随后仅补充CPU测试、注释和文档。

相关CPU测试共27项通过，包括局部 grouped-query heads、音频窗口head分片与
分别切片再打包QKV的数学语义，以及诊断profile不能进入训练的限制。
这些测试不证明硬件分片归约与单卡BF16舍入一致，真实结果如上。

## 下一步定位

证据表明 TP1 的局部 HF-order 运算移植不足以解决 TP4 差异，尚不能确定最早偏差层。
下一步逐rank独立保存诊断trace，比较完整边界状态及局部分片，定位音频、语言投影、
LoRA和归约中最早偏差点。禁止多rank向同一路径写trace，禁止用参考激活替换模型计算。

RowParallelLinear源码显示本地矩阵结果先输出，再归约；BF16局部结果已舍入，
仅把后续求和升为FP32不能保证恢复单卡完整矩阵乘法。这是待实测定位的机制候选，
不是本次模型误差已归因的结论。不放宽概率阈值，不把BF16主矩阵执行改成FP32来过验收。

现有四卡FSDP actor + TP1 vLLM已验收路径仍保留。真实FSDP恢复、完整verl worker、
更大清洗数据上的学术基线与RL泛化实验仍是未完成工作。

证据：`span-tp4-hf-order-suite-v1-launch.json`、`span-tp4-hf-order-suite-v1-verified.json`。
