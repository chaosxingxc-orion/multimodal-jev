# 有限动作条件信用分配候选

此前联合动作采样的诊断显示：只奖励 Final 时，前置动作的精确期望策略梯度
为零，有限样本仍会产生前置输出梯度噪声。这里测试消除该噪声的候选，
不据此声称它必然改善共享 LoRA 参数的梯度或任务指标。

当前策略独立采样各 Boolean Span，奖励为 Final 正确性加权前置事实正确性。
对每个 Span 的两个候选精确积分，条件 advantage 为
`A_j(a) = w_j * (1[a=gold_j] - p_old_j(gold_j))`。
Final 权重为 1；前置权重为 auxiliary_weight 除以有效事实数；缺失事实为 0。
行为概率及 advantage 均停止梯度，不进行组内标准差归一化。

使用实际 verl vanilla PPO：逐 Span 边缘概率比，clip=0.2、dual clip=3，
log-ratio 限制 [-20,20]；通过 rollout_is_weights 实现旧策略有限支持积分。
这些权重不用于修正后端漂移，完整动作概率准入仍然必须通过。
这是独立命名的边缘 PPO 代理目标，偏离旧策略时不等价于联合动作 PPO。
verl 返回的 clipfrac/KL 是展平支持上的默认诊断，不当作行为分布加权的真实 KL。

同策略点的梯度等于负精确期望奖励梯度；它也是按当前 gold 概率缩放的
分类交叉熵梯度。因此本轮单步实验不能区分“裁剪机制收益”，不能凭有限支持
枚举声称原创 RL 算法；要进一步对照期望奖励优化、SFT、联合采样 GRPO，
并检查多步和多 seed 的收益、校准与泛化。

9 项数学测试验证精确枚举、on-policy 梯度、缺失监督与问题数尺度。
固定训练镜像的 CPU 容器完成 10 组 actual verl 对照，所有损失/梯度一致，
梯度最大误差 0，覆盖未裁剪、PPO clip、dual clip、log-ratio clamp。
CPU 证据根目录 `span-conditional-loss-cpu-v1`，没有 NPU 更新。

实际训练协议 `span-rule-conditional-step-v2`：ASR 0.6B、同初始 SFT adapter、
物理 NPU5、同 matched-v2 的 8 请求/4 来源/每 Span 4正4负、seed17、fresh
AdamW、lr=1e-4，各一次更新，aux=0 和 0.2 两分支。执行/导出 BF16，master
及 Adam FP32。不读取 selection 预测来构造目标或奖励。采样动作仅沿用完整
准入记录接口，损失消费精确候选而非 32 个随机动作。
导出始终 pending，实际优化器审计、独立 HF 重载和完整 216 请求 HF/vLLM
准入通过前不能继续训练该版本或宣称工程验收，更不能宣称学术收益。

v1 在输入文件校验时因漏挂载 CPU 审计目录退出 1，未构造模型、未更新参数。
v2 增加该目录的只读挂载，在新根目录从相同初始策略启动，保留 v1 失败证据。

v2 两分支均完成实际更新，容器退出 0。独立 verifier 核查 112 个 FP32 Adam
state 的 step=1、行为策略身份、精确 advantage/权重、记录损失与 checkpoint。
aux0 的训练请求期望奖励从 0.487953 升至 0.510888，aux0.2 从 0.596319
升至 0.613173；梯度范数分别 0.464844、0.511719。局部概率改善不证明准确率
或泛化提升，不把不同辅助权重的绝对奖励相互比较。

`span-rule-conditional-evaluation-v1` 已派发两个版本各 216 请求的独立 HF
重载和 vLLM 全动作概率验收；在其完成前导出仍为 pending。
