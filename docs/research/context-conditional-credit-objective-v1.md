# 节点条件积分：独立候选目标与离线验证

当前正式GRPO/RLOO及其队列不变。本候选将此前输出方向诊断转为可自动微分的明确loss，
仅在tools/reports中实现，没有接入actor或获得训练准入。
配置：configs/experiments/context-conditional-credit-design-v1.json。
后续直接核对原论文确认：核心梯度是Local Expectation Gradients的独立Bernoulli特例，
不作为新估计器创新主张。原始方法与适配边界见
[已有研究核对](context-conditional-credit-related-work-v1.md)。

## 目标与成立条件

对每个完整行为动作a和节点j，固定其他位，计算Delta_j=R(a_-j,1)-R(a_-j,0)。
损失为 `L=-mean_query sum_j p_theta,j * stopgrad(mean_G Delta_j)`。
在theta等于行为策略、其他位按独立Bernoulli行为分布采样时，其负梯度是原始期望reward
梯度的无偏估计。节点求和、query与G取均值；不再除节点数，也不组标准化。
这是坐标梯度代理，标量loss数值不是负期望reward，不能用它的绝对值解释任务效用。

每批只更新一次，不使用ratio/clip、不重复使用旧行为批次；它不属于原GRPO/RLOO配方。
独立helper要求当前和detached行为概率精确相等，奖励停止梯度且完整有限。
这只是公式层保护，不能替代生产端policy/input/action身份绑定和训推一致性检查。
未来分布式接入还需保持原query权重、padding零权重及累积归一化。
KL、排序和概率辅助项沿用简单任务配置；当前helper只验证RL项。

## 本次实证范围

11项本地检查通过，1.96秒。三位全部8动作穷举，覆盖单标签F1、耦合reward和加性reward；
利用共享线性参数映射比较期望代理梯度与完整分布期望梯度。这是小型数学模型，非Qwen梯度。
另验全等错误采样的有效方向、query/G复制不改变尺度、过期行为及非法输入拒绝。
六位全部64动作验证：独立Final/过程输出的加性过程reward，对Final输出坐标的原始
期望reward梯度贡献为零；共享参数仍可有迁移或冲突，不能推广为实际训练无收益。

既有第74步真实TRAIN128归档也重放：先重验前次报告及其15项来源hash，重建G16动作
并核对原action身份，再逐节点枚举0/1计算reward pairs，反向得到输出logit方向。
128条均与前次解析条件积分方向一致，最大绝对差1.1099519740820352e-16。
这是CPU FP64、已有行为策略上的局部目标验证，没有新音频推理、NPU更新或heldout评测。
完整文件身份与验证范围见context-conditional-credit-objective-validation-v1.json。

## 不承诺的性质与后续

仍有p*(1-p)饱和；部分错误节点仍可能得到零方向；不把非零方向等同可观测准确率收益。
奖励枚举增加CPU开销，通用实现最多2*N*G评分/query；可用F1解析差值优化，须实测。
不需要额外音频前向不等于零计算成本；一次optimizer更新也没有自动的KL上限。

先完成既有简单GRPO/RLOO及标准复杂Q对照。该候选若进入工程，须独立验证真实输入/动作
来源、DP和累积权重、BF16数值链路、连续更新及新进程恢复，然后才能以同Stage1起点、
全量数据和固定评测计划比较Final、NLL/Brier、成本和迁移。当前不派发、不新增队列。
