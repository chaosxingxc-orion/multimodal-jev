# 共享工程与算法收口决议

本决议复用 [唯一工程入口](../engineering/context-prob-v1-baseline.md)，不建立新运行基座，
不改当前训练及队列。顺序固定为任务定义、工程/配置冻结、有限验收、正式实验。

## 当前实际执行

本次重新验证原锁的364项身份，全部通过。原锁的stage2_released=false是历史快照；
当前Stage2准入以arithmetic-v2独立发布和验收收据为准。
live-v43（local Unix 1794720525.3656836）确认原GRPO容器running/nonOOM，四rank均266/634，
64/128/192/256检查点已提交。下一普通观察不早于1794721125.3656836。
这是训练进度，不是完整选择集曲线或RL收益。

共享Qwen3-ASR-0.6B、q/v LoRA r4/alpha8、一层Transformer概率head、verl actor/vLLM，
BF16执行/导出与FP32 master/Adam/loss。175卡4–7，DP4，global128=micro8×accum4×4。
全量clean SLURP TRAIN40513；selection9226；clean TEST12991不调参选模。
固定策略评测、Stage1指令消费适配、Stage2任务RL采用同一工程的三种用途配置。

| 算法 | 冻结角色 | 实际状态 |
| --- | --- | --- |
| GRPO | 原生verl组标准化优势，联合Bernoulli动作clipped surrogate | 正式主线运行中 |
| RLOO-clipped | 原生verl leave-one-out优势，同一个surrogate与预算 | 已封包，原队列等待GRPO及其完整选择曲线 |
| PPO-critic | 单步上下文V(x)基线 | 仅设计；value更新及恢复未实现，不可派发 |

算法注册沿用configs/releases/context-prob-v1/algorithms.json。
DataParallelPPOActor是复用的执行循环，不意味着当前训练有critic。
每query采样G16个60-bit动作，reward为集合F1，clip0.2/dualclip3，一个behavior批次一次更新。
共同损失为RL + 0.01×逐候选求和KL + alpha×ranking + alpha×binary NLL；
alpha在前80%更新从0.3余弦衰减到0.1。不是单纯CE，也不能将混合目标收益全归于RL。

## 任务定义是新增实验的前置条件

每个任务先交付部署可构造的Q/候选定义、prob及harness消费规则、真值来源和缺标mask、
合法动作、reward/loss、数据划分、指标/选模规则和比较假设。
简单exclusive_event任务已进入正式训练；不等待复杂任务设计再重跑它。
复杂属性任务使用既有前瞻设计context-native-attribute-comparison-design-v1.json：
Direct60、Q-input124、Q-credit124（18 scenario + 46 action + 60 Final）。
Q-credit只增加0.125×scenarioF1+0.125×actionF1；Q两组均无过程辅助loss。
标签是原生Final投影，不能称独立语义过程gold；缺失语义槽位标注不补false。
复杂任务须完成自身全量输入审核和真实NPU训推/更新/恢复验收后才能派发。

## Step-wise算法的调整边界

保留原GRPO/RLOO作为首轮Direct/Q-input/Q-credit的估计器。
一次前向的Q/Span不是环境时间；不能沿问题编号套GAE或生成式reward-to-go。
Final不会读取同层刚输出的过程概率，先研究共享上下文和共享参数的收益。

这里有一个需要明确的数学边界。固定输入x，若策略输出独立的Final与过程动作，
且奖励为R_F(a_F)+lambda×R_P(a_P)，则

`E[R] = E[R_F] + lambda E[R_P]`，
`d E[R] / d z_F = d E[R_F] / d z_F`。

z_F为Final输出logit：过程项对Final输出坐标的原始期望reward梯度为零。
这个结论来自独立分布归一化，不依赖toy实验。它不意味着过程训练对Final无用：
共享参数可能产生正迁移或梯度冲突；有限采样、GRPO标准化、clip和KL也改变实际更新。
因此Q-credit改善Final时应首先解释为共享表示/优化收益，不能直接称概率链的因果推理。
不为制造耦合而临时增加一致性reward或修改Final决策规则。

已有第74步单批诊断：128组中112组奖励恒定，原生GRPO/RLOO优势为零；其中26组阈值决策错误。
这支持研究信用稀疏，尚不足以推断完整训练规律或直接替换正式算法。
后续独立候选优先级：

1. 分项优势与固定尺度：解决Final恒定时GRPO标准化抵消过程lambda的问题；
   必须先固定分母、零方差处理及clip，作为独立变体比较。
2. 节点条件积分：奖励可评分时，固定其他动作，枚举本节点0/1，使用
   `p_j(1-p_j)[R(a_-j,1)-R(a_-j,0)]`估计原始期望reward的logit方向。
   不增加音频前向，但增加奖励计算；不能冒充原GRPO，也不保证解决概率饱和。
   若升级训练，先定义完整loss、行为策略绑定及单次更新边界，再有限验收。

这两项不成为当前简单基线的前置任务。PPO-critic也不因“通用”而优先扩建。

## 有限交付顺序

1. 原GRPO完成634更新及固定11点、每点完整9226 selection评测。
2. 原RLOO-clipped从相同Stage1终点、重置optimizer，以相同曝光/预算完成对应曲线。
3. 沿原条件队列开展冻结策略FSC、clean TEST和MINDS评测；目标集不参与选模。
4. 简单闭环后，再验收并发布既有复杂Q比较；依据Final收益、概率质量和NPU成本决定算法变体。

吞吐候选保持独立，未通过真实NPU容量/概率/恢复验收前不替换当前micro8配置。
仅真实阻断缺陷触发受影响路径修复，不重做已通过的全套验收、不拼接修改前后曲线。
当前仍缺完整RL曲线、TEST及迁移结果，不能宣称学术目标已达成。
