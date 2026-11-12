# 任务先行、统一工程与RL配置基线

2026-11-12，执行用户最新顺序：任务定义和设计→工程搭建、验收和锁定→模型训练。
撤销此前“全量processor就绪后自动恢复Stage1”的推进安排。当前没有本项目训练派发
控制器运行；既有只读CPU预检可完成并归档，不触发训练。NGA其他项目进程不处理。

## 1. 现在实际用的是什么

最近NPU两步前缀调用`span_hybrid_actor`，stage=instruction，仅优化分类CE，没有RL。
已实现的task分支使用`span_finite_credit.verl_loss`：

```
A_j(a) = w_j * (1[a=y_j] - p_old,j(y_j))
L = E_{a~p_old,j}[PPO_clipped_surrogate(ratio_j(a), A_j(a))]
L_task = L + lambda(t)*CE
```

期望对每个有限候选精确求和；各Span边缘分别裁剪；调用固定verl的
`compute_policy_loss_vanilla`，clip=.2，dual clip=3，没有critic、GAE或GRPO组标准化。
这是“有限候选精确期望的边缘PPO定制变体”，不能称为标准PPO或GRPO。
它只精确处理当前可加奖励，不能直接宣称精确优化集合F1或组合正确率。
历史小样本RL未形成可信的独立任务提升证据；混合RL+CE仅有真实四rank CPU梯度证据，
不能称本轮全量RL已经运行。

## 2. 统一底座，不为算法各造一套系统

保留一个固定镜像、模型/processor、音频处理、Q编译、typed策略、LoRA/FSDP、vLLM、
数据/监督映射、保存恢复和评测链路。算法配置只切换优势估计、critic及明确登记的loss。
用统一的request/action样本接口，不把候选概率假装成生成token轨迹。

候选底座：本地Qwen3-ASR-0.6B，BF16执行，FP32 master/Adam，语言q/v LoRA r4/alpha8，
175物理卡4–7 FSDP DP4，vLLM TP1。镜像固定`52ba94e1…`，verl commit `e9405d…`。
HF/vLLM修复后256请求一致性已通过；全量恢复、新Q、新算法未全部验收，因此这是
release candidate，不把部分检查通过写成稳定正式基线。

冻结后普通实验仅改变登记配置。真实工程缺陷另建修复分支/发布版本，重跑受影响验收，
曲线保留版本边界；不在运行快照上热修，不把每个实验都变成基础设施重构。

## 3. 登记三种通用算法配置

固定同一完整Span赋值为一个动作。当前策略
`pi(a|x)=product_j pi_j(a_j|x)`，joint logprob为各Span logprob之和。
它是一轮contextual-bandit决策，不是Q1答案驱动Q2的时序环境。

| 配置 | 优势估计 | 价值模型 | 本项目实现边界 |
|---|---|---|---|
| PPO-critic | `R - V(x)`，一轮下为GAE的退化形式 | 需要，仅消费合法音频/任务输入 | 标准clipped PPO；需新增并验收critic生命周期 |
| GRPO | 同输入G个动作，`(R-mean)/std` | 不需要 | 保留组相对优势；使用单一联合动作ratio，明确是typed-action适配 |
| RLOO | `R_i - mean(R_others)` | 不需要 | 登记为RLOO优势＋PPO裁剪，不能冒称无裁剪REINFORCE原实现 |

本地固定镜像源码实际存在三种优势函数及PPO loss，已核验core_algos SHA256
`fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24`。
源码支持不等于我们的音频Span端到端接入完成。
算法出处：PPO [1707.06347](https://arxiv.org/abs/1707.06347)、GRPO/DeepSeekMath
[2402.03300](https://arxiv.org/abs/2402.03300)、RLOO
[2402.14740](https://arxiv.org/abs/2402.14740)。本轮核验的是本地实现，不宣称重新检索所有最新变体。

初始可比较配置登记在[配置文件](../../configs/experiments/rl-baseline-design-v1.json)：
G=16、temperature=1、clip=.2、不用dual clip、每批行为样本2轮优化、相同参考KL及
衰减CE。PPO/GRPO/RLOO使用相同输入、采样预算和joint ratio；critic成本单独计量。
这些是待任务定稿与工程验收的初始值，不是已证明最优或已允许运行的正式锁定配置。
最终发布还须固定LR、有效batch、critic细节、随机种子集合和完整评测预算。

现有精确边缘PPO作为第四个定制诊断配置保留。与采样joint PPO的差异包含枚举/采样、
信用分配和裁剪层级，不能把差值单独归因于“精确优势更好”。需要分因素消融时单独登记。

## 4. Step-wise应该改什么

这里step是语义判决节点，不是默认的时间步。首要调整是奖励、信用、归一化，而不是
把每个Span机械地传给GAE当作连续状态。Final经共享隐藏表示学习；当前离散前置输出
没有被Final读取，不得用RL记号伪造因果依赖。

**先不改优化器。** 三种基线先用相同Final reward，建立可复现的任务曲线。然后在同一个
基线、相同输入上加入有可靠监督的Q信用：`R=R_Final+0.2*mean_valid(R_Q)`。
Final与过程分量分别记录；缺标mask。辅助CE固定并单列，不做RL/SFT胜负对照。

**再改节点信用。** 研究`A_j(a_j)=E[R|x,a_j]-E[R|x]`。可加正确性奖励可精确计算；
绑定或集合关系因子用小范围枚举/预先登记的采样，不能声称穷举大集合。
只有Final奖励且离散动作因子化时，独立过程动作的精确Final信用是0。
给每个过程动作广播同一组优势可能产生采样噪声；条件信用是否改善Final需要实测。

此时若由联合ratio改成节点ratio，就成为新的算法变体，必须另立ID，不在GRPO/PPO/RLOO
基线名称下静默替换。过程reward与CE同时变化的收益也不能只归因于RL信用算法。

**处理尺度与稀疏性。** 不对Final和每个Q分别单位方差标准化后再声称过程总预算仍为0.2。
GRPO总奖励组标准化会改变跨query权重；RLOO保留奖励尺度但方差可能更高。
G=16近均匀60类下全错概率约`(59/60)^16≈76.4%`；记录零方差组，不暗删难组或
自动重采样来抬高数据效率。CE可提供补充梯度，不能将混合目标收益全部称为纯RL收益。
多Q loss按真实query归一化；类别平衡在外层预先定义，不能指望GRPO组内奖励乘常数
在标准化后仍保留类别权重。过程梯度范数/方向与Final同报，奖励预算不等于梯度预算。

## 5. 三个发布门槛

**任务冻结**：I/S/B等Q的真值语义、完整定义表、单/多意图和缺标政策、固定菜单、
Final关系、简单/复杂切片、source划分、reward与CE、指标和迁移协议全部版本化。
现有slot定义仍有draft/缺标，不能把当前intent工程发布扩大为全slot训练发布。

**工程冻结**：上述任务接口下实际processor/上下文、BF16、vLLM概率、四rank梯度和
padding/mask、连续更新/保存恢复、采样复现、critic（仅PPO）、完整评测及资源上界
通过。发布镜像+代码commit+配置+数据hash+验收矩阵。必要的短更新只能作为明确登记的
工程验收，不以此替代或提前开展研究训练。

**实验启动**：在锁定版本上准备Stage1的规约/Q消费训练及独立指标；固定Stage1起点后
运行已验收算法配置和step-wise消融。先报告每种算法自己的持续提升，再判断节点信用
是否提供额外收益；冻结权重和Q形式做FSC等迁移。基线复现和Omni同口径定位仍需完成，
不因新增算法配置取消原目标。

当前三个门槛未全部满足，新训练派发为false。继续可做任务设计、只读调研、已有验收
归档和必要的实现审计；不再以“工程跑通了几步”代替任务已经设计清楚。
