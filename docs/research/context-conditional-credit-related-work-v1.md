# 节点条件积分的已有方法依据与研究贡献边界

结论：本项目当前节点条件积分候选的核心梯度，是Local Expectation Gradients在独立
Bernoulli决策与可计算任务奖励上的特例。它可作为有依据的算法适配候选，不能声称发明了
新的梯度估计器。这里修正的是学术定位；已冻结的设计、运行配置和算法比较不变。

本次直接访问五篇原始来源，针对方法相关段落阅读；不是综述穷尽检索或论文全篇复现。
抓取时间、原始文件SHA、正文提取SHA和阅读范围见同名JSON。

| 来源 | 已核对内容 | 与本项目关系 |
| --- | --- | --- |
| [Titsias与Lázaro-Gredilla，2015，Local Expectation Gradients](https://papers.nips.cc/paper_files/paper/2015/file/1373b284bc381890049e92d324f56de0-Paper.pdf) | 第3节式8–11与算法1：固定其余变量，对一个变量精确求条件期望；复用pivot样本 | 当前梯度的直接前身 |
| [COMA](https://arxiv.org/html/1705.08926) | 式4：中央critic估计Q，固定其他agent动作，边缘化本agent动作作baseline | 与此前节点反事实baseline形式相关；当前无多agent环境、无learned critic且进一步积分本节点，不是复现COMA |
| [ARM](https://arxiv.org/html/1807.11143) | 定理1式4的增强/反向配对采样；式5的单Bernoulli精确梯度 | 是二元随机决策的相关估计器，当前未采用其耦合随机采样 |
| [DisARM](https://arxiv.org/html/2006.10680) | 对ARM增强随机性做解析条件积分；相关工作明确介绍LEG；实验包含REINFORCE LOO | 不应把任何Rao–Blackwell化统称DisARM；本候选不是它的实现 |
| [Rao-Blackwellized Stochastic Gradients for Discrete Distributions](https://arxiv.org/html/1810.04777) | 对高概率类别求和、尾部采样；相关工作明确区分此法与逐变量LEG | 当前是逐节点积分，没有高概率完整动作截断 |

## 与本候选的精确对应

LEG式11对一个变量写为sum_a_j grad q_j(a_j) R(a_-j,a_j)。取q_j为Bernoulli，
参数为logit z_j，得到

`dJ/dz_j estimator = p_j(1-p_j)[R(a_-j,1)-R(a_-j,0)]`。

这正是已有候选loss `-sum_j p_j*stopgrad(mean_G DeltaR_j)`在行为策略点的负梯度。
原论文先讨论每个因子独立参数；本项目共享模型参数可通过链式法则对各logit梯度求和。
已有小型共享参数测试验证该恒等关系，但不能据此声称共享Qwen训练方差一定更低。
逐坐标条件化不直接给出相对RLOO、GRPO或共享参数梯度协方差的统一优越性结论。

当前任务属于带可计算gold奖励的单步上下文策略优化；解析枚举可行来自离线完整标签。
这与环境交互、多步探索或学会生成推理链不同。推理时不提供gold，也不运行奖励枚举。
不能将此方法对当前单标签F1的可行性自动推广到缺标槽位、开放槽值或工具环境回报。

## 怎样形成有价值的研究结果

需要证明的不是公式换名，而是三个相连的结果：

1. 可配置候选定义和语义规约能否支持端到端语音概率决策，并迁移到FSC/MINDS等任务。
2. 相同起点和曝光下，LEG式信用是否相对标准GRPO/RLOO改善Final、概率质量和计算效率；
   全等奖励组仅是诊断，不能替代完整选择集/测试结果。
3. 有标签支持的过程规约是否增加Final收益。独立节点加性奖励不直接耦合Final输出坐标，
   因此必须保留Q-input/Q-credit对照，不能以过程准确率或一致性分数冒充推理收益。

计算对照需区分音频前向成本与reward计算：G16复用同一音频表示，逐节点枚举有额外评分。
通用实现2*N*G是保守上界；缓存原动作reward可复用与原位相同的分支，F1还可解析差分。
这些优化均不是已经测得的吞吐收益。ARM/DisARM提供另一种奖励调用成本的参照，
当前不据文献阅读新增训练队列或宣称候选优于它们。

既定优先级不变：完成当前简单基线全曲线与冻结迁移评测，再用可靠数据比较复杂Q和独立
信用候选。若未来论文强调新算法，必须给出超出现有LEG/COMA/ARM体系的明确改动、
理论或实证依据；目前尚未建立这一创新性结论。
