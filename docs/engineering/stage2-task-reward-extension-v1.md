# Stage2 任务奖励与算法扩展约定

用户要求为任务相关奖励及算法调整保留空间，尤其区分完全正确与部分正确。
本设计针对当前简单单音频意图判决；仅使用clean TRAIN可信监督。
现行冻结双信用GRPO仍是排序/存在性二路，奖励为节点级0/1。
本次新增可测试的任务奖励组件与配置设计，未将新loss接入冻结训练器，也未派发实验。

## 三路信号

1. 排序：正确候选相对错误候选的排序，保留原pair credit。
2. 存在性：单候选是否属于该benchmark指定事件，保留原member credit及概率辅助loss。
3. 任务级：整组候选集合的质量，新增可选分支，区分exact、partial及false positive/negative。

第三路并不替代前两路，也不等同于概率校准。三个loss系数独立声明并记录梯度范数，
不能把新增reward直接塞进仅接受二值奖励的旧advantage函数。

## 奖励定义

给定完整可信真值集合Y和模型采样集合A，E=1[A=Y]，
Fβ=(1+β²)TP/((1+β²)TP+β²FN+FP)。定义：

`R_task = exact_weight × E + overlap_weight × Fβ`

两个权重非负且和为1，exact_weight严格大于0，以保留完全正确的额外奖励间隔。
初始设计示例为0.5/0.5、β=1；尚不是经实验证实的最优设置。
β<1更强调precision、β>1更强调recall，需在训练前预登记，不从selection或目标集调参。

| 单一正确意图的例子 | 示例奖励 |
| --- | ---: |
| 只选中正确意图 | 1 |
| 正确意图加1个错误候选 | 1/3 |
| 正确意图加2个错误候选 | 1/4 |
| 60个候选全选 | 1/61 |
| 只选错误意图 | 0 |
| 一个都不选 | 0 |

这就是当前数据可支持的“部分正确”：命中真值但包含误选。单标签意图任务的错误top1
不能凭语义相近获得部分分；scenario/action部分信用需另行可信监督和独立协议，本阶段
不暗中引入复杂任务。空输出和纯错误输出的任务分同为0，但FP/FN诊断及存在性信用不同。
拒绝缺失标签、非单一事件真值，不把缺标变为false。

后续可注册FP/FN代价、合法性或任务成功等不同版本组件，但必须有独立可评分监督、
明确范围及有界尺度；不接受任意配置表达式或未实现组件。当前执行组件只支持上述exact/Fβ。

## 与GRPO/RLOO的结合

同一音频一次模型输出可在CPU上抽取G=16组完整Bernoulli候选集合，所有任务reward
分量使用同一组动作，无需新增声学forward。采样动作必须绑定真实behavior policy、
输入、候选顺序、种子、query及update，不能基于gold构造“更好的”采样集合。

连续任务reward用独立的GRPO或RLOO组内优势：GRPO减组均值、标准差下限默认0.25；
RLOO减其他样本均值。零方差组优势为0，记录其比例；不要制造人工差异。
整体reward不能用于每次都相同的阈值决策再伪装成G个样本，否则没有组内学习信号。

未来接入的loss为`L_existing + lambda_task × L_task`。默认enabled=false、lambda_task=0，
配置预留0.1/0.2等预登记实验值，并保留exact-only和exact+overlap的对照。
任务动作概率使用所有候选bit logprob之和，ratio绑定同一完整集合的current/behavior；
不得把整组reward复制为各节点奖励后宣称等价joint-action GRPO。
joint ratio在60候选时可能方差大，必须报告log-ratio分位数、clip率和KL，再决定是否
需要独立命名的条件信用算法。不能直接exp到溢出，也不能随意截断改变目标。

独立sigmoid的NLL辅助项和Stage1 reference KL仍保留。任务效用、排序损失、概率质量
分别观测，不能以集合F1上涨掩盖NLL/Brier退化。

## 配置、实现和发布

- 配置：`configs/experiments/context-stage2-task-reward-design-v1.json`，不含本地机器路径。
- CPU组件：`src/multimodal_jev/context_task_reward.py`，提供配置验证、reward分量和连续组优势。
- 已有固定脚本入口继续用本地宏加载；新增配置必须纳入新运行包、行为记录和checkpoint身份。
- 正式训练入口仍拒绝不支持的objective；新字段不能静默忽略或假称已经启用。

有限验收需覆盖：实际采样重放、joint logprob/ratio、reward审计、DP/microbatch/padding权重、
BF16训推一致、LoRA导出/恢复，以及无额外音频forward。通过后按用户授权直接发布修复版
并接管对应阶段，不等待旧CPU前置任务；在跑的NPU工作按准确句柄接管或保留。
源训练只消费TRAIN；selection只按预登记规则筛选检查点；FSC/MINDS冻结迁移不参与调参。

当前发布边界是“CPU奖励计算可用、算法/训练接入待验收”，不是已经支持任意reward配置训练。
