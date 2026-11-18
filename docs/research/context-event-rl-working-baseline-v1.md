# 第一个上下文概率头RL工作基线

本页保留最初研究定义，末尾工程进度为撰写时快照，不作为当前派发依据。
当前权威入口为`docs/engineering/context-prob-v1-baseline.md`，冻结配置为
`configs/releases/context-prob-v1/{experiment.json,algorithms.json,LOCK.json}`。
Stage1局部执行已准入，Stage2连续运行尚未发布；不能把早期pending清单当作全部仍缺失。

任务和比较方案固定于`configs/experiments/slurp-context-event-rl-v1.json`；工程准入
仍未完成，不代表训练已开始。这是第一套可检验研究基线，参数不是已证最优。
保留后续多适用候选/槽位语义研究，首轮先利用完整可靠的SLURP事件监督。

## 数据与输出

全量clean learning40513，selection9226，官方TEST不参与优化/选择。每条音频供给
完整60候选定义，逐项判断是否为benchmark指定事件，输出独立sigmoid概率。
一正59负只适用于这一明确语义，不能作为“语音中是否提到某意图”的负标注。
请求级自然录音权重为1，补齐为0；通过全覆盖日程分散类别，不改变标签先验。

两阶段都训练同一backbone LoRA+共享浅Transformer head。Stage1先学习音频、
问题和候选绑定，两轮634更新，Lrank+LbinaryNLL，均按query内均值、全局query
分母归一化。对比loss是监督项，不能叫作RL；NLL提供概率标尺，但不保证校准。
旧Final60的CE训练不重命名为本阶段，也不直接复用其输出概率作为新head的KL参考。

## RL目标

Stage2从本Stage1最终联合策略开始，GRPO为首个运行算法，RLOO保留相同数据、
起点、预算的配置基线。一次backbone前向得到60个概率，采样G16个完整二元tuple；
不将G16复制成16次相同音频forward，不生成文本或把采样答案反馈到后续问题。
探索分布的因子化只是计算约定，不声称候选语义独立。

奖励为完整集合F1：`2TP/(2TP+FP+FN)`；空真值/空预测为1，其余按公式。
SLURP本轮真值始终单正例，无空真值。缺任一标签时拒绝完整集合奖励，不补负例。
一正60候选的返回空集奖励0、返回全部2/61、仅正确候选1；多正例只返回一个
不会获得满分。这比逐位准确率避免多数负例投机，但不是概率校准目标。

保留固定verl的组优势、joint PPO ratio/clip=.2/dual-clip c3，不擅自修改算法。
完整组先算优势再拆DP；零信号组保留在全局分母。RL loss加固定Stage1参考的
Bernoulli KL（逐候选求和，系数.01），加rank/NLL辅助，两系数均从.3在前80%
更新余弦衰减到.1。两轮634步，每query每更新仅计一次辅助和KL。

F1可能鼓励不理想的概率阈值行为；必须同时报告NLL/Brier、返回数量、precision/
recall和完整集合正确率。观察到退化就报告，不以提高F1掩盖概率质量下降。
过程奖励和复杂任务信用分配后续单独对照，不把未标注中间问题编造为过程真值。

## 效果与迁移

按配置预定检查点对完整selection评测；起点是同一新head策略的RL step0。
展示全部轨迹，检查是否持续提升，不只挑最好一步；不新增RL对SFT胜负实验。
选择按macro set-F1，再按binary NLL，再按最早步；阈值固定.5，不在TEST上调整。
原60类top1/macro-F1仅作业内基线联系指标，不替代候选集质量。

FSC首先以冻结SLURP权重、替换任务候选定义/规约测试配置迁移，源域RL step0
与选定策略使用同一FSC评测人口；不在FSC再训练或调阈值来冒称迁移收益。
FSC较简单，不替代SLURP主任务。复杂多标签/槽位类型实验须另有有效监督及规约。

正式训练前仍需完成生产token/runtime身份、多卡原生verl更新/恢复、联合策略
在vLLM的激活和较广输入一致性。当前两真实音频的融合HF/vLLM概率已完全一致，
256条60类/micro8对照正在运行；这些工程证据不构成任务效果。
