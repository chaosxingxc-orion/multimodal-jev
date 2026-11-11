# 规则学习的同起点、同请求预算对照

此前 temporal RL2 在规则任务上没有稳定收益，并非规则学习的等预算比较。
本实验直接对显式规则 Final 进行优化，检查真实音频条件下的学习方向。

固定 Qwen3-ASR-0.6B 与 SFT adapter
`7341430053a9b4f36f5d5b8fe80616889af1857c71187d22b4895b062f557140`，
四个分支都从全新 FP32 master AdamW 开始，lr=1e-4、clip norm=1、seed=17。
BF16 基座/LoRA 执行/导出；完整音频和 Q1/Q2/Final 模板一次前向，不生成文本。

| 分支 | 优化目标 |
| --- | --- |
| SFT Final | Final 候选归一化交叉熵 |
| RL Final | Final 正确率奖励，verl GRPO/PPO 联合动作损失 |
| SFT auxiliary | Final CE + 0.2 × 已审核前置事实平均 CE |
| RL auxiliary | Final 正确奖励 + 0.2 × 已审核前置事实平均正确奖励 |

Final-only 在此指监督目标，四分支输入均为 structured；不要将其与
`final_only` 单问输入 arm 混淆。未来单问/三问对照是另一个独立维度。
辅助项使用审核标签，不奖励自洽、答案长度或额外问题数量，不把缺失标签置 false。
Final 与前置输出共享 backbone；离散预测不反馈到另一个 Span。

使用 balanced-rule-controls-v1 seed17 schedule 的第一个 batch，四个独立
source group、每组 R/NOT R 两次请求，共8个请求，三个 Span 各4正4负。
四分支消费完全相同请求、各一次更新；RL 每请求32个完整动作，记录非恒定
reward group 数。更新数/输入预算相同，但计算和监督利用方式不同，不宣称 FLOPs 等价。

只准入已通过 v2 216 请求验收中的此8个请求。启动时重新核对原始审核来源、
规则真值、互补规则、schedule、源代码和 runtime 证据。原数据集的
`training_admitted=false` 不原位修改；限定范围的 protocol.json 固定哈希，
其授权仅覆盖当前一次学习试验，不授权任意批次或完整训练。

所有分支先做全动作概率 gate，随后优化；输出为 pending checkpoint，
不会发布 READY。独立 HF 重载和216请求 vLLM 全动作 gate 通过之前禁止下一步。
计划重算216请求的 Final accuracy、macro-rule BA、all-span、互补规则翻转、
事实预测+规则执行器基线及相对初始策略的配对变化。
另以有限候选精确枚举验证过的期望奖励报告局部优化方向，不能替代准确率。

这8个请求及其余探针来源都属于已见 learning population。一次更新仅用于
实现与方向诊断，既不是学术基线复现，也不能报告未见音频/规则泛化、统计显著
收益或优胜算法。正式结论仍需更多更新、多 seed、同预算对照，以及独立审核的
source-heldout 和 task-heldout 划分。学术 SLURP 基线复现仍是独立未完成项。

首轮 `span-rule-matched-step-v1` 的 SFT 分支完成，RL 分支在真实更新和保存后
因回执重复 `scope` 字段失败；该轮不作为完成对照，不在原目录重试。
修复回执写入后，在 `span-rule-matched-step-v2` 从共同初始 adapter 和 fresh
optimizer 重跑。四分支均退出0，实际112个 Adam state 的 step 均为1。

四分支输入均为5710 token、含重复请求的音频26.62225秒；两个 RL 分支的
8个请求组全部有非恒定 reward。独立检查重算保存的行为分布、完整动作 logp、
奖励与前向准入，并核对真实 checkpoint 和 optimizer 内容。

| 分支 | 梯度范数（裁剪前） | 同8请求期望奖励（更新前→后） |
| --- | ---: | --- |
| SFT Final | 2.34375 | 0.487953 → 0.500265 |
| RL Final | 2.09375 | 0.487953 → 0.502631 |
| SFT auxiliary | 2.34375 | 0.596319 → 0.606000 |
| RL auxiliary | 2.109375 | 0.596319 → 0.610991 |

辅助分支包含额外0.2奖励，不能与Final-only的绝对值直接比较。
上述数值仅为更新所用8请求的局部期望奖励；不能据此断言 RL 优于 SFT。

RL 的记录 loss 接近0是组内中心化 advantage 在初始 ratio≈1 时的数值，
不是零梯度；实际梯度非零且 Adam step 已推进。不能比较 SFT CE 与该数值的大小。
四个新版本在 `span-rule-matched-evaluation-v1` 完成独立 HF 重载和216请求
HF/vLLM 评测，均 READY；最大候选 logp 误差 1.1920928955078125e-7。
完整对照见 `docs/engineering/span-rule-matched-comparison-v1.md`：未观察到稳定
RL 优势，不能把工程 READY 等同学术收益或下一批训练授权。
