# 固定镜像原生verl复核第74步奖励组

接续[真实动作重放](context-actual-reward-groups-step74-v1.md)，本轮在正式固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
调用真实verl的`compute_grpo_outcome_advantage`与`compute_rloo_outcome_advantage`。
`core_algos.py` SHA256严格匹配训练绑定的
`fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24`。

输入是上轮已绑定TRAIN/音频/张量/gold并逐组验证动作hash的128×16奖励，
按实际request ID完整分组；没有重新采样或调用语音模型。

| 检查 | GRPO | RLOO |
| --- | ---: | ---: |
| 非零优势组 | 16/128 | 16/128 |
| 112个恒奖励组的最大绝对优势 | 0 | 0 |
| 与本地公式的最大绝对差 | 0 | 7.8231e-8 |

这补齐了前一报告仅本地公式诊断的限制。RLOO的小差异在预设2e-6检查阈值内，
恒奖励组结论精确一致。它仅复核两种优势估计器，不是NPU actor参数梯度、clipped
surrogate或optimizer重放，不能由此断言整个混合loss没有梯度。

容器`b1624173b79ae995a66869d1768ba0bc20dcdeb0c0b436d49218654620e20b47`
准确wait返回0，独立inspect确认exit0/nonOOM，无NPU设备或privileged权限。
只读两文件输入包hash、镜像、runner/input hash及输出均已收集核验，热路径位于/data。
原训练/输入准备/后续队列均保持，不升级算法、不新增训练前置门槛。

证据：[完整优势及收据](context-reward-replay-native-verl-v1.json)。
下一步仍是收齐预登记的完整selection曲线和概率质量，再判断是否采用独立的信用/探索变体；
不能仅凭这一个TRAIN批次选择算法或宣称跨任务收益。
