# 首条12步规则隔离训练链验收

`span-transfer-conditional-s17-a2-v3` 退出0，独立全链 verifier 通过。实际使用
Qwen3-ASR-0.6B、事实SFT起点、seed17、structured输入、aux0.2条件信用分配，
12次更新消费96请求（重复计）、70,514输入token、338.678秒音频。约60分钟
墙钟包含加载、逐版本HF/vLLM验收和审计，不能当作纯训练吞吐。

全部13个版本通过固定216探针加相邻批次的完整候选/联合动作gate，最大
logp误差1.1920928955078125e-7。12次更新的独立HF重载误差均0；实际112个
FP32 Adam状态依次推进至step12，数据游标及父子adapter身份一致。执行与
导出BF16，未合并adapter。没有在selection上计算训练奖励。

最终adapter：`78177b420c6573ee3290b745155b201b2465e41f7ed5e5ce340bd970d67f564b`。
固定checkpoint：`run/update-0012/run/policy-1`。整个轨迹无例外语义组训练；
该组的探针只用于按固定容差验收概率一致性，不用于loss、reward或挑checkpoint。

该结果证明本次单卡顺序控制器的真实verl损失更新和vLLM概率生命周期。
每步进程重载成功，不等于已完成意外中断恢复、四卡持续训练或完整Ray trainer。
不能从工程验收推断Final准确率提升或跨任务收益。

下一阶段已派发：

- NPU4：`span-transfer-conditional-s17-a2-evaluation-v1`，初始和固定末次策略各
  消费learning全部2,520请求及selection全部2,304请求，共9,648次直接音频
  HF前向。单独报告四个来源×规则单元、配对来源级区间。无新checkpoint选择。
- NPU5：`span-transfer-sft-s17-a2-v1`，与条件方法相同事实SFT起点、seed17、
  三问、aux0.2、12次更新及平衡批次，改用候选交叉熵。仍逐版本验收。

这两项完成前不作算法胜负结论；联合GRPO、其余seed/输入形式及跨数据集
实验仍待执行。SLURP原生学术基线复现仍是独立缺项。
