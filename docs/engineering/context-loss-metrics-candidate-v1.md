# 简单任务混合损失分项记录候选

当前实际actor把内部objective返回的metrics丢弃，只记录混合context_loss和原生pg_loss。
因此无法直接区分RL、排序、概率监督、自定义reference KL。本候选复用这些已有分项，
不增加backbone/rollout调用，不改目标函数或累积系数。当前正式训练和排队包没有安装此候选。

实现：`tools/reports/context_loss_metrics_candidate.py`。从实际arithmetic-v2封包actor
SHA256 a785016786955441b129b5432dec900259aaf80c197385f2e6773d563d865b29派生独立源码，
逐一核对替换锚点，使用独立注册名称与ContextVar。不是运行时monkeypatch或启动器。

## 口径

- ranking/probability numerator先乘world_size/global_real_requests，再分别乘当前系数。
- sampled RL和KL已经归一化，不再乘上述因子；KL另乘当前KL系数。
- 保留原始归一化分项、加权分项、重建总值与原context_loss。
- microstep分项先在同rank求和，再跨rank取均值；不能直接平均所有microstep。
- 容许分项独立转Python float后的FP32舍入误差，以绝对误差2e-7加分项绝对值和的2e-6
  作为一致性检查容限。这不是优化阈值、校准阈值或实际NPU误差验收结果。
- stage/分母/辅助CE关闭契约不符、非有限分项或总值不一致时拒绝。

原始loss张量及累积缩放返回路径保持；新增逻辑只消费原objective已经detach产生的数值。
不试图从旧日志还原缺失的原生分项，也不把loss标量当作梯度强度。

## 实际验证与边界

8项本地CPU检查通过，0跳过：从已固定actor提取实际registered_loss函数体，
使用真实辅助损失和可微采样项替身，检查FP32/BF16 logits的loss、梯度、一次FP32 AdamW
更新精确相等，覆盖padding零梯度、错误分母、隐含CE、来源篡改和microstep汇总。

首次BF16检查失败：测试替身把采样项也留在BF16，违反实际概率/损失FP32合同；修正替身
为FP32后通过。没有放宽总值检查容限来掩盖失败。

本地没有verl/omegaconf，因此未声称真实原生surrogate、完整actor、分布式或NPU已验收。
下一步在固定镜像中使用真实目标函数验证loss/梯度/更新一致性与运行成本，之后才考虑
把此候选纳入新的实验身份。当前已有11点评测和GRPO→RLOO→迁移队列保持。

测试与源码身份见同目录`context-loss-metrics-candidate-v1.json`；生成源码及JUnit在
`.local/decision-objective-research/context-loss-metrics-candidate-v1/`及同名前缀JUnit文件。
本轮364项基线身份、六组1078/6/2/1097/1073/1167依赖核验无变化。
