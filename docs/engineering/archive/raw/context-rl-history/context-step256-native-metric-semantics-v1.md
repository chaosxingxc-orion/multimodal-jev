# 第256步原生 loss 日志口径核验

本次只读已独立收集的真实四卡 update-0256.json，并复验原收集报告中的四份 SHA256。
核对正式 arithmetic-v2 封包内 actor/三个 objective 源码与冻结 config 身份；原生 verl
dp_actor.py 的 SHA256 等于 actor 强制校验值。未改变运行包，也未增加远程观察。
数值与来源见[核验记录](../../../context-step256-native-metric-semantics-v1.json)。

## 可直接确认的口径

`actor/pg_loss` 在本配置中是混合目标，不是纯 policy-gradient loss。
registered_loss 返回 RL + ranking + binary NLL + 自定义 reference KL，原生 actor
将整个返回值记入这个通用字段。`actor/kl_loss=0` 只对应被关闭的原生外层 KL。

每个 `jev/context_loss` 是一个 microstep 的贡献，已乘 world_size/global_real_queries。
registered_loss 返回时再乘 accumulation=4；原生 actor 在反传和累计 pg_loss 时除以4，
这两个因子抵消。因此每卡应把四个条目**相加**，再对四卡取平均。

| rank | 四个 microstep 求和 | 记录的 actor/pg_loss |
| --- | ---: | ---: |
| 0 | 0.0187364510 | 0.0187364510 |
| 1 | 0.0226099959 | 0.0226099959 |
| 2 | 0.0478958869 | 0.0478958869 |
| 3 | 0.0435428862 | 0.0435428862 |

四卡逐一精确相等；全批混合目标为 **0.0331963050**。
把16个microstep直接平均会得到0.0082990763，少算四倍。这里核对的是日志读法，
没有发现据此需要更改当前训练缩放的证据。单独rank0也不是完整global batch目标。

## 对后续优化的影响

1. 不根据 `actor/pg_loss` 的名称，把下降归因于RL增强；也不把外层KL日志0解释成无KL。
2. 中心化优势在ratio≈1时，可让RL标量接近0而梯度非零；必须结合奖励差异组比例、
   概率质量与完整选择集效果，不能用RL标量大小判断是否学习。
3. 下一版若补分项，直接保留既有objective返回的auxiliary/sample metrics；记录原始
   ranking/probability numerator、RL、KL及系数，以相同全局真实query分母汇总。
   不额外重算backbone，不把未加权分项与加权总值混用。分项和与混合目标应可复算对齐。
4. 日志改动先独立验证不会改变返回loss、梯度或更新，再进入新实验版本；当前封包不变。

这是一次实际TRAIN批次更新前的目标核验，没有恢复原始分项或共享参数梯度，
也不是完整选择集收益。此前由概率反推的分项仍保留其近似边界，不能冒称原生分项日志。
