# 正式RL前缀吞吐与下一版本优先级

日期：2026-11-15。当前任务/配置入口为 [context-prob-v1-baseline.md](../../../context-prob-v1-baseline.md)。
本分析不改变训练代码、reward、batch、运行包或条件队列。

## 证据范围

输入为 `context-arithmetic-formal-live-v6.json` 中已完成步骤的rank0精简计时，
每条保留远程原文件路径和hash；汇总为 `context-formal-throughput-prefix-v1.json`，
source SHA256 `89b6bae59518041cda38fce27f14390314d9ae3459be7f7503ffee7ac121d524`。
排除首次启动步，统计step2–29共28步、3584条真实query。
这是运行中前缀观察，不是全程稳态、逐rank最终审核或完整模型收益证据。

## 时间分布

| 项目 | 平均秒/更新 | 说明 |
| --- | ---: | --- |
| 总耗时 | 100.427 | global128，约1.275 query/s |
| 准备阶段 | 73.330 | 包含rollout及组装/分发 |
| rollout | 65.675 | 准备阶段子项，不与准备阶段重复相加 |
| 固定reference | 32.610 | rollout子项，当前首次覆盖全部miss |
| rollout其余 | 33.064 | 包含behavior、切换、绑定等，不能当纯forward耗时 |
| 准备阶段其余 | 7.655 | 差额，不进一步冒称已定位单项原因 |
| actor更新 | 23.953 | 约总耗时23.85% |
| actor之后 | 3.143 | 总量减准备和actor |

音频/processor缓存3584/3584命中，前台等待均值约0.000090秒，输入绑定约0.268秒。
后台准备约1.619秒，与前台重叠，不能再次累加到更新总时间。
固定reference缓存3584/3584未命中，每步观察到两次策略切换；第二轮是否加速尚未观察。
rank0 actor峰值allocated约37.11GiB，不包含其他进程及设备总占用，不能证明micro32能容纳。

## 对应的真实调用路径

已封存的 `context_rollout_session.py` 逐条填充reference缓存，再切回当前策略。
`context_online_batches.py` 的prepare逐条调用current_engine.score。
`context_rollout_rpc.py` 每query同步提交、等待和验证回复，服务单owner串行处理。
rank0完成准备后才分发给四rank actor；DP4不会自动并行该rollout服务。
G16由一次概率输出采样16个完整动作，并不是同音频做16次模型前向。

固定reference缓存键绑定固定策略、实际编码输入和读出位置，不含epoch；
同策略、输入和数值协议满足身份验证时才可能跨轮/跨算法复用。现有命中结果
不能提前证明后续轮或RLOO已经使用了可复用缓存。

## 下一版本的有限优化顺序

1. 批量提交rollout：保留逐query输入、位置、概率及行为身份；先验证批量前向
   与当前逐条路径的数值和采样一致性，再测端到端吞吐。批量API本身不保证模型批量加速。
2. 固定reference复用：验证同reference、输入张量、位置、processor及数值协议的缓存身份，
   统计首次建设成本与之后命中成本；跨算法共享应单独报告计算开销，不隐藏预计算。
3. actor micro32：在同global128下，DP4意味着accumulation1；容量与训推数值一致性
   先用有限探针验证，作为新实验配置，不拼接到现有曲线。

保持其他耗时不变，仅将actor时间减半，总体理论加速为1.1354倍；
理想化完全去掉reference计时，总体为1.4809倍。二者是算术反事实，不是已测收益，
reference首次计算仍有真实成本。当前证据不足以把主要瓶颈归因于fsync或Ceph，
保留本地fsync与热数据写/data的存储规约。

当前正式GRPO、RLOO对照及下游评测继续使用封存版本。效率候选不成为简单任务
完成训练和评测的额外前置条件。
