# 后续过程奖励与优化控制的静态依赖复核

2026-11-12 UTC。对当前真实控制器引用的P12、lower-LR D12和三臂utility
各12步计划逐项重新检查，共60组训练/suite、120个容器计划，全部通过。
检查直接读取共享存储，不依赖旧prepared验收标记。

每个计划检查config-template及source-manifest哈希、全部已知pins、源码快照，
再按Docker实际bind的最长匹配路径解析容器内文件，确认映射到宿主文件的
内容哈希一致。额外重新验证lower-LR控制器的12步预算、原初始化与连续状态配置。
未改配置、源码快照、训练顺序或预算，也没有创建新训练容器。

| 组 | 训练计划 | suite计划 | 本次结果 |
|---|---:|---:|---|
| P过程奖励 | 12 | 12 | 已知依赖通过 |
| lower-LR D | 12 | 12 | 已知依赖通过 |
| correctness | 12 | 12 | 已知依赖通过 |
| smoothed-utility | 12 | 12 | 已知依赖通过 |
| uniform-credit | 12 | 12 | 已知依赖通过 |

这不是NPU可运行性、动态checkpoint可用性或学习效果验收。后续新导出的
adapter、FP32 master/Adam及完整概率必须由原控制器在实际前驱完成后
重新认证、固定身份并验证容器挂载；不能提前生成成功证据。

[复核凭证](intent-p-lr-utility-preflight-v1-verified.json)固定配置、实现与完整
报告身份。详细120项结果在共享实验目录`intent-p-lr-utility-preflight-v1.json`。
本次未遇到需要修复的静态依赖，原队列继续执行；最近普通运行观察仍为13:12 UTC。
