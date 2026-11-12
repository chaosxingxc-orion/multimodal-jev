# 新目标工程中的正确性复跑验收

2026-11-12。候选utility三臂已排队，其中correctness使用新版训练/验收源码，
保持原D的优化目标和因素。为避免把工程变动误认作算法效应，新增独立CPU审计工具
`tools/jev/verify_intent_correctness_replay_cpu.py`，不修改排队源码快照。

原参考：`slurp-intent-finite-resume-step2-prepared-v2`及同名step2-suite-v2。
待复跑：`intent-utility-engineering-prepared-v1/correctness/training02`及suite02。
新旧实际配置/模板的非路径训练因素已对齐；显式correctness与历史默认目标等价。
未来额外配置字段也纳入严格比较，不只白名单抽查lr、seed。

工具要求两套完整D2接受产物、178输入训推gate、相同初始化策略身份、相同实际
来源/监督，以及逐rank相同前向候选概率。随后比较四rank FP32 master和每一个
Adam字段、BF16导出身份，要求精确相等。优化目标不同或动量/来源不同直接拒绝。
实际训练/suite Docker退出由调用方单独核验；工具本身在无Docker的CPU容器里
不能认证进程终态，不把产物通过当作实际容器成功。

固定镜像CPU控制使用已接受原D2与自身比较，并对四个真实shard分别扰动master、
一阶矩、二阶矩和step各一个FP32相邻值，确保能检出BF16导出可能掩盖的微小差异。
新correctness复跑还未执行；控制通过只能证明审计可用，不能提前宣称新旧数值相同。

即使未来精确对齐通过，也只排除这些更新中的工程差异，不证明utility优于正确性、
更长训练收敛或跨任务泛化。任何不一致应先归档定位，不通过放宽阈值隐藏。
