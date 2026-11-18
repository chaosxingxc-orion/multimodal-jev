# 完整简单曲线后的自动本地分析

后台PID969099：`python -m tools.reports.run_context_endpoint_followup --spec .local/decision-objective-research/context-endpoint-followup-v1/spec.json`。
当前实际状态waiting_original_experiment/grpo_v1，尚无完整RL报告。

依次等待原GRPO、原RLOO实验owner释放，确认完整训练及11点selection成功。
只读打开既有owner.lock，不创建缺失前置锁；采用非阻塞共享锁做本地检查，最多每60秒一次。
这不是远程训练观察，不缩短15分钟remote节流。状态失败则保留原句柄和证据并结束，
没有重启/推理/训练/收集/设备租约能力。

释放前置锁后才运行已有完整训练和曲线独立审核，以及固定step634-step0来源组配对分析。
输出各算法的决策区间、NLL/Brier区间到：

- docs/research/context-simple-grpo-probability-endpoint-v1.json
- docs/research/context-simple-rloo-probability-endpoint-v1.json

完整曲线审核会重新检查实际产物，不能仅靠status成功字段生成报告。分析前后校验1181项
本地依赖；既有输出拒绝覆盖。所有报告complex_training_admitted=false，科研收益解释和
书面复杂训练准入仍需依据实际结果，不自动做阈值决策。

6项本地流程检查通过：缺owner不创建、持锁不提前执行、失败终态拒绝、分析期间已释放
前置锁、已有输出不覆盖、依赖变化及计算失败不发布成功。真实计算函数由此前完整Stage1
自身配对和真实未完成GRPO拒绝验证覆盖；这里的成功流程测试使用替身，不称已完成RL审核。

已通过/proc核对后台实际存活，原GRPO/RLOO owner及两observer也存活；六组原运行/排队
依赖保持。未改变原GRPO→完整评测→RLOO→完整评测→FSC→TEST→MINDS顺序。
本进程在原owner释放后只读本地产物，不持前置锁等待后继，所以不阻塞RLOO派发。

状态/日志/启动收据/spec位于`.local/decision-objective-research/context-endpoint-followup-v1/`。
证据见同目录文档前缀的JSON。未来如需改其固定依赖，应先处理这个独立等待进程的版本，
不得默默修改后继续宣称同一分析身份；当前训练与原队列的依赖约束仍单独保持。
