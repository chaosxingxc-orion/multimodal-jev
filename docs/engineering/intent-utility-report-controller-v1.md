# 三臂奖励曲线报告自动收尾

2026-11-12 UTC。新增只读宿主进程PID3703812，完整argv已核实存活；等待
`intent-utility-curves12-controller-v3`（PID3689035）完成。每600秒检查精确
前驱进程，终态后重新认证所有36组实际训练/suite及correctness状态重放，
再逐步生成margin、KL、分类修复/退化和梯度/裁剪报告。

冻结协议为`intent-utility-curves12-prepared-v1/protocol.json`。报告要求三臂
各12步全部齐全；缺臂、缺检查点、来源/优化器/目标漂移或实际验收失败均阻止
发布。不会根据分数筛选checkpoint，也不会发起训练或占用NPU。
未修改原控制器、已冻结源码或FSC依赖；此报告是独立分支。

入口`tools/jev/queue_intent_utility_report.py`；新队列和完整报告相关17项
本地测试通过。真实进程当前处于等待，不代表效用训练或最终报告已完成。
源码/config/protocol及前驱launch均已固定哈希。

- [实际启动与存活凭证](intent-utility-report-controller-v1-launch.json)
- 根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-utility-report-controller-v1`
- 完成后输出：上述目录`run/report.json`与`run/terminal.json`。

输出仍是89条学习来源的完整诊断，不替代独立开发集或FSC迁移收益。
