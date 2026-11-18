# 将分类曲线与更新稳定性放在同一报告中

2026-11-12 UTC。补齐已登记效用研究中的margin、更新后KL和裁剪诊断读取，
复用实际训练与独立suite产物，不增加NPU预算，不修改冻结控制器或实验快照。

`tools/jev/report_intent_update_metrics.py`先调用既有独立比较入口，确认实际
训练/suite容器退出、状态恢复、BF16导出及完整后端概率验收，之后读取完整
Final候选分布。输出全学习人口、实际更新批次、其余学习来源三个组：

- gold相对最强错误候选的log/probability margin；不是top1相对top2置信差。
- 更新前后正确数、错误修复和正确退化；不以净正确数掩盖两者。
- 更新后的双向KL、TV、ratio越界概率质量及优势方向上受裁剪的概率质量。
- 原类别权重下的概率变化与KL，以及未加权值。
- 四rank训练日志中的单次全局梯度范数、norm=1裁剪触发情况及微批PPO裁剪率。

更新后概率质量不是训练日志的clipfrac；梯度裁剪不是策略ratio裁剪。
空的“其余学习来源”组明确返回零样本和null指标。其余学习来源仍不属于开发集。
所有效用臂均按同一正确性及概率诊断口径报告，不比较不同目标的原始reward尺度。
本入口不输出参数位移，现有Adam方向工具承担该独立分析，不能把梯度norm当位移。

`tools/jev/report_intent_utility_curves.py`要求原登记的三臂各12步全部齐全，
在打分前核对共同初始化、非目标因素、源码、连续状态、实际来源日程；逐步
调用上述独立入口，保留每步before/after值和累计来源使用次数。不选最好一步。
它是手动/后续编排可调用的只读报告入口，尚未加入冻结队列；全量实际运行待36组
训练与suite全部完成。它补充现有分类曲线，不替代原accuracy、macro-F1与NLL报告。

后续已启动[独立报告等待进程](intent-utility-report-controller-v1.md)，在原三臂
控制器完成后自动执行本入口。没有修改冻结训练队列，实际报告仍待前驱完成。

真实Q7已执行成功，89条平均gold log margin −3.866222→−3.919944；正确数
2/89→2/89，修复/退化均0。平均KL(old||new)=0.006748724，旧质量越界比例
7.44994%；训练微批clipfrac仍全0。全局梯度norm=0.27534023，未触发norm裁剪。
新报告的所有重叠分组KL、概率漂移指标与先前独立Q7诊断一致。
这些都是学习人口证据，尚未支持分类提升或泛化结论。

本地43项相关测试通过，包含错样本修复/对样本退化抵消、错误gold margin、
空补集、不同权重、损坏概率、rank分歧、不完整曲线、更换目标/优化器/源码、
不连续状态、来源日程变更及实际验收失败等负对照。真实效用协议的预检正确
拒绝缺失的首步launch，没有把prepared状态当作完成。
本地测试没有执行新的张量/NPU路径。

[验证记录](../../../intent-update-metrics-v1-verified.json)保存源码、真实Q7报告和旧诊断身份。
全量报告调用：

```bash
.venv/bin/python tools/jev/report_intent_utility_curves.py \
  --protocol /mnt/cephfs/xingchao/experiments/multimodal-jev/intent-utility-curves12-prepared-v1/protocol.json \
  --protocol-sha256 <预先固定的protocol文件SHA256> \
  --output <新的报告文件>
```
