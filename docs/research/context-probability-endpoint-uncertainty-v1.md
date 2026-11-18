# 简单任务固定终点概率质量配对区间

最新终态：真实 GRPO 复核 v2 已成功退出，完整报告与结论见
[固定终点实测结论](context-grpo-fixed-endpoint-conclusions-v2.md)。集合 F1 的差值区间
为 [+0.059904,+0.082831]，Top1 区间跨零；NLL/Brier 差值区间均在恶化方向。
下文“正在运行”“尚无真实成功路径”等表述保留为历史状态，不能覆盖此次已完成结果。

2026-11-16 执行补充：GRPO 的完整 634 步及 11 点曲线已完成，当前已启动真实固定
终点报告复核。新入口 `tools.reports.run_probability_endpoint_report` 复用本文件下述
原分析函数，配置 `configs/portable/operations/context-probability-endpoint-report.json`
从本地宏读取实验、数据和输出路径。入口使用原后继队列相同的 scoped Python3.12
求和兼容层，不改训练、评分器或原预测。相关原统计测试 13 项通过。

本次独立输出目录 `/srv/multimodal-jev-xingchao/reports/grpo-probability-endpoint-v2`，
`attempt.json`/`process.json` 保存配置、实现及准确进程身份；成功应产生 `report.json`
和 `COMPLETE.json`，失败产生 `FAILED.json`。未写入原后继预定结果路径，避免与
现有队列竞争。首次 v1 漏接求和兼容层，已主动中断并保留失败现场；v2 是否成功
必须查看实际终态，不能据这段记录假定已产出置信区间。

当前阶段TRAIN多次出现集合F1与NLL/Brier方向不一致。为正式完整selection提供概率质量
不确定性分析，新增独立只读工具`tools/reports/context_probability_endpoint_uncertainty.py`。
不改现有评分、11点评测、选模、训练损失或复杂准入，不使用TEST。

复用既有完整训练和11点曲线审核入口，仅对固定step634减step0计算区间；同时保留原有
集合F1/精确集合/top1配对分析。每query按60候选平均binary NLL及Brier，按来源组整体有放回
抽样，保持录音权重。使用与原决策区间相同的PCG64 seed317、2000次draw及线性分位数，
实际核对draw SHA。正差表示恶化，负差表示改善。

任一侧发生错误的端点概率（gold概率0或负例概率1），对应NLL无限：保留所有query、
报告两端无限query数量，将NLL差值/区间标为不可定义，绝不裁剪概率或删除错误样本。
Brier仍在完整人口上给出有限区间。正确的0/1概率不被误判为无限NLL。

这只是条件于固定模型输出的来源组采样区间，不是训练随机种子不确定性、联合置信覆盖、
选中checkpoint的无偏区间或显著性检验。不能仅凭区间通过自动放行复杂训练。

## 已执行验证

13项本地检查通过，无跳过：包含恒定差值解析解、不等来源组大小、排列不变的差值/抽样、
端点概率处理、同一输出零差控制，以及既有配对/输入/角色约束。最初测试将既有评分器
在不同输入顺序下约2.8e-17的点值加法误差要求为逐位相同；修正测试为1e-14容限，
配对差值、区间和draw身份仍要求精确相同，未改评分器。

实际Stage1完整9226条/2146来源组自身配对，3.52秒完成；NLL/Brier与原审核点值一致，
两项delta及区间均严格零。当前未完成GRPO输入被完整曲线审核拒绝。
证据：engineering/context-probability-endpoint-population-control-v1.json。
没有真实完成的RL成功路径或RL概率改善结论，完整结果仍待原队列。

## 完整曲线完成后的入口

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:tests:. python -m tools.reports.context_probability_endpoint_uncertainty \
  --directory .local/decision-objective-research/context-rl-grpo-arithmetic-v2-experiment \
  --algorithm grpo_v1 \
  --release /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v1 \
  --output docs/research/context-simple-grpo-probability-endpoint-v1.json
```

首次调用必须等待完整11点审核成功。该工具未加入自动队列，也不启动推理。
