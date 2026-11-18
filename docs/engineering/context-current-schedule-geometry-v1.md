# 当前全量日程的microbatch几何核算

基于当前context-full-learning-schedule-v1两个完整epoch及实际processor张量收据，
重新验证manifest、两epoch文件、40513唯一输入和每batch四rank完整索引。
共81026次真实曝光；尾部零权重padding仍进入执行形状统计。未读取旧Span任务的长度结果代替当前输入。

保持每个global128人口、每卡32条顺序不变，将相邻micro8组合并为16/32，得到：

| micro / accumulation | 两轮四卡合计调用组数 | token填充比例 | processor特征填充比例 | 最大单micro token槽位 |
| --- | ---: | ---: | ---: | ---: |
| 8 / 4（当前） | 10144 | 0.1252% | 7.4614% | 24832 |
| 16 / 2 | 5072 | 0.4696% | 24.4639% | 49664 |
| 32 / 1 | 2536 | 1.1565% | 43.1330% | 99328 |

这里填充比例是`1-原始张量长度之和/合批后长度槽位之和`，特征原始长度本身已含processor
padding。它不等于有效声学帧或原始waveform填充。token槽位也不等于attention FLOPs；
当前融合attention及逐请求head可以使部分算子不计算尾部padding。

结论：当前micro8已充分压低token填充。增大micro可减少调用次数，但直接合并会明显增加
声学特征填充，单次token输入规模上升；不能从参数量推断容量或速度。新分组策略可能改善
这些数字，但将是另一种调度，不能与此处保持rank顺序的核算混淆。

这只是全量形状算术，不是NPU容量、memory峰值、梯度归约或数值一致性验收。正式配置不变。
后续优化顺序仍先批量rollout与固定reference复用，再用同global128的有限micro16/32探针
验证峰值、端到端耗时、实际训推和恢复一致性，不能直接升级为micro32。

输入来源hash、精确计数及限制见同名JSON；执行脚本为
.local/decision-objective-research/analyze_current_schedule_geometry_v1.py。
首次直接JSONL读取遇已有Ascend尾部诊断，随后复用原严格parse_stream解析器识别该尾行；
未忽略任意非法行、未改原收据。计算完成，但没有新NPU任务或训练/评测结果。
