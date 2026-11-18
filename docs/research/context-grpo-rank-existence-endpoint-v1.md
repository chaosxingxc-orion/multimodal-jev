# GRPO 固定终点：排序与接受／拒绝的分解

完整训练／曲线复核和固定终点置信区间现已完成，见
[实测结论](context-grpo-fixed-endpoint-conclusions-v2.md)。集合 F1 改善，Top1 未见明确
改善，NLL/Brier 恶化；下文末尾“仍在运行”属于当时记录。

以既定 step0 与 step634 比较，全量 selection 9,226 录音、60 个候选，阈值保持
0.5。使用已保存预测，不增加推理，不用 TEST，不改变选模或训练权重。
机器报告：[context-grpo-rank-existence-endpoint-v1.json](context-grpo-rank-existence-endpoint-v1.json)。

| 指标 | Stage1 step0 | GRPO step634 |
| --- | ---: | ---: |
| 正例对每个负例的排序正确率（平局计 0.5） | 98.5704% | 98.3771% |
| 正例漏检率 | 25.5148% | 18.2636% |
| 负例误接受率（每个负候选） | 0.1556% | 0.2961% |
| 每 query 错误接受候选数 | 0.09181 | 0.17472 |
| 正例平均概率 | 0.69356 | 0.81894 |
| 最难负例平均概率 | 0.16242 | 0.17245 |
| 正例 NLL | 0.79522 | 1.43962 |
| 负例平均 NLL | 0.00558 | 0.02307 |
| 正例 Brier | 0.20290 | 0.17852 |
| 负例平均 Brier | 0.00129 | 0.00294 |

集合 F1 从 0.73804 提升至 0.80900 的同时，漏检减少、误接受增加；排序指标没有同步
提高。正例平均概率和 Brier 改善，但 NLL 恶化，说明平均概率提高不足以保证对数损失
改善。低概率正例会受到更强的对数惩罚；本报告没有进一步定位该误差的样本分布，不能
将其归因于某类音频、标签或训练机制。

两项总 proper score 按 `(正例分项 + 59 × 负例分项) / 60` 重建，与原曲线差值
小于 1e-12。NLL 两端均没有无限值；脚本保留概率端点，不裁剪、不删除错误样本。
pair accuracy 包含大量容易负例，因此不能代替 top1，也不能把 98% 解释为意图准确率。
“存在性”在此仅表示 benchmark event membership，不表示完整语义存在性。

结果支持现有研究方向：排序、接受／拒绝和原生概率质量分别报告，避免单个集合 reward
掩盖退化。它不证明新双信用 loss 会改善这些指标；仍须完成同起点的 joint、ranking-only、
existence-only 实验及迁移评测。不得凭这组选择集结果临时调整阈值或宣布复杂任务准入。

复用入口：`tools.reports.context_rank_existence_diagnostics`；共享配置为
`configs/portable/operations/context-rank-existence-diagnostics.json`，所有地址与身份由本地宏
集中绑定。报告记录 curve、数据清单及预测文件哈希，核验完整人口、两端实际输入一致性，
独立重建 NLL/Brier。14 项测试覆盖阈值、平局、单调变换、端点、分项加权及人口冲突。
这次诊断不是完整 runtime 重审；独立的完整训练／11 点曲线／来源组 bootstrap 任务
仍在 `/srv/multimodal-jev-xingchao/reports/grpo-probability-endpoint-v2` 运行，区间待其终态。
