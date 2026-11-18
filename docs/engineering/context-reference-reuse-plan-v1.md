# 固定 reference 的全量复用规模

2026-11-16 全量归档核验补充：固定脚本
`tools.reports.audit_context_reference_inventory` 已逐条核验 GRPO 的 40,513 条 reference
缓存，文件哈希匹配原收集清单，缓存中的输入张量及 marker 位置与独立输入审计完全对应，
无缺失或额外输入。每条 60 个概率，精确等于 0 或 1 的数量为 0。JSON 文件实际共
649,024,006 字节，明显高于纯 FP32 payload 的约 9.72 MB；这说明存储格式有压缩空间，
尚不能证明 I/O 是训练瓶颈，也不能把概率未达端点当成新策略 raw-logit 饱和检查。
报告：[context-grpo-reference-inventory-v1.json](context-grpo-reference-inventory-v1.json)。

入口使用 `configs/portable/operations/context-reference-inventory.json`，机器路径只由本地
宏绑定。7 项记录校验测试通过，真实全量检查退出 0。首次运行因归档入口软链接被拒绝，
本地宏改为真实归档目录后通过，校验规则未放宽。没有复制缓存、改训练配置或跳过前向。
旧缓存仍缺少新版执行身份与实际数值等价证据，因此当前 `cross_runtime_reuse_allowed=false`。
新版批量推理必须先验证等价性；否则应由新版冻结 reference 重新建库，仅在同一执行身份
的后续实验间复用。

后续实测补充（完整 GRPO 634 步已归档）：第一轮 40,513 次 reference miss，第二轮
为 0；两轮平均完整步时分别 101.137 和 67.708 秒，reference 平均耗时分别 32.790
秒和约 0 秒。来源：[完整阶段耗时](context-full-rollout-efficiency-v1.json)。因此下文
“实际运行尚未到第二轮”仅描述当时状态，现已有跨轮缓存命中的完整记录。
这是同一旧版本不同缓存条件下的观察，不能据此宣称新版提速；两个轮次的策略状态和
样本顺序也不同。新版本对照必须匹配缓存冷热条件，并计入建库与校验成本。

2026-11-15：读取封存GRPO/RLOO配置、两轮完整日程、40513条独立输入缓存收据后核算。
本分析不修改运行包、排队包、reference缓存或优化器，不代表已经完成吞吐优化。
完整机器结果：[context-reference-full-schedule-reuse-v1.json](context-reference-full-schedule-reuse-v1.json)。

## 已核实的复用边界

GRPO/RLOO使用同一个Stage1终点、base、contract、输入缓存及日程，但reference目录不同。
每轮317更新、40513真实录音、63个零权重补齐slot。全部输入张量（含位置）的指纹共40513个，
没有可据此进一步合并的不同录音。缓存绑定不含epoch和算法名称。

| 场景 | 新reference前向数 | 解释 |
| --- | ---: | --- |
| 单算法首轮，冷缓存 | 40513 | 63补齐输入此前已出现，不新增reference |
| 同算法第二轮 | 0 | 理论命中全部40576 slots；实际运行尚未到第二轮 |
| 当前两算法分别冷启动 | 81026 | 两个独立缓存目录 |
| 下一版共享同一已验证reference | 40513 | 只建一次，两算法共用；尚未实施 |

这不是把两轮都算作新reference开销。当前代码已经有轮内/跨轮缓存，优化增量主要在跨实验复用。
当前behavior仍每批重新前向；不能复用旧behavior或把优化器状态共享给RLOO。
G16也不是16次音频前向，采样动作不能作为cache内容。

40513×60个FP32概率的纯payload约9.72MB，实际JSON重复contract/hash及文件开销更大，
该值不是实际磁盘空间估计。当前实现每次session.prepare仍读取/绑定缓存，不能称命中成本为零。

## 时间估算与实验要求

正式step2–29观察到reference平均32.6104秒/128条。仅按此前缀速率外推，省40513次前向
约10321秒（2.87小时）；其中含切换/写缓存，不能当纯forward时间，更不能称实测总任务加速。
缓存搬运、验证、读取、首次建库成本仍需计入。实际第二轮命中与RLOO冷启动行为应由后续日志确认。

下一版本若实施共享，需封存reference策略、实际输入/位置、数值协议和引擎来源身份，
从独立审核产物发布本地不可变缓存，复用前逐条校验。当前namespace绑定policy/base/contract，
没有额外完整runtime源码digest，因此不能仅依据目录名或policy ID无条件跨runtime复用。
验证共享前后概率/采样/更新及恢复一致后，再测完整端到端吞吐。
这属于新效率配置；不原位改当前GRPO/RLOO目录，也不把预计算成本隐藏在算法对比之外。

工具 `tools/reports/analyze_context_reference_reuse.py` 已完成全量核算；2项测试覆盖
补齐重复、跨轮命中、同张量异录音和异context不合并。另以独立集合计算复核全部81152 slots一致。
原训练1078、TEST1081及MINDS队列1097依赖均保持不变。

最新普通观察live-v12四rank67/634 running/非OOM。该观察的commit列表使用了错误的
`commit-*.json`模式，不能用于判断checkpoint缺失；生产实际文件名是`checkpoint-*-committed.json`。
本次不提前轮询或重启，后续到期观察使用正确名称，最终checkpoint有效性仍由正式训练审核器证明。
