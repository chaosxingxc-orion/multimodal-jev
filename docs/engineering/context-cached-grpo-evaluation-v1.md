# 缓存 GRPO 的完整源评测与迁移角色

本入口只处理简单SLURP任务。执行完整634步训练的独立审核后，评测预登记的
0、64、128、192、256、320、384、448、512、576、634，共11点。
每点均为完整9,226条selection；训练reward、prefix和末步分数不能替代完整曲线。

## 来源与模型接口

`context_cached_grpo_source_plan`读取绑定SHA256的真实training audit配置，重新调用
`context_cached_formal.audit`核验容器/传输/源码/全部更新/固定reference，再逐点绑定策略。
step0必须为实际Stage1第318步终点，不能换成selection最优点或效率实验终点。
后续各点核对checkpoint提交、每步policy提交、完整训练contract、两rank optimizer分片声明、
LoRA/head策略身份和原生adapter。只绑定已完成产物，不生成分数或按目标集选择模型。

`prepare_context_cached_grpo_selection`复用已封存replica评测器的封包实现与客户端，
仅切换本轮GRPO来源校验和预登记634步检查点规则；模型source manifest保持原样。
`context_cached_grpo_selection_operations`复用四卡租约、容器wait/collect、完整输入重建、
每策略HF256面板exact检查、macro日志及native scorer。音频直入，service32、rollout32、
macro128，模型计算与Stage1评测一致；不存在转写后调用文本模型的分支。

## 固定脚本与配置

后台入口`tools.reports.context_cached_grpo_evaluation_sequence`，模板
`configs/portable/operations/context-cached-grpo-evaluation-sequence.json`读取
`${CACHED_GRPO_EVALUATION_SEQUENCE_CONFIG}`。声明以下字段：

- `predecessor`：正式GRPO后继driver/config/terminal准确路径及driver/config SHA256。
- `points`：11个预登记step及各自package/remote_root/operations_root/tag。
- `output`与`selection_dispatch_allowed=true`。

父runtime、release证据挂载、输入缓存、host/lease从正式训练及其Stage1评测配置继承。
禁止覆盖任一上游正式训练、效率、reference、有限验收或评测目录。
通过pidfd等待前驱，不新增轮询；失败保留句柄并停止，不自动重试或调用模型调度。
每点写不可变完成记录，完整曲线产生前不发布选中策略。

独立入口及模板分别为：

| 功能 | Python模块 | 模板宏 |
| --- | --- | --- |
| 来源计划 | `context_cached_grpo_source_plan` | `CACHED_GRPO_SOURCE_PLAN_CONFIG` |
| 评测封包 | `prepare_context_cached_grpo_selection` | `PREPARE_CACHED_GRPO_SELECTION_CONFIG` |
| 单点评测/审核 | `context_cached_grpo_selection_operations` | `CACHED_GRPO_SELECTION_OPERATIONS_CONFIG` |
| 完整曲线 | `collect_context_cached_grpo_curve` | `CACHED_GRPO_CURVE_CONFIG` |

这些模板均位于`configs/portable/operations`，本地绑定放`.local/config`。
完整曲线collector重用replica曲线的跨点一致性检查与既有迁移发布格式，
报告set-F1@0.5、top1、macro-F1、NLL、Brier及集合指标、全部点与相邻退化。
选模规则保持set-F1最高、NLL最低、最早步。输出`source-evaluation.json`冻结：
`common_Stage1`、`source_selected_new_GRPO`、`fixed_endpoint_new_GRPO`。
现有`evaluate_context_transfer_macro.source_binding`可重算这些角色；
角色接口可消费不代表目标域推理协议或目标宽度数值验收已通过。

## 证据边界与队列

来源绑定/新封包/单点/完整曲线及序列47项CPU检查通过；与共享封包、point审核和曲线
合并回归70项通过。测试覆盖实际生成封包与源码不变、真实commit布局、错误策略/来源/
reference/人口/传输拒绝、完整顺序和失败停止、发布结果被迁移角色接口重算。
完整模型和NPU运行在这些测试中为显式替身，不作为本轮实际效果证据。

实际本地宏`.local/config/context-cached-grpo-evaluation-sequence-v0.0.42.json`，
等待`context-cached-formal-successor-v0.0.41`；结果根为
`/srv/multimodal-jev-xingchao/operations/context-cached-grpo-evaluation-v0.0.42/`。
后台驱动PID2324817（启动记录1794953434.7692635）已在启动后现场核验存活，
等待准确正式训练前驱；终态为上述结果根下terminal.json。
这是受控队列登记，不代表正式训练或评测已执行。
本入口不派发TEST/FSC/MINDS，不训练、不调整阈值或算法参数。
本轮GRPO真实曲线、TEST及跨数据集迁移成绩仍待产出；复杂任务全部延期。
