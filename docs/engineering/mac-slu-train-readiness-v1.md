# MAC-SLU 跨任务扩展前的训练数据与评分接口审计

2026-11-12，仅打开本地 TRAIN 标签和 TRAIN 音频归档，未读取 dev/test 标签，
未运行模型、训练或挑选外部测试样例。该工作为后续 SLURP-only 策略的中文多意图
迁移准备接口，不将已接触的训练数据或开发任务称为完全未见基准。

## 本地数据的实际状态

完整流式读取 `audio_train.tar.gz` 的归档成员，并与 `label/train_set.jsonl` 对齐：

| 项目 | 实测数量 |
| --- | ---: |
| TRAIN 标注记录 | 18,000 |
| 唯一音频成员 | 17,999 |
| 缺失音频 | ID `4207` |
| `semantics` 为空的记录 | 1,663 |
| 含空意图实例的记录 | 3,701 |
| 同一 domain/intent 重复出现的记录 | 1,831 |
| `split_sens` 数与标注意图实例数不同的记录 | 2,179 |
| 无空标注容器的记录 | 12,636 |
| 明确标注的 domain/intent 类型 | 67 |
| 明确标注的 domain/intent/slot-role 关系类型 | 489 |

各问题类别可能重叠，不能相加作为排除总数。`4207` 同时属于空 semantics；
12,636 条无空容器记录都有对应音频。没有完全相同 query 字符串的重复组，
但本次没有语义近重复、波形相似度或说话人隔离证明。

此前指南的 17,997 是引用的论文数量，本地发布实际有 18,000 条，不擅自删除三条
使数量吻合。完整证据及逐 ID 清单保存在共享 experiments 下，索引和 SHA 见
`mac-slu-train-readiness-v1.json`。本次只验证归档成员与文件身份，没有解码每条波形
或完成听审；12,636 是结构上无空容器的数量，不是已通过语义清洗的可训练数量。

上游提示要求无法匹配领域/意图时输出空列表，但这不能单独解释本地每个空
semantics 或空意图实例的来源。暂将其标记为待解释，不作为“全 false”的奖励。
有明确标签的正项可保留为部分监督；未经完备性确认，不为其余命题补造负项。
有空标注不等于已证明数据错误，也不直接从官方评测分母删除。

## 固定官方源码的实际评分行为

读取并固定 `Gatsby-web/MAC_SLU@d54453f63e0dc57b17d3cb2cfea35fb7da6a4489`
的 `metrics.py` 与 `slu_icl.py`。官方完整代码只存共享 experiments，不复制进项目。
评分器 SHA256 为 `821b6389b618e474a915cf6f63f61defc868e2fec8d3df545e5ed77dc5b0f149`。
`audit_mac_slu_scorer.py` 实际执行固定评分器的四个合成反例，均复现：

1. 将原始嵌套字典 gold 直接交给只接受 list 的评分器，会将 gold 归一化为 `[]`；
   空预测的 Overall、Intent、Slot F1 均为 1。必须显式转换并校验，禁止直接评分。
2. 交换两个正确语义帧的顺序，Overall 为 0、Intent 为 1、Slot F1 为 1。
3. 将两个意图各自的槽值对调，Overall 为 0，但 Intent 和 Slot F1 仍为 1；
   官方 Slot F1 的集合键仅为 `(slot_name, value)`，不保留 domain/intent 绑定。
4. gold/prediction 的 ID 不同但按行内容相同，仍得到满分；入口需自行校验 ID 对齐。

这些是评分行为的构造样例，不是模型分数。正式保留官方规范化/顺序语义的列，
另外报告绑定关系指标；不能把自定义严格指标称为原论文 Overall/slot F1。
固定来源、四项输出和测试结果见 `../research/mac-slu-official-scorer-fixtures-v1.json`。

仅用 AST 读取上游提示常量，得到 84 个 domain/intent 及 193 个 domain/role。
TRAIN 中的地图/查询途经点、收音机/播放列表、收音机/播放历史三项不在官方
提示菜单，分别出现 1/1/2 次；所有明确标注 role 都在菜单中。不能对这些样本
按 gold 单独添加候选；后续若采用扩展菜单，应独立固定“官方菜单+TRAIN-only
补充”的项目协议，不冒称原版候选条件。

## 对一次填充 Jev 的实际约束

67 种已出现意图及官方 84 项菜单都是多标签任务，不能直接改成单项 Choice；
当前原生 symbol readout 每个 Span 最多 62 个值，也不支持原样装入这两个菜单。
应按事先固定的定义拆为 Boolean 命题，并保留多意图的 domain/intent/slot-role
关系。1,831 条重复意图记录进一步表明，仅用意图集合不能恢复调用次数或实例值绑定。

下一项实现应是严格原始标注转换器和固定任务定义，保留空标注的 unavailable
状态、重复实例、原始分母和独立 input/target 文件；随后完成 TRAIN 音频解码及
少量无 gold 输入的 processor/Span 检查。正式跨任务评估使用固定 SLURP 末次策略，
不得根据 MAC-SLU 外部成绩回选 SLURP checkpoint。当前尚未发布 MAC-SLU Span 数据，
不从这份结构审计推断跨任务收益。
