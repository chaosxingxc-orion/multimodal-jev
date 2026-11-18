# SLURP意图字段与60类映射核验

2026-11-12。对official-clean-intent-v1全部40,513 learning录音及其9,192来源进行
原始TRAIN、clean-v7、source targets、发布标签四层核对。没有读selection/test样本
内容，不执行模型。此前全clean TRAIN有91个原始intent字符串的统计与本次learning
子集88字符串不同，均不是91/88个独立语义类别。

## 完整学习人口的结论

- 规范类为60个scenario/action组合；原始intent字段有88字符串。
- 136个来源、873条录音的原始intent只写action，共28种省略scenario的别名。
- 所有这类差异均满足raw_intent==action，没有发现第三种字段不一致。
- `query`对应9个规范类，`remove`对应2个；不能仅按raw intent建立一对一映射。
- 每条发布录音及其所有来源的scenario/action均与发布的规范标签一致。
- 全部9,192学习来源的完整JSON记录，在原始TRAIN与clean-v7间相同。

因此当前发布与`slurp_official_training`使用的组合映射是正确的，不需改成88类，
不需要修复现有训练标签。以后scorer、定义关联、数据适配统一按冻结candidate_pairs
查scenario/action，raw intent只保留溯源。原始字段值不同不自动算标注错误。

## 对语义任务冻结的影响

排除了“clean-v7修改学习内容”与“本项目raw字段映射错误”作为已发现语义问题的原因。
cooking_query/recipe的定义重叠、transport query/ticket等边界仍须在独立语义层解决。
不能把raw命名修正当作定义正确性认证，也不能因原始标签存在就把其他类全设为语义负例。
完整记录一致只是数据一致性结论，未听音频、未证明原作者类别优先级。

这完成了原生标签身份的全学习人口检查；不表示新Jev监督或正式TEST协议已经冻结。
原生60类复现基准仍保留，后续语义派生任务若改变真值政策必须单独版本化及计分。

## 证据与复核

工具`tools/jev/audit_slurp_intent_namespace.py`核验已固定manifest、learning、tasks、
source targets、clean和original metadata hash；逐录音source-target hash一致。
共享`slurp-jev-benchmark/intent-namespace-audit-v2/report.json`保留完整交叉表和136条
差异来源的录音ID。v1保留历史；v2与当前整理后的工具hash匹配，数据结论相同。

[独立复核](intent-namespace-independent-check-v2.json)从原始文件重新计算交叉表、
136来源/873录音集合及全部映射，不复用工具resolve函数。3项测试验证同action跨scenario
不能混类、raw字段不能覆盖pair、候选映射歧义/本体外pair必须拒绝。
任务v2已登记唯一原生标签键与审计证据；没有更改actor、模型、数据gold或启动训练。
