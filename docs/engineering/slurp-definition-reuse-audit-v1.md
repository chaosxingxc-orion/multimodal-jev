# SLURP 定义复用范围核查 v1

2026-11-11。只读参考Omni已发布材料，未修改其数据或知识，不读取本项目TEST来构造训练定义。

已检查 `repro/publication/wiki/S-Task-and-Knowledge-Design.md` 及发布工程的
`knowledge/slurp/semantic/slot-audit-v1/{release,snapshot}.json`。
前者给出关键任务边界：实体包括内容/背景/修饰语，不只限于操作必要参数；全局类型词表、意图条件知识、当前query存在性是三个层次。
因此不得依据旧intent_bindings硬裁剪类型、将TRAIN未出现组合作为非法组合，或将未标注类型一律视为可靠负例。
文档含dev/test诊断示例，这些示例不能复制进本项目思考要求、few-shot或训练规则；这里只沿用任务边界及禁止泄漏原则。

历史slot-audit-v1声明构造来源为原TRAIN的9,354个utterances、排除2,160个，使用其自身旧分区，非本项目clean-v7/source80。
卡片中的类型描述还可能是基于值词频的统计归纳，不等于官方语义角色定义；包含值格式约定的卡也不能直接充当存在性判据。
没有逐ID证明其构造来源只属于本项目learning之前，不直接复用其例子、统计或教师产物，否则会污染selection。
这些材料可作为审核线索，不能因为标记TRAIN就当作当前学习集授权。

下一步定义产物需逐类型保存全局角色定义、成立证据、近邻边界、负例成立条件、来源和review状态；学习例子只能从已验source-bundles的learning来源取得。
配置化思考要求引用这些可审查的条件，不编造缺参/取消/紧急度标签。规范不完备与标注冲突保留mask，不因引入思考配置解除质量限制。
本次没有发布新的类型金标或可训练release。
