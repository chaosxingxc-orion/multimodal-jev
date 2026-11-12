# SLURP Jev 全本体与评分接口

2026-11-12 UTC。四卡等待期间完成正式任务的标签映射和缺标评分接口；不把工程三布尔准确率当作SLURP基座指标。

实际产物：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/ontology-audit-v2`。`ontology.json`固定60个意图、55个业务类型和omni的10个Fold。`audit.json`记录源文件hash及40,713条学习集录音的标签支持统计。另从clean-v7原始TRAIN metadata直接重算，意图/类型支持与source bundle统计逐项相同；独立回执见同名JSON。

意图必须用 `(scenario, action)` 组合确定：91种raw intent写法对应60个规范类别。`query`对应9种scenario，`remove`对应2种，禁止只按这两个原始字符串推断类别。60个规范ID与omni历史意图schema精确相等；55个类型与S1 Fold精确相等，无需隐式重命名或丢类。旧schema仅用于ID核对，其未清洗示例及槽位值没有进入监督。

学习集的类型来源极不均衡：game_type和music_album各只有1个source_group，audiobook_author和sport_type各4个，alarm_type和transport_descriptor各6个。cooking_query仅4条学习录音。后续均衡采样必须按来源去重记录曝光量，不能把同一来源重复采样当作新增泛化证据；个别类别无法支持来源内训练/验证双侧划分，需要如实报告支持数。

`slurp_jev_benchmark.score`提供规范化评分：

- 每个gold样本显式给出全部55类型的bool/None；None仅为缺标监督mask，不是模型输出状态。intent可为规范ID或None。
- 模型结果为 `intent_ranking`（规范ID有序列表）和 `types`（类型集合列表）。两任务协议分别验收；一个任务输出失败不抹掉另一个任务的有效结果。
- 缺失模型结果留在分母；重复/陌生样本ID拒绝整个结果文件。重复标签、陌生类型、控制标签混入业务集合均作为非法输出。
- intent报Top1、Top3及有gold支持类别的macro-F1，保留全部60类逐类计数。
- 类型按已审核mask计算micro P/R/F1、支持类型macro-F1；只有全类型审核样本计算集合exact、正样本完整覆盖、空样本指标。额外类型数仅在已审核负类上计，不能把缺标候选解释成已确认误报。
- 非法输出在已知正类上计FN，在已知负类上计invalid，不冒充TN。空样本同时报有效输出FPR和包含非法输出的failure rate，F1必须和协议失败率、监督覆盖率一起报告，禁止通过不返回结果降低表面误报。

21项CPU测试覆盖手算混淆矩阵、多类macro、部分标签mask、空样本缺失输出、控制标签误用、独立任务协议、未知/重复ID和映射缺失。已检查全部当前输入hash；v1审计在评分器增加报告字段前产生，保留不可变原件，v2绑定最终源码，映射与全量统计相同。

本轮未读取选择集或TEST标签，未改原始数据，未把所有available标签批量标为reviewed。此产物仍不是可训练全量数据或共同评测名单：缺少全意图/全类型语义准入、正式评测监督发布、模型推理结果和来源组统计区间。下一步应先补全审核发布，再用相同映射及候选协议制作音频请求和模型对照。
