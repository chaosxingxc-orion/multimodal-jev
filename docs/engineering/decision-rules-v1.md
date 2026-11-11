# 思考要求下的规则判决与学习侧定义证据

2026-11-11。新增decision_rules组件，用明确的fact/all/any/not/except树表示可重算的Boolean任务；渲染完整语义定义，而非让模型背内部规则ID。
ThinkingSpec与业务判据分开：思考预算改变不会自动改变gold；业务谓词改变则重新计算答案。
规则适配器生成显式思考配置的Noul契约，无文本推理输出，尚未接入训练release。

事实必须显式标记reviewed/unavailable/conflict。只有reviewed接受Boolean；缺失和冲突保留None，不当false或actor拒答金标。
使用三值逻辑：已知false AND unknown可判false，true OR unknown可判true；无法确定的规则目标保持不可评分。
原始来源包annotation_status=available不等于reviewed，当前没有自动将SLURP标签转换成已审查事实。
结构family只表示算子形状；except与all/not等语义等价形式仍需统一留出登记，不能靠形状不同就宣称规则隔离。

规则与配置相关33项测试通过，包括完整Boolean真值组合、缺标/冲突传播、规则变化重算、字段/算子拒绝、结构分组及已有thinking契约。
这些是规则程序测试，不是模型思考效果或RL收益证据。

另发布私有学习侧定义审查证据：
`/mnt/cephfs/xingchao/datasets/slurp-clean/clean-v7-a3cd74482c32b74f/learning-definition-evidence-v1`。
覆盖9,223个学习侧文本来源、40,713个录音、60个scenario/action组合和55个实体类型，抽取316个待审上下文示例。
先按已验learning目标名单准入来源；完整TRAIN元数据只作哈希认证及按ID筛选。未打开selection targets或TEST。
卡片保存原标注、上下文、类型/意图支持统计；定义与成立判据仍为空，review=pending，optimization_allowed=false。
原文例子留共享数据目录，权限0600，不提交Git、不进入actor。

独立复核全部卡片统计及316个例子的learning成员身份、原文、intent、逐实体token-span与录音数，见[凭证](slurp-learning-definition-evidence-v1-verified.json)。
这为下一步逐类型定义审查提供当前划分内的依据；没有把抽样例子代表全类型正确性，也没有由未标注自动生成负例。
