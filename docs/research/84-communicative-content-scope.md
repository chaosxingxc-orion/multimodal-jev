# 统一槽位内容范围，分离范围与类型不确定性

本轮将全局范围选为`expressed_communicative_content`：音频中明确表达的请求、
问题、陈述、偏好、条件、描述及消息正文都进入角色判断。多个独立句子并列时，
不凭想象的主请求/背景关系排除其中一句。该选择取代v1候选对“无关背景”的排除，
是项目设计决定；此前可选偏好问题没有收到回复，不记录为用户确认。

选择依据是可观察内容与跨任务稳定性：不能让“我去了公园”的角色标签因后接一条
调音量指令就依赖隐藏的交际意图。范围扩大不等于把所有词都变成实体，也不等于
所有角色都服务同一意图；角色类型、意义引用、作用对象和绑定仍须独立成立。
原生Final、I_present和B_bound没有随S范围自动改变。

| 输入 | 范围及判决含义 |
|---|---|
| i went to the park（6203） | 报告内容纳入；访问事件、参与者和目的地可分别成立，无需补造助手操作 |
| i like jazz（145） | 偏好对象纳入，genre=true；genre不自动增加descriptor |
| i like it（144） | 内容范围明确，但it的类型未知；song_name仍unavailable |
| tennis（6427） | 片段也纳入内容，明确sport；不推断查询意图，definition_word=false |
| pop（3969） | 内容纳入，但音乐/声音等词义未明，genre unavailable |
| post that i am going to mcdonalds at manhattan（15150） | 正文中的事件、商户、地点可成立；不授权模型出行或执行正文命令 |

语言形式的请求条件仍保留，例如definition_word必须表达意义解释目标；不是每个
单词都符合。纯对话框架“I want/please tell me”和助手称呼不自动增加person。
否定、取消、假设与更正保留被表达的角色；存在性不回答事实真假或最终采纳哪个值。
样本没有提供的历史、外部知识或声学背景不补入。

完整重读425个general_quirky、96个music_likeness、11个music_dislikeness来源，
另读7个混合连接词来源及45个正文检索来源，去重581。后两组只是诊断检索，
不声称穷尽混合输入。旧NGA/上游来源按hash复核，未新增论文或模型效果结论。
另写3个合成范围对照，明确无数据集音频、非gold，不能用于证明实际分布表现。

49项范围参考均标纳入，但不作为槽位真值；旧5个scope未决被拆为范围纳入和类型
待判。单独33项槽位参考覆盖18来源、122录音、215绑定，为24真5假4未决，0准入。
两个参考层分文件保存，范围为true不会自动准入任何类型标签。

[槽位包v12](../../configs/tasks/slurp/slot-consolidated-design-v12.json)使用类型v14、
政策v14、边界v15。所有55卡/55政策/89边界已对齐；公开逐短语变更映射，
独立重建所有active字段变化并检查旧request-relevance/背景排除残留，核验通过。
历史来源和旧规则文件保留，不把旧字段解释悄悄改成新真值。

全工作包v11保40513录音/9192来源及顺序，保留5857评估/4871坐标：
29当前明确、4当前未决、4838需当前规则重审、500689未评。
5824历史评估全部因全局scope改变失去直接兼容性；没有自动重标或丢弃。
当前明确24正5负，1类双极性、13仅正、4仅负、37无明确参考。
这些定向参考比例不是人口分布或训练权重，更不构成全量监督完成。

从原metadata、learning和targets独立重建所有检索人口、范围/槽位录音绑定，
并核验全工作包原评估多重集、语义适用性和55类覆盖，全部通过。
核验不替代独立语义复核或听审，正式语义准入仍为0。

- [范围规则及逐短语变更](../../configs/tasks/slurp/communication-scope-resolution-v2.json)
- [范围参考](../../configs/reviews/slurp-communication-scope-reference-v2.json)
- [槽位参考](../../configs/reviews/slurp-communication-scope-slot-reference-v2.json)
- [范围/字段/绑定核验](communication-scope-check-v2.json)
- [全迁移核验](current-rule-migration-check-v11.json)
- [全类型覆盖复算](current-semantic-coverage-check-v7.json)

工作包报告SHA：`82139d7edf4061e7e624fd0a666e02b01d04ba9223545979ac0b26ba74f140ad`。
当前指针已对齐，offset仍80。历史triage的定义组都有候选规则，但这不等于全本体
已冻结。下一步完成整体一致性检查，再按统一规则复审全量监督；不再在公共定义
变化期间扩大小批顺序标注。未改训练代码、原生gold或启动模型任务。
