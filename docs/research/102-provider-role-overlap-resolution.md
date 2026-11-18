# 服务提供者与交通角色重叠规则

本轮完成上轮定位的定义缺口，保留原生SLURP标签不变，仅修订项目语义槽位契约。
当前入口为 `configs/tasks/slurp/slot-consolidated-design-v14.json`，类型/政策/边界分别
v16/v16/v17。65张意图卡及正在执行的意图教师校准不变。

## 定义裁决

business_name已经把明确服务提供者纳入定义，因此同一表达的交通提供者允许同时满足
business_name和transport_agency；不要求出现两个实体、第二次提及或额外商业交易。
这不是根据一个已判positive的类型自动闭包，而是分别检查同一输入是否支持两项定义。

AA flight、Ola cab等服务归属构式可以表达运营/提供关系，无须解析公司全名或外部所有权。
Samsung phone、Wemo plug等仅标识物理产品的品牌修饰不因此表达公司角色。单纯打开
应用也不成立；应用访问与明确归属的出行请求并列时，可以分别支持软件和提供者角色。
股票/投诉语境可建立组织角色，但不自动建立交通关系。

business_type包含明确表达的商业机构或服务类别。book a taxi/flight明确提供服务类别，
可同时表达transport_type；仅描述车辆实体、机场位置或标题中的flight不自动表达服务。
无商业/服务角色证据时，不从交通类型推导business_type。

源规约见 `configs/tasks/slurp/provider-boundary-resolution-v1.json`。这是项目任务语义
选择，不声称复现官方实体标注边界，也未证明跨域效果。

## 参考与迁移

从已读62个学习来源中选18个角色对照，每个审阅business_name/business_type/
transport_agency，共54建议：20正、24负、10未决。参考原文保存在
`configs/reviews/slurp-provider-boundary-reference-v1.json`。裸Uber、find me an Uber等
在不引入品牌知识时保留歧义；IRCTC site不凭外部身份补交通域；Ola event不猜赞助关系。
原生gold不决定问句或新真值。所有参考为同作者文本审核，无音频复听或独立语义认证。

独立原metadata和官方learning/target逐项重建了54参考、273录音绑定（91不同录音），
核验精确变更：3张类型卡、对应3政策及12组边界，52张类型卡原样保留。
55/55/89结构、共同角色规则一致性和文件pin检查通过；证据
`docs/research/provider-boundary-check-v1.json`及`provider-contract-consistency-v1.json`。
结构通过不代表自然语言规则已全面认证。

新工作包：
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/annotation-workpack-current-rules-v13`

保持40513录音/9192来源及全部旧评估，现5911项评估、4921坐标；57当前明确参考、
10当前未决、4854待按当前规则重审、500639未评。当前兼容参考没有自动成为训练标签。
迁移器未改；独立重建全部来源绑定、评估多重集、逐类型语义兼容性和坐标状态通过。
report SHA256 `b5e278a4fe5117fedd4ede0b4c705c700db911ad0da83acc58bfa1414fc84b7e`，
证据 `docs/research/current-rule-migration-check-v13.json`。
相关24测试通过；新增语义训练准入0，未派发训练。

## 下游边界

主task、监督计划及上下文候选已切换新槽位包。既有blind/shared教师工作包仍绑定v13，
明确标为旧槽位规则；新槽位请求前须重建，历史输出不覆盖。意图定义/请求未变，因此
122次意图校准无需中断或重发。新文本尚无processor/NPU上下文验收。

下一步按原校准计划收尾并审核分歧；槽位新请求须使用新版完整规则，不能凭本轮
规则清理直接宣布输出耗尽已解决。过程奖励的语义重叠审计应纳入同一提供者双positive，
不因本体包含关系重复奖励而声称更多独立推理步骤。

按间隔约定于启动后732秒检查：PID3830498和实际runner命令匹配，66/122请求终态，
总进程尚在运行；距上次观察约602秒。未部分评分、未重启，下一普通检查至少再隔600秒。
