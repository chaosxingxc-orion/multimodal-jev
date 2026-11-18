# 显式属性问句与未提供的属性值

本轮继续执行任务设计→工程冻结→训练的顺序，没有模型forward、训练或下载。

## 任务规则

对general_frequency、time、transport_descriptor、weather_descriptor统一采用：
**输入显式表达了具有确定类型的属性/角色，就可判存在；不要求同时给出答案值。**
这包括明确问该属性的值。任务通常需要某个参数、泛泛提到领域、原生intent标签，均
不能替代输入证据。输出仍是类型存在的布尔决策，不生成或恢复属性值。

四类政策明确切换为`activated_role_allowing_unspecified_value`，限定为输入显式
激活，而非凭任务常识补槽。其他51类不自动继承此政策。新规则见
[property-role-boundary-resolution-v1.json](../../configs/tasks/slurp/property-role-boundary-resolution-v1.json)。

| 来源表达 | 判决 | 理由 |
|---|---|---|
| how often does the train arrive and depart from penn station | frequency=true，transport_descriptor=true，time=false | 明确问服务频率；到发动词不单独建立时刻或时长 |
| what is the time | time=true | 明确问时刻，答案尚未提供 |
| how long does it take water to boil | time=true | 明确问时长 |
| how long is the nile river | time=false | 此处是空间长度 |
| what time zone is arizona in | time=false | 时区身份不等于时刻 |
| what is the current temperature in miami | weather_descriptor=true | 明确问气象属性 |
| what temperature should the oil be for frying french fries | weather_descriptor=false | 烹调温度不等于气象温度 |
| weather this week | weather_descriptor=false | 泛指天气，未表达具体条件/属性 |
| can i wear sandals | weather_descriptor=unavailable | 可能是气候、着装规定等，不能用gold天气意图消歧 |

time保持与calendar/day-part/recurrence的边界；普通时态、now修饰词、泛指日历不全部
变成time。单独when若无法确定日期/时刻粒度则未决。每三小时可分别满足频率及区间
时长；这与只问how often不同。交通属性包括明确问到的票价、到发、服务频率、行程
时长和路线条件；交通方式、服务名称、地点本身不足以自动建立该属性。

## 与原生标注的区别

上游指南143–145段把询问time/date的词标作query_detail；提供的时刻又使用time或
time_start。天气208/210段给出rain条件示例。原生标注不能直接证明本项目激活政策；
本轮是明确的新语义任务选择，不是修正官方定义或认定原生空标为错。

53个参考判决中15个与原生类型投影不同，不能称15个官方标注错误。原生评分继续
维持原标签与分母；新S_present需要单独发布监督和指标。旧true/false/unavailable
均需检查受影响语义，不能只把原先未决项改掉就宣布迁移完成。

## 阅读与核验范围

完整阅读四类原生阳性并集889来源/3939录音，以及固定问句regex选择的280来源。
两组去重1129来源；原生各类来源数为frequency68、time610、transport_descriptor6、
weather_descriptor243。检索只用于离线证据选择，不是在线Q路由或自动标签规则。

形成46来源53个参考判决：33 true、14 false、6 unavailable；550个录音判决绑定，
296条独立录音。首批16来源中4个未决属性坐标完成规则下重判。上述数目不与旧审核
直接相加，当前没有完整语义监督发布。

从原始metadata及完整40513学习录音/9192来源身份重新构造两个证据集合，验证来源、
参考文本、metadata hash、全部绑定和原始target hash。检查新版55定义、55政策、
89边界：4类及12条相邻边界改变，其余51类定义逐字段相同；news/event/work规则保留。
这是独立路径的结构核验，不是独立标注者语义复核，也未复听音频，优化准入仍为0。

- 新规约入口：[slot-consolidated-design-v5.json](../../configs/tasks/slurp/slot-consolidated-design-v5.json)，类型v7/政策v7/边界v8。
- 参考判决：[slurp-property-role-boundary-v1.json](../../configs/reviews/slurp-property-role-boundary-v1.json)。
- 核验记录：[property-role-boundary-check-v1.json](property-role-boundary-check-v1.json)。
- 共享证据：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/property-role-boundary-evidence-v1`。

当前完整规约仍无processor/NPU验收。接下来收敛个人属性/关系/定义目标边界，随后
统一迁移监督；意图定义映射、全量有效标注、工程与标准算法生命周期仍需完成。
本轮没有生产代码变更，执行身份/依赖/作用范围的直接核验，不以无关测试替代验收。
