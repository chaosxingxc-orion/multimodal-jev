# 事件省略、设备、平台、品牌与太阳现象的角色边界

本轮集中处理历史triage中五组狭义定义问题，不启动工程训练。读取134个事件省略
检索来源、309个结构物/提醒来源、283个平台/品牌来源、31个太阳现象来源、
74个品牌/结构物来源及13个显式对照，去重820个learning来源。
全部参考文本已读；没有复听。检索并非完整语义覆盖，更不是标签生成规则。

| 判决类型 | 规则 | 源样本处理 |
|---|---|---|
| event_name | 日历条目查询/删除可以通过what/anything等表达事件内容；有参与者及时间的明确预约构式也可 | 8505“what's on my calendar this month”、8698“put me in ... with al”→true |
| device_type | 要求设备角色；开关动作不自动赋予结构物智能设备属性；提醒记录不等于硬件 | 13534“open the door”、3985“set an alarm at six”、1634“show me my alarms”→false |
| app_name | 声明式词汇约定允许社交语境tweet指明Twitter、Google搜索动词指明搜索服务 | 15684“write a tweet ...”→true；8237“hey google ...”仅称呼→false |
| business_name | 公司/商户/服务提供者角色须表达；产品品牌修饰语与应用平台不自动扩展成公司 | 868“wemo plug socket off”→false；15378向Samsung投诉→true |
| weather_descriptor | 太阳/月球天文属性和日落时刻本身不是气象属性，sunny预报和环境温度可以是 | 2471“what time does the sun set”→false；4053“is it sunny outside ...”→true |

事件规则仍区分容器与内容：日历空闲时段不是已表达事件，购物清单不自动成为事件，
仅执行open/send/switch等操作也不新增事件。语法残缺或referent领域未明保留未决。
设备规则不要求所有设备都出现smart字样；插座、电视、吸尘器、咖啡机等硬件角色
本身可以成立。普通门窗不能因可被智能控制的假设而标true；alarm clock和clock
若未明确硬件/软件则保留未决。不能以“任何词都可能有其他语境”把明确缺失改为unknown。

tweet→Twitter是本数据集年代下的显式词汇约定，不是从gold意图推平台，也不代表
跨数据集/跨年代天然有效。迁移需保留该约定的版本和适用范围。智能机械门窗和
tweet鸟鸣排除作为规则组成部分明确标注为设计扩展，未冒称本队列观察到这些样本。

[槽位包v11](../../configs/tasks/slurp/slot-consolidated-design-v11.json)使用类型v13、
政策v13、边界v14；5张卡和19组相邻边界更新，其余50张卡逐字段不变。
固定Q算子和题目选择协议不变，全局request_relevant_role范围未扩展。
64项参考建议覆盖59来源、213录音、234绑定，30真、28假、6未决，0准入。
其中20项与原生投影不同不等于20处官方错标；新语义与原生任务分别保留。

历史triage五组定义问题对应的6个坐标均获得当前明确参考判决。
这不代表全本体已冻结：通信范围、完整意图定义及其他输入歧义仍需处理，
旧triage是v7证据，不能拿其计数描述当前状态。

工作包`annotation-workpack-current-rules-v10`保留40513录音、9192来源、全部历史
评估及原顺序：5824评估覆盖4842坐标，1534当前明确、31未决、3277需重审，
500718尚未评估。当前明确参考106正1428负；20类双极性、12仅负、23无当前明确参考。
旧参考兼容数下降来自相邻规则契约变化，并非删除原记录或语义收益。
应集中完成公共定义再复审全量监督，不继续在规则变动期间扩大零散小批判决。

原metadata、learning和targets独立重建了6组检索人口及全部录音绑定；核验了
卡片变更范围、全工作包原评估多重集、版本适用性和55类覆盖统计，均通过。
身份/计数核验不构成独立语义认证。未改生产代码，未重复运行不受影响的训练测试。

- [规则及证据组](../../configs/tasks/slurp/remaining-role-boundary-resolution-v1.json)
- [参考判决](../../configs/reviews/slurp-remaining-role-boundary-v1.json)
- [规则/绑定核验](remaining-role-check-v1.json)
- [全迁移核验](current-rule-migration-check-v10.json)
- [覆盖复算](current-semantic-coverage-check-v6.json)

工作包报告SHA：`ff6c6725939a15a6def30dd550a99292014457b3c1adc650383272e419692f16`。
任务、监督、上下文入口已更新，offset仍80，新语义准入0。
新版上下文未验收，标准RL配置尚未接入验收；本轮没有模型推理、训练或远程状态查询。
