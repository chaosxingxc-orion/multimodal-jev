# 槽位面板拆分与服务提供者边界

上一轮发布全65意图固定参考校准及122请求，本轮确认其PID3830498和实际命令存活、
尚无总终态。启动后约130秒检查时已完成16请求；该次检查早于约定600秒，后续不得
沿用此频率，下一检查以本次观察时间为起点至少600秒。未读这些部分输出作选择或评分，
未重启。工作转向可独立完成的槽位失败诊断及离线评分验证。

## 两请求单变量对照

S_present-00此前4096与8192均耗尽reasoning预算且最终正文为空。
本轮保留同一来源、system消息、全部ontology（包括邻居/边界）、输出schema及服务参数，
只将8个问题按既有顺序分为两个4问请求。两请求串行，每次上限4096，不重试。
完整记录位于`.local/research/teacher-slot-split-v1/run/`，原执行session24334已正常结束。

| 分组 | 问题 | 结果 | 输入/输出token |
|---|---|---|---|
| 前4 | alarm_type/app_name/artist_name/audiobook_author | HTTP200/stop，4项完整，均negative，20.37秒 | 21740/1039（reasoning799） |
| 后4 | audiobook_name/business_name/business_type/change_amount | HTTP200/length，正文空，56.20秒 | 21740/4096（全部reasoning） |

独立检查实际请求的service参数、system及全部user字段，确认只有questions变化，两个
分组精确覆盖原8问且无重复；重新严格解析前组完整ID及非空理由。
证据 `docs/research/teacher-slot-question-split-observed-v1.json`。
共43480输入、5135输出，只有4/8完整判决；此拆分未解决全面板完成问题。
这是单次随机响应对照，不能断言特定问题导致耗尽，更不能证明语义准确或批量可用。
不继续提高预算、不修改完整工作包或actor菜单、不将失败补负。

## 实际规则缺口

检查后半组请求所带的规则，发现需要审定的角色边界：

- business_name包括明确表达的组织、商户或provider；同时排除单纯产品品牌修饰语。
- business_name/transport_agency的shared_rule要求区分商业角色和交通提供者角色，
  coexistence仅举“交通提供者与目的地商户可以共存”，没有明确同一提供者的双类型关系。
- business_type包括商业机构或服务类别，与taxi等交通服务类别的关系仍需明确。

不能因为相邻类型存在就默认为互斥，也不能仅凭一个类型positive就自动闭包另一个。
AA flight中服务/运营者关系是否已表达，应由统一语义规则决定；不能简单套用
Samsung phone等实体产品品牌规则或外部公司知识，也不能猜测评测者想要哪个答案。

已读learning人口内全部23条原生transport_agency来源，以及固定词面检索41来源，
合计62个不同来源。原生标签只定位材料，不供应新语义真值：

- 23条均为Uber/Ola，不能单靠这些原生标注解决AA flight。
- 12657为AA flight；12829/12830为Ola cab；12330/12918同时表达打开应用和安排车辆。
- 14718问Ola/Uber股价，12841仅说open Lyft；13443是Samsung phone，15378是向Samsung投诉。

问题与完整source IDs、原metadata身份登记于
`configs/reviews/slurp-provider-boundary-review-issue-v1.json`。当前本体不变、未新标注、
未准入。下一步针对这组边界作统一裁决和参考delta，再显式迁移受影响监督；不展开新的
任务族，不依教师输出迁就规则。

## 评分链路补充验证

新增完整本地响应目录测试：合法全量面板的额外答案保存但不计gold；length即使有合法
JSON也不算完成；正文空/缺项保留失败分母；无总终态拒绝评分；实际发送消息变化拒绝
评分。与已有解析、面板及选择测试共30项通过。测试覆盖离线评分行为，不代替教师质量。
新语义监督准入保持0，研究训练未派发。
