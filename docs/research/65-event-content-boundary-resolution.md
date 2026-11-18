# 事件内容与助手操作的边界

2026-11-13。本轮完成event_name任务级规则修订和参考对照，没有模型forward或训练。

## 任务规则

event_name表示被表达为请求内容或参数的事件、场合、约定或活动。
当前助手操作动词本身不自动成为事件标签。

| 表达 | 新规则下参考建议 | 原因 |
|---|---|---|
| turn off smart socket | false | 仅指定当前控制操作 |
| remind me to turn off the lights in ten minutes | true | 关灯是提醒内容 |
| recommend a movie which is released last friday | true | 上映是限定影片的过去事件 |
| play country | false | 播放是操作，country是音乐类型 |
| send email stating I will attend the party | true | 聚会/参加是消息内容 |
| create an empty calendar event | true | event类别已明确表达，不要求具体值 |

事件不局限于日历意图、专有名称或名词短语；子句、过去事件、条件和时间锚点也可
表达事件。取消、否定和假设不自动擦除事件存在性。本任务不预测事件实际发生、
用户承诺或环境状态。普通静态属性和内容种类本身不转成事件。

“事件参数”是供模型使用的任务规约，不是根据当前样本原生意图选择问句的路由。
所有actor仍直接消费音频及固定定义，参考文本只用于离线审核。

## 依据与已检查范围

上游指南139–141段将cancel/schedule等操作与business meeting/chat内容区分。
本轮读完learning中全部778个原生event_name阳性来源（3091录音），横跨24个原生
意图，涵盖提醒活动、消息内容、生日/节日、过去会议、条件触发与控制动词作为提醒内容。
完整队列已存于共享`event-role-boundary-evidence-v1/sources.jsonl`。

该队列是原生正类审查，不代表全部语义事件人口。扩展至子句/发生过程是项目语义任务
设计，不能宣称原生所有正负标签与新规则一致，也不据此估计总体错标率。

## 对照判断及版本迁移

对固定首批16来源全部重判，并选择28条边界对照，共44条参考建议：30 true、12 false、
2 unavailable，绑定180录音。4项与native投影不同，不自动判定原生错误；独立语义
复核和新增训练准入均为0。没有听审音频。

首批16条中实际有**8项**event_name从未决改为新规则下明确建议；进度消息中的7项
是计数口误，以独立核验的迁移清单为准。其余8项不变。这是定义修订后的重新判断，
不是模型准确率提升，历史记录不覆写。

规则文件：`configs/tasks/slurp/event-role-boundary-resolution-v1.json`。
统一入口升级为`slot-consolidated-design-v3.json`，绑定类型v5/政策v5/边界v6；
仅event_name语义改变，其余54张类型卡逐字段不变。6条相邻边界加入事件内容规则，
此前news_topic修订保持不变。所有旧policy definition_ref须显式迁移，不能只换hash。

证据：`docs/research/event-role-boundary-check-v1.json`，记录778来源身份、44条参考
metadata与180录音/target绑定的独立重建、55/55/89依赖校验和版本差异。
这是结构与身份核验，不是第二语义标注者；本轮未修改生产代码，没有运行无关测试。

仍需收敛作品识别性引用、未赋值属性与个人属性/关系边界，再进行统一监督迁移及全量
审核。当前任务未冻结，新文本尚无processor/NPU验收，训练派发继续为false。
