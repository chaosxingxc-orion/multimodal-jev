# 槽位逐判决审核：从候选定义到独立监督层

2026-11-12。对16条learning来源的38项类型命题逐项阅读完整参考文本，编写证据及
限制，见[人工编写的审核表](../../configs/reviews/slurp-slot-decision-reference-v1.tsv)。
这里的审核者是Codex，依据参考文本，未经音频复听或第二审核者复核；不是人类专家
标注。案例按已发现边界挑选，只作方法诊断，不估计总体错误率，不作为小样本主训练。

| 判决状态 | 参考判决数 | 关联录音判决数 |
|---|---:|---:|
| positive：当前定义成立且与原生标签一致 | 16 | 74 |
| negative：全文可确认不成立且与原生标签一致 | 9 | 53 |
| unavailable：定义边界或证据不足 | 7 | 33 |
| conflict：候选语义判断与原生标注冲突 | 6 | 25 |

185个录音判决关联78条录音、16个来源组，不是185条独立样本。全部记录绑定源metadata、
定义卡、存在政策和review hash，且optimization_allowed=false。未修改原gold、原生
population或训练入口。

## 判例展示了哪些必需区别

- 4181包含seven pm与PST，time/time_zone成立，全文没有日期表达，date不成立。
- 4771询问this time zone的时间：时区引用成立，但没有给出钟表时刻/时长/偏移，
  按explicit_filler政策time不成立。这不适用于另行定义的角色激活任务。
- 15772的number支持个人信息字段；它没有邮箱地址。his是否单独满足person卡的
  指代政策尚未明确，保持unavailable，不能因原生未标就判false。
- 9609同时有Pandora app和Hoizer radio station，两类型成立；9231仅说Pandora radio，
  电台身份是否充分不确定，不能用app_name标签补造radio_name负例。
- 2229的CST/EST在转换语境表示时区，和原生place_name赋类冲突；time的nine am
  仍有独立支持。不能因一项冲突无条件丢掉整条音频所有判决。
- 10006的rice符合当前宽泛food_type定义，但原生仅有ingredient；这是定义与标注
  对应关系的冲突，不等同宣布原作者必须按我们的新定义改标签。

## 监督结构与现有代码的真实边界

现有`slurp_training.project`先从native entities推导类型存在，再要求review.value
等于原生值。因此它可以消费与原生一致的审核结果，却不能表达“保留原生标签、另外
发布新Jev语义真值”的派生层。不能通过伪造native_target或关闭校验绕过这一限制。

新任务冻结后的监督入口需要明确区分native annotation、reference semantic verdict、
audio evidence status与optimization admission。输入定义/菜单先编译，监督后绑定；
冲突判决只有经过明确的新语义发布和准入才能使用，不能在同一原生基线名下重标。
未审核的其他类型保持缺标，不能从38项审核扩充成16×55个完整标签。

本轮构造器只生成审计绑定，不接训练。v2将多来源录音的完整审核检查明确为每个
source_id×entity_type都已审，而非只检查source_id出现；本批数据两者结果相同，
但规则差别对后续批次重要。两版结果保留，不原位覆盖共享产物。

## 独立核验与下一步

共享发布：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/slot-reference-adjudication-v2`。
从全部learning目标重新枚举每个审查来源的录音，核对185个绑定无漏项/重复、源音频
身份、review hash、状态/原生一致关系及逐类型来源覆盖。见
[独立核验](slot-reference-adjudication-check-v2.json)。没有重新解码/复听音频。

后续需按完整人口的定义×来源覆盖展开审核，并独立保留原生评测。不能把此次局部
审核当作全量槽位训练准入；也不因此退回小规模训练路线。任务冻结还需正式解决
指代/相邻类型政策、语义监督发布、奖励归一化和完整评测定义，随后才统一建设工程。
