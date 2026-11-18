# SLURP意图语义卡与任务冻结缺口

2026-11-12。已为全部60个原生意图编写候选语义定义、包括/排除范围、相邻类别及
学习证据引用，见[语义卡](../../configs/tasks/slurp/intent-definition-candidates-v1.json)。
可逐行审阅的[定义表](../../configs/tasks/slurp/intent-definition-authoring-v1.tsv)保留英文
任务描述；证据与审核状态独立于未来actor输入，不能把来源标签、文本示例或类别频数
一起作为模型输入。此次不修改已发布tasks.json，不训练，也不改变原gold或人口。

## 来源与审核范围

证据构造器`tools/jev/build_intent_definition_evidence.py`固定official-clean意图发布身份
`38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`，核验learning、
tasks、原始learning targets及TRAIN metadata的hash，并逐录音核验原target身份和标签。
仅用40,513条已准入learning录音关联的9,192条源utterance构造证据；已知pending来源
不进入。按`sha256('intent-card-v1:'+source_id)`排序，每类最多取5个不同source ID，
不按内容挑选典型例。cooking_query只有2个，合计297条参考文本，已逐条阅读。

共享证据包：
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-definition-evidence-v1/packet.json`，
SHA256 `47cca08c6a39a50122e972cdba378455ce2ca8b4f0d4de3cc5eb01696405e730`。
来源身份和逐类发现见[intent-definition-authoring-audit-v1.json](intent-definition-authoring-audit-v1.json)。

这里的审核是参考文本与候选定义的比对，不是音频复听，不是全量语义复审，也不是
宣布这些定义已经成为官方标注规则。未使用selection/test样本编写定义。全人口计数
只表示来源覆盖，不表示已逐条阅读。所有卡保持draft、optimization_allowed=false。

## 改变下一步任务设计的发现

297条中有30个类别发现至少一项需审查的边界或证据充分性问题。这不是30类标签均错误，
也不能估计全量错误率；相同类别内未抽到问题不能视作通过。

| 边界 | 学习证据 | 对任务的影响 |
|---|---|---|
| cooking_query / cooking_recipe | 13479为食谱请求，13488为糖替代；recipe中10059也是配料替代 | 两条query证据无法支持互斥语义定义；不能靠名称臆造边界 |
| transport_query / transport_ticket | query的12280请求车票，ticket的12590询问时刻表 | 查询与购买的语义规则不能保证与所有原生标签一致 |
| email_querycontact / email_addcontact | querycontact的15976为添加常用联系人 | 联系人检索与修改存在反例线索，需回到源记录审核 |
| general_greet / general_quirky | 常规问候与助手感受问句重叠，quirky另含零散主题词 | quirky不能直接定义为有可靠负例的开放域拒识类别 |
| lighton / lightup、wemo on/off | high setting与开灯混杂；部分off例同时出现off/on | 不能以动作词或亮度词投影独立过程gold |
| 日程、列表、外卖 | calendar_set中出现取列表；列表中出现订货；外卖下单中出现附近店铺 | 不能把原标签投影出的属性宣称为独立语义判断依据 |
| 缺失上下文 | slow、evening、go back、it is dark等 | 词面不足不等于音频已证伪；需检查是否缺少本来应有的上下文 |

不能为了覆盖以上例子而把每张卡扩大到相互重叠，再声称I_class具有可靠互斥语义。
也不能直接删除问题类、合并60类或重写gold来获得更高准确率。下一步先审边界和源记录，
区分标注错误、定义错误、上下文缺失和合理的主意图优先级。任何后续人口变化都需独立
版本、公开原分母和影响范围。

进一步核对了11条关键learning源记录（13479、13488、10059、12280、12590、15976、
7656、1299、326、4571、4572）：原始TRAIN与clean-v7的完整JSON记录逐条相等，
sentence_annotation也保留同样的语义线索。因而这些问题不是本次clean-v7重标造成，
不能通过修复本项目scenario/action映射解决。核验见
[原始来源比对](intent-boundary-source-check-v1.json)。此结果不证明原作者的意图优先级，
也不证明音频内容；其余来源未作原始/clean逐条比对。

对应的冻结处理规则：已验证的原生标签用于原生口径；若新定义与原标签冲突或证据
不足，定义驱动监督保持unavailable/conflict并记录具体判决，不自动生成全类负例。
待逐条裁决后再以新版本发布监督mask及覆盖统计；目前尚未生成这一新训练发布。

## 原生基线与定义驱动任务的关系

保留原生60类分类及其官方clean标签，作为原任务复现口径；这不自动证明所有同标签
样本满足我们新写的定义。新Jev任务要求额外建立定义到标签的可审计映射，才可把
“其他类别”作为I_class的负例。I_present的缺席更不能从单标签数据推导。

定义驱动Q是固定算子加任务级定义；以上证据审查不能用于按当前query主题挑选问句。
样本例只用于离线审核。槽位负例和绑定关系仍需独立准入，不借意图卡扩大监督权限。
对模糊但原生标签已发布的样本，此轮只登记问题，不将其强行转成unknown或false。

冻结前依次完成：上述高风险边界审核；对应任务的正/负/缺标覆盖；固定菜单及token预算；
定义消费、Final和跨域评测协议。完成后再开始工程release验收和正式训练。仅补齐卡片
数量不足以解除`training_dispatch_allowed=false`。

## Cooking后续完整来源审核

已扩展至这两个原生类别全部167 learning来源/681录音，逐条拟合当前recipe定义，
142支持、7不支持、18未决；不把拟合判决视为互斥I_class或音频认证。
详见[完整审核](37-cooking-definition-boundary-adjudication.md)。query两条均支持recipe定义，
因此不能用现有query卡的自指描述作为可迁移语义边界；原生60类标签及分母保持不变。
