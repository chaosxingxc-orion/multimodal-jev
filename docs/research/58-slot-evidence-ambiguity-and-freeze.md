# 槽位证据歧义与任务冻结边界

2026-11-12。复核既有9231 radio_name与10205 ingredient两项未决。现有规则已明确
要求电台/频道角色以及烹饪输入角色；“Pandora radio”与“recipe for rice”仍有无法
从本句排除的解释。保留unavailable，不为减少未决数量强行改成false或双true。

## 服务与电台的完整原生app_name队列

阅读learning中全部42个原生app_name来源、99个原生radio_name来源及16个原生
ingredient来源的完整参考文本。对42个app_name来源逐条作radio_name建议，不按
预期真假选例。这个队列来自原生正类，不代表全语义app存在人口或总体标注错误率。

| 参考建议 | 来源 | 录音绑定 |
|---|---:|---:|
| true | 9 | 27 |
| false | 24 | 83 |
| unavailable | 9 | 21 |

其中5项true与native一致，4项不同：9213“my rock station on pandora”、9328
“classic rock station”、9392“radio station on spotify ... yesterday”、9649
“hip hop channel”。在当前定义允许描述性电台引用的前提下，有明确station/channel
角色不需要恢复专有台名。旧9231未决、9609 true与本轮相同。

纯“open Pandora app”“play Spotify”等只给服务访问请求，没有台/频道引用，建议
false；“my Pandora radio”“personal channel”“Howard Stern on Sirius”等角色
仍不足以唯一确定，保留未决。不借外部品牌知识代替本句证据，不用radio词自动判true。

逐项建议见[TSV](../../configs/reviews/slurp-app-cohort-radio-reference-v1.tsv)。共享
app-cohort-radio-reference-v1及app-cohort-radio-semantic-v1分别保留原绑定及语义/原生/
准入分离字段。42建议、131录音绑定全部pending；未听审、非独立语义认证，未改gold。
[独立身份核验](app-cohort-radio-check-v1.json)重建完整队列及全部录音关系、核对hash。

## 冻结条件的澄清

**定义缺陷与输入歧义分开。** 定义未给规则、规则互相矛盾，必须解决后才发布；规则
明确而某个输入缺证据，可以在冻结任务下保留unavailable。不能要求所有数据都能
强制给出true/false，进而无限扩充规则或改变原意。

这一点并不自动让当前任务就绪：仍须全量状态覆盖、逐定义可靠正负监督、已知分歧
处理、固定有效规模与权重、选择集评测协议。不可用仅有42或38条建议的小集代替
规模化任务；也不可把全量空mask包装成可训练数据。缺标mask与“决策模型输出unknown”
不同，后者需要单独任务定义及真值。本轮将该政策写入semantic-supervision-plan-v1。

ingredient的recipe-for未决不通过改变食品词表解决，现有规则不改。所有既有本体
hash与审核发布保持不变，本轮仅新增角色证据及冻结条件澄清；没有工程接入或训练。
