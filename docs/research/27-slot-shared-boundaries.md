# 槽位共同边界：角色区分、共存与标注争议

2026-11-12。在既有55类定义声明的89对相邻类型上完成全量learning证据统计；其中
46对在同一句有原生共标，29对存在规范化后相同的filler。规范化仅用NFKC、大小写及
空白，不做词干、同义词或模糊匹配。同词跨上下文不同类型不是自动错误；同句共标
也不是意图—槽位绑定证据。

首轮阅读14对的106条不同来源参考文本（107个pair-source观察），编写双方共用的
[14条区分规则](../../configs/tasks/slurp/type-boundary-rules-v1.tsv)。
[结构化卡](../../configs/tasks/slurp/type-boundary-rules-candidate-v1.json)保留证据ID、
共存规则、未解决边界和剩余75对清单。全部仍是candidate，不产生训练标签。

## 可以由学习证据支持的规则结构

- date与general_frequency分别描述日历定位和复现规则，可以同时表达；星期词不能
  单独决定类型。6824的repeating和mondays分别被标两类。
- time与time_zone分别描述时间表达和解释时间的标准；4181中seven pm与PST共存。
  问“现在几点”不等于提供了一个时刻值。
- app_name与radio_name分别描述访问平台与播放的电台；9609同时出现Pandora和
  Hoizer radio station。只出现平台，不保证出现了独立电台引用。
- food_type与ingredient可以分别描述成品与投入材料；9985同时出现steak与chicken。
  这只是可用的语义区分，不能直接替代该数据集所有原生标签。
- email_address与personal_info区别在于地址引用和联系信息字段/属性；问contact
  details不要求已经说出地址，地址本身也不隐含所有字段均被表达。

共用规则用于供给模型判断依据，不能在after-forward按关键词覆盖结果，也不根据
当前query主题挑选本轮规则。尚未冻结输入编排，不能把本文件当已部署提示词。

## 需要保留的反例和未知

| 边界 | 原生learning证据 | 当前处理 |
|---|---|---|
| house_place / place_name | 226的clean the floor和2904的please clean the floor分别标不同类型 | 不能编造角色差异；保留争议，未重标 |
| food_type / ingredient | 10205的recipe for rice与10006的how to cook rice标不同类型 | 成品/材料规则尚不足以重建标注；逐例裁决待完成 |
| date / timeofday | tonight在11890为date，在1514/8625为timeofday | 需要同表达优先级或允许共标的明确政策 |
| time / timeofday | night在6718/7172为time，在3663/4735为timeofday | 不把time扩成所有时间表达来强行吞并差异 |
| artist_name / person | Prince、Elton John有不同上下文赋类；13879一般介绍仍标artist | 人物身份和音乐主题都不是完备的原生判别规则 |
| business_name / place_name | Walmart在12061/15151为business，在7931为place | 商家作为地点的重叠需要原生约定，不假装语义互斥 |

0次共标不证明互斥，0个共享词面也不证明类型无歧义。本次未复听，也未裁决上表
哪一个原生标签应修改。争议不扩散成整类剔除，更不能把相邻类标注自动当可靠负例。

## 复核与后续冻结

证据构造器`tools/jev/audit_slurp_type_boundaries.py`消费已认证slot-support报告并复核
全部依赖hash，仅从当前learning ID对应TRAIN metadata构造证据。共享产物为
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/type-boundary-evidence-v1/report.json`。
范围、hash和阅读名单见[type-boundary-review-v1.json](type-boundary-review-v1.json)。

下一步继续完成剩余边界审查，并将可定案的逐判决监督与未解决项分别登记。完整任务
冻结仍需覆盖、菜单、定义消费和迁移评测；这14条规则不替代全任务验收。正式工程与
训练仍按任务冻结后的顺序推进。

## 全89对规则补齐（v2更新）

后续已阅读剩余75对的全部固定抽样参考例，并逐对写入
[扩展定义表](../../configs/tasks/slurp/type-boundary-rules-extension-v1.tsv)。
[v2结构化规则](../../configs/tasks/slurp/type-boundary-rules-candidate-v2.json)合并全部89对，
旧14对产物保留，阅读范围见[type-boundary-review-v2.json](type-boundary-review-v2.json)。
剩余“未写候选规则”的声明边界为0；这不表示未解决的语义争议为0，也不宣称89对是
完整本体的全部混淆关系。

新增规则覆盖身份/类别、创作者/作品、平台/内容、人物/属性/关系、交通提供商/服务/
模式/属性等边界。它们共同要求用表达的角色区分，可共存时保留多标签；0次原生共标
不建立互斥约束。各规则仍附具体未解决限制，不将“上下文判断”冒充可验证的标注依据。

新发现继续阻止直接自动生成语义负例：Logan在近似放映时间问句中分属event/movie；
Olive Garden有business_name/business_type不同赋类；CST/EST被标place_name的例子
与时区定义不符；音乐场景movie_name和唯一music_album来源仍需裁决。以上均未重标，
不以编写规则的完成量代替原始标签正确性的证据。

下一步转向逐判决监督裁决及固定任务菜单，不再停留于重复扩写同类原则。已有全人口
native统计可作为原生复现证据，但新定义任务的positive/negative覆盖仍须另行发布。
