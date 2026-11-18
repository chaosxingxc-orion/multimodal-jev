# 首批全类型判决的新规约重审

按原hash顺序重新阅读首批16来源及当前55类定义、政策和相邻边界，全部880坐标
显式评估，覆盖旧正、负、未决三类。没有按native标签自动补负例或仅更新旧未决。
当前规约仍为类型v8/政策v8/边界v9；本轮不修改定义或派发训练。

新版参考建议40真、837假、3未决，共30坐标相较旧建议改变。26个旧未决得到明确
建议；另有3个旧time正例改false、1个旧music_descriptor正例改未决。

| 来源 | 变化 | 原因 |
|---|---|---|
| 17101：my inbox ... at the moment | time true→false | 泛指当前语境，不自动成为时刻/时长/偏移或顺序 |
| 2346：current temperature | time true→false | current修饰词不独立构成time |
| 15116：recent news | time true→false | 未定量recency过滤未明确时刻、时长、偏移或事件间顺序 |
| 11334：podcast of folk songs ... rating above four star | music_descriptor true→unavailable | 评分可能修饰播客或歌曲，不能用原生标签确定附着对象 |

首批仍有三项未决：9877“my audiobook”的内容所有者是否建立person，15420
“on twitter”的发布渠道是否同时建立media_type，以及上述评分附着歧义。
前两项登记为待收敛的类型边界；后一项是输入歧义，允许保留mask，不为单一样本
强造规则。先前五组边界已有规则不代表完整本体已冻结。

新版[参考文件](../../configs/reviews/slurp-full-slot-sequential-reference-b0001-v2.json)
逐来源保留全类型负例清单、正/未决引文、重读解释和变化明细。旧记录不覆写。
使用已有binder生成880建议及3190录音判决绑定，覆盖58独立录音，产物位于
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/full-slot-reference-b0001-v2`。

从原始metadata、完整learning/targets重新构造身份和绑定，核对全部55类型覆盖、
引文、定义/政策/边界hash、30处变化及建议/绑定多重集，证据见
[核验记录](full-slot-reference-b0001-check-v2.json)。这是独立路径的结构核验，
不是独立标注者语义复核或听审；40/837/3是参考建议，正式准入仍为0。

此批为全9192来源日程的重审进展，不是小规模训练数据。当前不可变迁移工作包未
原位改写，新重审独立发布且未合入其计数。下一步解决两个残余公共边界，继续固定
来源日程的全量审核，并以显式版本合并新建议；不将未决输入补成负例。
