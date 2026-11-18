# 人类所有者与内容渠道

首批全类型重审剩下的两项定义边界采用统一角色规则；本轮不训练、不改原生标签。

**person包含用于识别请求对象的人类所有者。** my audiobook、my playlist、my inbox、
my calendar、my car遵循同一原则。单纯I want、please tell me或呼叫助手不算人类
任务角色。公司所有者、助手所有者或无法确定人类的指代不自动转为person。
所有者不自动成为作者，也不自动建立personal_info或relation。

**media_type包含明确承担内容获取、承载或发布职责的渠道。** 在Twitter发布投诉、
从Facebook读取帖子、通过Pandora播放内容可以建立渠道角色，同时可能独立满足
app_name；不要求两个独立名称片段。只打开应用、查股价、找联系handle、检查登录或
读粉丝数等账户元数据，不凭平台名称建立媒体渠道。裸平台名或含糊的“Facebook info”
保留未决。内容种类如news、audiobook仍可直接支持media_type。

上游188/196/200/201段明确给出BBC、radio、Facebook发布/读取作为media_type的例子；
person例子涵盖参与者和联系人。人类所有者扩展及双谓词政策是本项目明确选择，不能
称为与原生标注完全等价。

## 证据和影响

按预先列出的文本模式读取278条所有者参考（1357录音）和155条平台参考（578录音），
共433来源；不是person/media_type全量阳性，也不是自动标注规则。判决只使用表达
的角色，不使用原生intent或平台词典路由在线Q。

形成33来源35个参考坐标：20真、10假、5未决；172绑定，162独立录音。15项与原生
投影不同不代表15个官方错误。初批两项未决9877/person、15420/media_type均有true
参考建议；此外15420中的my complaint同样表达内容所有者，因此person旧false也
需要变true。不能只为my audiobook修个例而放过相同规则的其他对象。

当前入口[slot-consolidated-design-v7.json](../../configs/tasks/slurp/slot-consolidated-design-v7.json)
引用类型v9/政策v9/边界v10。仅2类和14条相邻规则改变，其他53类型卡逐字段不变。
证据分别为[规则](../../configs/tasks/slurp/owner-channel-boundary-resolution-v1.json)、
[参考建议](../../configs/reviews/slurp-owner-channel-boundary-v1.json)和
[结构核验](owner-channel-boundary-check-v1.json)。共享产物在
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/owner-channel-boundary-evidence-v1`。

从原始metadata、完整learning/target重建两组来源及全部建议绑定，核对定义、政策
和边界hash。此为独立路径结构核验，不是独立标注者认证或音频听审；全部0准入。
没有生产代码变更，不运行无关训练测试。当前工作包v8/v8/v9和首批v2建议仍为不可变
历史证据，需要明确迁移person/media及相邻边界，不能把旧hash贴到新版定义上。

评分修饰播客还是歌曲仍是输入歧义，保持mask。已发现的公共边界均有规则不等于
全本体冻结；下一步统一当前工作包、完成受影响重审并继续全量来源日程。完整任务
监督、意图映射、上下文及RL工程验收仍未完成，训练派发为false。
