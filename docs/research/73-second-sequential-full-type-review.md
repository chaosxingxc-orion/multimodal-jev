# 第二批固定来源全类型参考审核

按既定hash顺序完成来源索引16–32的全55类型参考判决，共880坐标：47真、821假、
12未决。来源包括股价、新闻、附近商家、音乐、邮件、日历及时间查询；没有按标签、
模型结果或容易程度选择。使用类型v9/政策v9/边界v10，未修改本体或训练配置。

每来源保留显式负例清单、全文缺失角色解释，以及正/未决的引文和理由。列表中的
负例是在阅读全部定义后逐类型判断，不由native未标注自动生成。

代表性区分：CNN作为文章来源不自动成为news_topic；this time zone可构成识别引用，
不要求恢复时区名称；who is the song playing by表达人物问题，但artist_name的
“named performer”与未赋值角色仍需统一政策。repeat reminder可能指重播一次或
设置周期，不补成frequency=true。日历集合、太阳落山的气象类型等边界集中登记，
不因单一样本反复修改本体。

12未决分别登记在[批次问题清单](../../configs/reviews/slurp-sequential-review-issues-b0002.json)，
区分定义范围和输入歧义。全部是同一作者参考文本建议，未听审或独立语义认证，
正式准入仍为0。34项与原生投影不同不等于34个官方错标。

交付：

- [完整参考判决](../../configs/reviews/slurp-full-slot-sequential-reference-b0002.json)。
- [逐判决及录音绑定核验](full-slot-reference-b0002-check-v1.json)：880建议、2860绑定、52独立录音。
- 当前工作包：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/annotation-workpack-current-rules-v3`。
- [全量合并核验](current-rule-migration-check-v3.json)：40513录音/9192来源不变，2998评估/2057坐标。

当前合并后1580坐标有当前明确参考、30未决、447需要按当前规则重审；503503坐标
尚无参考。当前批与此前证据有1坐标重叠，故880新评估只增加879个独立坐标。
全部原文和版本留存，无自动投票或训练准入。

从原metadata、完整learning/targets重建身份、全部绑定和原评估多重集，直接核验
通过；生产代码未变，不重复运行上轮已通过的同一测试集合。工作包报告SHA为
`befa1472857bb5e43bae62f810ea08022e3bc7f5deca2bbaade9da516d1d10b3`。
这些核验不认证参考语义正确性或音频可听性。

下一固定来源偏移32。完整任务仍未冻结，下一阶段继续批量审核并集中收敛公共边界，
而非扩展研究任务或重启小样本训练；标准RL、完整学术复现与冻结迁移收益尚未完成。
