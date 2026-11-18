# 已知规约冲突修正与自动检查

本轮检查完整55类型、55政策、89相邻边界的结构和共享规则一致性，并阅读全部
相邻规则摘要。发现两类会影响监督的明确冲突，以及上一轮遗漏的顶层范围指针。

第一类：39组边界仍含“如果输入未建立所需角色，就保留未决”。这将完整输入中
明确没有该角色的负例，与角色/指代/附着真正不明确的样本混为一谈，与现有负例
政策相冲突。修订为：完整输入经复核明确缺失时false；真正无法决定的歧义才
unavailable。未评估和原生标注缺失依然不能当负例。此处只修规则，不自动改变标签。

第二类：music_descriptor/playlist_name允许根据播放列表标题中的描述词增加
属性，与内容范围规则的字符串/标题引用政策不一致。改为必须有独立的音乐属性
表达，标题里的字词不能自动提供该属性；真正标题/描述兼用歧义保留未决。
3个人工逻辑对照独立登记，未冒称数据集音频或可训练gold。

第三类：上一版逐卡scope字段已更新，但类型/政策/边界文档顶层scope_contract
仍指旧request-relevance候选；类型及边界顶层scope也陈旧。新包明确指向内容范围
v2，并将先前元信息保留为provenance。上一轮active字段核验没有覆盖这些顶层字段，
不能把它解释成顶层指针也已通过。

[槽位包v13](../../configs/tasks/slurp/slot-consolidated-design-v13.json)使用类型v15、
政策v15、边界v16：55类型卡逐字段保持原样，55政策除定义版本引用外内容不变；
39组边界按公开的精确替换修正，其中一组同时修正标题规则。没有新增角色种类。

新增[规约检查工具](../../tools/jev/audit_slot_contract_consistency.py)，检查：

- 类型/政策集合及重复、边界引用/重复、文档和范围hash。
- 全局与逐类范围一致性，旧范围政策残留。
- 相同role ID在类型/政策/边界中的内容漂移。
- 已知缺失/未决混淆和标题属性捷径的原始错误措辞。

旧版检出44项：39缺失/未决冲突、1标题捷径、3旧范围契约指针、1类型顶层scope。
新版本通过，6个回归测试通过，另独立重建所有行变更范围。
这是有限模式和结构检查，不是自然语言定理证明；换一种措辞的冲突可能漏检，
通过不证明所有语义边界正确，也不认证任何训练标签。

命令：

```bash
.venv/bin/python tools/jev/audit_slot_contract_consistency.py \
  --package configs/tasks/slurp/slot-consolidated-design-v13.json \
  --output /tmp/slot-contract-check.json
.venv/bin/pytest -q tests/test_slot_contract_consistency.py
```

输出文件采用独占创建，保留已有核验记录。旧版失败报告和新版通过报告均归档。

迁移工作包`annotation-workpack-current-rules-v12`保留40513录音/9192来源，
5857原评估/4871坐标不变。15坐标当前明确，4856需重审，500689未评；
当前参考12正3负，0准入。18个上一版兼容评估受到相邻规则变化影响而需重审，
并非删除历史、把未知自动补负或发现18条错标。原metadata绑定、全评估多重集、
语义适用性及55类统计独立重建通过。

- [修正规则](../../configs/tasks/slurp/contract-consistency-resolution-v1.json)
- [旧版失败](slot-contract-consistency-before-v1.json)
- [新版通过](slot-contract-consistency-after-v1.json)
- [逐行变更核验](slot-contract-consistency-delta-check-v1.json)
- [迁移重建](current-rule-migration-check-v12.json)
- [覆盖重建](current-semantic-coverage-check-v8.json)

工作包报告SHA：`75f5aa74bf47deef3c418fe9ce4774eb5cb381e9def035a342a83e9232206190`。
本轮未新增音频听审、模型推理或训练。当前任务/监督/上下文入口已对齐，offset80。
后续应在统一规则下推进全量复审、意图映射与监督准入；这些仍是训练前置交付。
