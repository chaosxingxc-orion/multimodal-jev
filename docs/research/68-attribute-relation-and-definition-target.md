# 个人属性、人物关系与词义目标

本轮完成首批全类型参考审核暴露的最后一组公共边界规则，没有模型推理或训练。
这意味着已登记的五组问题都有规则版本，不代表全本体、全人口或监督已通过验收。

三类分别要求以下证据：

- `personal_info`：输入明确表达个人属性或联系字段，包括年龄、婚姻状态、职务、
  部门、住址和联系电话。允许问值但不供值；企业联系字段也包含在此任务定义中。
  人名、收件人、物品所有格、任意与人物有关的事件不自动成为个人属性。
- `relation`：用于指称人或人群的亲属、社会、职业角色，或明确的人际关系。
  “哥哥”“牙医”“和谁结婚”可成立；裸露的已婚状态、名人身份、公司名和物品
  所有格不自动成立。职务字段与具体职业称谓独立判断。
- `definition_word`：被明确询问词义、定义或概念解释的目标。普通事实、外貌、
  规格、拼写问题不算；未明确方向的describe或usage请求保留歧义。明确的
  “define this”“this word means什么”可有指代目标，无需恢复其身份。

| 参考表达 | 规则下判决 |
|---|---|
| is the celebrity single or married | personal_info=true，relation=false |
| who is Jennifer Aniston married to | personal_info=true，relation=true |
| send email to my brother | relation=true，personal_info=false |
| who assassinated Lincoln | personal_info=false，relation=false，definition_word=false |
| what shape is a ball | definition_word=false |
| describe what a chair is | definition_word=true |
| how to spell the word treble | definition_word=false |
| look up etymology in google | definition_word=unavailable：问词义还是主题检索不明确 |

这不是正则标签器或样本专用Q。规则作为任务定义提供，模型直接从音频判决。
上游50/51/146/148/192段提供联系字段、亲属角色及forensic定义目标示例；原生
definition_word还包含外观/拼写，personal_info也有企业联系信息。新语义范围须独立
发布；55建议中17项与native投影不同，不称17个官方错误，也不修改原生评测。

## 交付与证据

阅读原生三类阳性并集432来源/1505录音，以及固定regex选出的57条不在该并集的
对照来源，总计489。各类原生来源：definition_word182、personal_info62、relation198。
形成45来源55参考判决：30真、19假、6未决；227个录音判决绑定，181条独立录音。
首批16来源的3个相关未决项按规则重判。全部是同一作者参考文本建议，未独立语义
复核或听审，优化准入0。

从原始metadata及完整学习人口重建来源、原生字段、target hash和全部绑定；新版本
55定义/55政策/89边界一致。3类和7条相邻边界修改，其余52类定义逐字段相同。
当前入口为[slot-consolidated-design-v6.json](../../configs/tasks/slurp/slot-consolidated-design-v6.json)，
类型v8/政策v8/边界v9。此前news/event/work/property规则保持。

- [任务规则](../../configs/tasks/slurp/attribute-relation-boundary-resolution-v1.json)
- [参考判决](../../configs/reviews/slurp-attribute-relation-boundary-v1.json)
- [身份及绑定核验](attribute-relation-boundary-check-v1.json)
- 共享证据：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/attribute-relation-boundary-evidence-v1`。

## 下一步监督迁移的真实范围

将历史工作包v3/v3/v4与当前v8/v8/v9按实际输入语义字段比较，结果见
[迁移影响清单](../../configs/reviews/slurp-current-rule-migration-scope-v1.json)。
16类自身定义/政策发生变化；计入相邻边界后42类需要明确重审。13类语义字段保持
相同，也不能因此自动获得从未做过的语义认证。

完整9192来源×55类共505560个判决坐标，其中386064个坐标涉及上述42类。
这是覆盖规划，不是完成量；不能把旧标签机械复制为新标签，也不能只审核已知
未决项。首批16来源已有建议中需重审672个坐标，包含原先正、负和未决三类。
各边界对照文件覆盖了其中一部分，但尚未合并或认证，不重复计数。

下一步先将所有规则版本统一到当前监督工作包，完成受影响坐标重审与分歧处理，
推进完整人口有效监督和意图映射；然后才进行输入/上下文、标准RL实现及工程冻结。
当前没有新版processor/NPU验收，没有新训练派发。本轮仅直接核验结构、hash和
迁移作用范围，没有修改生产训练实现。
