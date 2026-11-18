# Cooking完整学习来源的定义边界审核

2026-11-12。读取原生cooking_recipe与cooking_query在已发布learning人口中的全部
167条参考文本，关联681录音。范围由原生类确定，不按文本内容挑例；不是全数据集
所有潜在烹饪语义的检索。没有读取selection/test样本，没有音频复听。

证据包：共享`slurp-jev-benchmark/cooking-cohort-evidence-v1/packet.json`。
逐来源判决：[slurp-cooking-definition-fit-reference-v1.tsv](../../configs/reviews/slurp-cooking-definition-fit-reference-v1.tsv)。
发布绑定：共享`slurp-jev-benchmark/cooking-definition-fit-v1/`。
[独立覆盖与绑定复核](cooking-definition-fit-check-v1.json)确认681录音完整对应学习类人口，
167个来源全部有审核行，所有录音绑定的判决hash一致。独立程序复核不等于第二位语义审稿人。

## 审核问题与结果

审核的是是否符合现有cooking_recipe卡的“请求食谱或实际食物制作指导”，不是是否
属于原始互斥类别。保留supported/not_supported/unresolved三态，不把定义拟合当作
已验证的I_class正负标签。

| 参考文本判决 | 来源数 | 对应录音数 |
|---|---:|---:|
| 支持当前定义 | 142 | 577 |
| 不支持当前定义 | 7 | 29 |
| 边界或上下文未决 | 18 | 75 |

原生cooking_query仅2来源/2录音，均支持cooking_recipe候选定义：
13479请求压力锅鸡腿食谱；13488询问曲奇中用红糖替代白糖。
原生cooking_recipe的165来源中140支持、7不支持、18未决。

不支持的7条分别为：每日蔬菜摄入量(10064)、芦笋季节(10075)、下载烹饪app(10140)、
jalapenos含义(10183)、recipe一词含义(10211)、食品家庭企业的陈述(10221/10222)。
这表示与当前“食谱/制作指导”定义不符，不是自动宣布原始数据错误，也没有推断它们
属于哪个其他类别。若修改定义吸收营养/词义/app下载，会引入新的跨类重叠，不能只为
吻合这批gold而扩大卡片。

18条未决包括freezing/good food等片段、菜肴成分与一般知识的边界、配餐、实际做饭
请求与求指导的歧义、陈述烹饪步骤而非请求步骤。定义语义明确前不转false。
原生成员与定义拟合有9条明确不一致（2条query支持+7条recipe不支持），18条未知。
这不是总体错误率，不能外推到其他原生类别或全音频。

## 直接影响任务冻结的处理

1. 原生60类基准、候选数、原始gold保持不变。保留原生复现口径，不合并cooking类
   获得更高分，不删除难例后与原分母比较。
2. cooking_query目前不是可用的独立语义卡：其定义依赖“两条原生样本”且没有可迁移的
   互斥边界。因此该卡不得被当作定义理解训练/评测题；这不是从原生Final中删去该类。
3. cooking_recipe的“定义匹配”不自动成为I_class：即使文本符合食谱请求，也未证明
   60类排他性和主意图优先级。当前绑定明确`exclusive_I_class_truth=null`，全部
   `optimization_allowed=false`，不把新拟合值写回native target。
4. 若选择独立语义存在任务，必须登记I_present/定义满足的真值政策，与native I_class
   分表。这批完整类内审核提供边界证据，但还缺类外潜在正例及可靠负例覆盖；不能以
   “其他原生类”补齐全部false。现阶段不静默切换算子来解除冻结条件。
5. 不将681录音变成缩小版主训练集。主目标仍是全规模SLURP任务；继续解决其他高风险
   定义及完整监督覆盖，统一发布新语义层、mask与人口影响，之后才冻结工程和训练。

本轮验证工具覆盖遗漏/重复审核拒绝、未决不转负例、定义拟合不授权互斥标签/训练、
一个录音的其他来源未审时不声称完整覆盖。3项测试通过。未更改actor/原生数据/已有
定义卡；尚无新增训练和效果结论。
