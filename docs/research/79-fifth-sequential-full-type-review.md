# 第五批全类型参考判决

固定来源区间[64,80)共16来源的55类型参考审核完成：28真、840假、12未决，
3080个录音×类型绑定、56条录音。继续使用slot包v8（v10/v10/v11），未改本体。
所有负例为全文逐类型缺失判断，不从native未标注生成；全部为同一作者参考建议，
未听审或独立语义认证，正式准入仍为0。

本批跨样本保持以下规则：

- alarm作为设备还是告警配置，6019、7472、1634均与上一批3985一样保留未决。
  不是每次出现alarm都自动给device_type或alarm_type。
- on repeat明确设置循环，支持player_setting及general_frequency；与先前
  repeat reminder是否只重复一次的歧义区分，不把所有repeat都变成周期。
- grocery list中的tomatoes是食物，但没有烹饪成分关系，不自动给ingredient。
- current song是特定作品引用；what music is this未明确个别歌曲类型，仍mask。
- next two hours为时长窗口，Friday night为日期和日段；不把它们混为同一时间谓词。
- for clock为不清楚的原始表达，不根据习惯或原生标签修成four oclock。

门/家居表面、浪漫灯光是否颜色、contacts集合/人类指代、泛指个人信息以及
裸sandle的天气关系等未决统一登记，未逐条增加本体版本。

交付：

- [参考判决](../../configs/reviews/slurp-full-slot-sequential-reference-b0005.json)。
- [问题清单](../../configs/reviews/slurp-sequential-review-issues-b0005.json)。
- [判决与绑定核验](full-slot-reference-b0005-check-v1.json)。
- [全工作包重建核验](current-rule-migration-check-v7.json)。
- [逐类覆盖独立复算](current-semantic-coverage-check-v3.json)。

当前包`annotation-workpack-current-rules-v7`保持40513录音、9192来源、原始顺序
和全部历史评估。5673评估覆盖4720坐标：3632当前明确、61未决、1027需重审，
500840尚未评。报告SHA：`f2bface49d8198d75b3b9526dcbc7dae01d3e65dfe23d1672abec7f1aab20998`。

当前明确参考为199正、3433负，33类型双极性、22类型仅负；10类型仍仅一个正来源。
逐类覆盖属于非随机参考集合，不能当作全人口先验、模型准确率或正式训练权重。
独立从原metadata/learning/targets重建全部绑定及评估多重集，复算语义兼容和55类
统计通过。生产代码未变，未重复运行无变化测试；结构核验不认证语义。

下一固定来源偏移80。范围候选未应用，任务包仍待完成；本轮未启动模型任务。
