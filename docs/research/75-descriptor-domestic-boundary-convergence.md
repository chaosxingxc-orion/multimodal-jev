# 音乐属性与家庭控制位置的共同边界

本轮从完整learning人口提取三组证据并阅读全部参考文本：音乐246来源/1080录音，
家庭位置177来源/800录音，陈述模式78来源/341录音，去重486来源。音乐组包含全部
原生music_genre/music_descriptor来源，家庭组包含全部原生house_place来源，
并加入固定词法检索对照；检索用于读证据，不是自动标签规则。另读上游指南154、
161–169、177–183段。没有音频听审。

选择两条项目语义规则，保留同一固定Q算子：

1. music_descriptor表示流派身份以外的音乐属性。仅以country/jazz作为筛选条件，
   不自动再标descriptor；soft jazz、best country songs of the nineties则同时提供
   属性与流派证据。down tempo等词在特定输入中无法区分风格名称与属性时仍mask。
2. 家庭房间仅定位受控设备、环境设置或清洁范围时，不另标place_name；房间明确
   充当会议场所、出行或交付目的地时可以满足place_name。家居区域与地点不存在
   全局互斥；判断的是表达的角色，不是房间词典。

上游示例支持这些区分，但没有证明所有语义互斥。上述边界是明确的项目定义选择，
不冒称恢复了官方唯一真值，也不按原生意图选择问题或裁决。

裸报告范围暂不改变。上游general_conversation包含we had great dinner today，
学习集亦包含偏好表达、独立报告和隐含需求；这说明当前request-relevant范围需要
统一处理言语行为，不能凭一个i went to the park就扩展全部55类型。
source6203的人物/活动/地点仍未决。下一次范围审议应覆盖完整相关言语行为及
隐含请求规则，而不是追加单句例外。

新版入口[slot包v8](../../configs/tasks/slurp/slot-consolidated-design-v8.json)使用
类型v10/政策v10/边界v11：只改变music_descriptor和place_name两张类型卡及其
11条相邻边界，其余53卡逐字段相同。历史版本保留，当前上下文仍未验收。

35来源35项参考建议为15真、14假、6未决，绑定152录音；其中10项与原生投影不同，
不等于10处官方错标。第三批5641/descriptor及2186/place在新规则下给false，
6203/place继续未决。全部仍是同一作者参考建议，正式准入0。

新规则与建议已合并为`annotation-workpack-current-rules-v5`，完整人口40513/9192
和全部历史原文不变。3913项评估覆盖2962坐标：1897当前明确、38当前未决、
1027需按当前规则复审；502598坐标尚无参考。规则修改使当前明确数下降、需复审数
上升，这是显式迁移成本，不能把历史结论直接升级为新规则标签。

从原metadata/learning/targets重建全部来源、证据组及录音绑定，并重建全工作包
原评估多重集、适用性和合并状态，通过核验。脚本复用既有独立重建方法，首次执行
发现适配后的审核路径多了boundary字段，修正该本地核验路径后通过；生产代码未改。
结构核验不构成独立语义认证，没有模型运行或训练。

证据：

- [规则与证据来源](../../configs/tasks/slurp/role-boundary-convergence-resolution-v1.json)。
- [参考建议](../../configs/reviews/slurp-role-boundary-convergence-v1.json)。
- [规则变更与绑定核验](role-boundary-convergence-check-v1.json)。
- [全工作包重建核验](current-rule-migration-check-v5.json)。

报告SHA：`65e76fbcb5ee2628fc79b37dd5bdf7e745f3b4e8e8269adb778ca125814801bc`。
任务及监督计划指向新包，固定顺序审核下一offset仍为48。尚未冻结完整任务，
标准RL接入、学术基线完整复现和迁移收益验证均未完成。
