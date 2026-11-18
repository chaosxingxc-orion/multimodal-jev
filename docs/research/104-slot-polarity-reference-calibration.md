# 全槽位正负参考与固定面板校准准备

本轮不改slot包v14的任何定义。先按当前契约重算覆盖，发现旧current-rules-v13仅
3类双极、4类仅正、3类仅负、45类两者皆无，30正/27负坐标。旧规则下的参考覆盖
不能继承为当前有效监督。coverage-v9及独立check-v9保存了修订前的完整计数。

## 新参考与完整迁移

阅读当前55类型定义和候选真实文本，重新写55正/55负，共110项参考，涉及60来源。
历史正例及原生类型只用于定位候选，不自动继承标签。尽量使用相近角色对照，包括
书作者/歌曲作者、作品指称/开放选择、日历日期/时钟、公司股价/交通服务、联系人集合/
邮件文件夹、产品品牌/组织角色。少量类别使用明确无关负例；不是全人口代表性抽样。

参考文件 `configs/reviews/slurp-slot-calibration-anchors-v1.json`，选取协议
`configs/reviews/slurp-slot-calibration-selection-v1.json`。same-author reference-only，
没有独立语义复核、复听或训练准入。已有明确/未知/冲突证据不得用这些新建议覆盖。

从原metadata、official-clean learning与source target独立重建110参考的全部917录音
绑定，涉及358不同录音。每类恰好1正1负、55类身份和引用hash通过；见
`docs/research/slot-calibration-anchor-check-v1.json`。

使用原迁移器合并，发布共享`annotation-workpack-current-rules-v14`，保持40513录音/
9192来源，所有旧评估原样保留。6021评估、4974坐标，当前161明确（81正80负）、
10未决、4803待重审、500586未评。未出现当前新旧参考冲突不等于语义已获认证。
全部55类现在有正负参考，但其中49类仅有1个正参考来源，不足以支持质量或泛化结论。

独立完整多重集、语义兼容性、来源/录音绑定及55类计数复算通过，相关11测试通过。
证据`current-rule-migration-check-v14.json`及`current-semantic-coverage-check-v10.json`。
当前workpack report SHA256：
`a6e74b01961911ac6c1e10f32fa3d08e3cd0923df40b8192cbadf705e45ca14f`。

## 盲校准输入

已准备共享`slot-teacher-calibration-v1`，110个目标判决按相同来源/固定面板合并为
89请求；全部完整返回时700判决，仅110个指定目标可对照参考，其余仍是未评分建议。
完整复用shared-v2面板，所有相邻规则保留；期望答案/历史理由与actor外离线绑定分开，
请求wire只有定义、问题、参考文本和输出协议。未按当前样本裁掉面板中的其他问题。

独立重建全部参考hash、完整面板、wire字段和request身份，证据
`slot-teacher-calibration-check-v1.json`；packet report SHA256：
`4688960b722bbd3be012fee109cb607eada9098724be8e356097d345872ada17`。
本轮尚未发送这89请求。先收尾原122请求意图校准、分析格式与质量，再登记槽位有限
预算与执行计划；不把准备完成登记为请求完成。

新语义监督准入保持0，原生数据不改、不启动训练、不据参考比例设置类别权重。
下一步是校准和独立语义审阅，不再用追加规则版本代替全量监督推进。
