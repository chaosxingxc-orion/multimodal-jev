# 当前语义参考的逐类型覆盖与训练均衡缺口

对current-rules-v5的全部9192来源、40513录音、55类型重新统计；统计单位是来源×类型，
重复审核不增加独立样本数。当前3913项评估覆盖2962坐标：1897明确参考中有
133正、1764负；38当前未决，1027需按当前规则复审，502598尚无参考。
全部新语义监督正式准入仍为0。

| 当前明确参考极性 | 类型数 | 含义 |
|---|---:|---|
| 正负都有 | 24 | 有参考证据，不等于足够覆盖或已认证 |
| 仅负例 | 20 | 不能通过重加权产生正类学习信号 |
| 两种都没有 | 11 | 可能是既有证据因相邻规则变化失效，不代表数据没有该类型 |

两种都没有的类型是business_name、business_type、currency_name、definition_word、
house_place、music_genre、news_topic、player_setting、playlist_name、time_zone、
transport_descriptor。它们需要明确当前规则复审，而非把历史参考直接改成当前有效。
仅一个正参考来源的类型是email_address、email_folder、ingredient、meal_type、
music_album、relation、song_name。同一来源的多条录音不增加独立语义证据数量。

若只在当前明确参考上恒答false，表面一致率为1764/1897≈92.99%。这不是全数据
准确率或模型指标：参考集合包含定向边界审议，且大部分来源尚未覆盖；不能用其
估计真实类型先验或生成训练权重。这个数只揭示聚合正确率可能掩盖正类完全失效。

## 对训练设计的直接约束

- 意图外层类别均衡不能代替每个S谓词的正负覆盖。现有sqrt_cap4和Stage1 cap8
  保持候选，不能从本次参考统计导出正式数值。
- 不靠按gold选择正问句改善数字；固定菜单与完整训练人口保持不变。
- 先集中收敛公共定义，再按来源补齐覆盖和复审；定向少数类证据用于发现边界，
  不作为尾类小集替代主训练人口，也不将定向样本比例当作训练分布。
- 新语义准入后才能重算实际mask、加权极性质量及来源集中度，冻结CE系数。
  数据量、正例来源数、来源组数与录音数分列，避免多录音制造虚假的证据规模。
- 过程指标必须包含正类召回、误报和有效监督覆盖；仅reward或总体准确率上升
  不能证明复杂判决链改善。主任务Final仍需独立指标和固定分母。

## 实现与验证

新增[audit_current_semantic_coverage.py](../../tools/jev/audit_current_semantic_coverage.py)，
输出每类来源状态、正负来源数、录音完整覆盖、来源组数和正参考集中度。
同一录音的所有source IDs必须均有当前明确且一致的参考才能计为正/负参考录音；
缺标、分歧和未决不补false，不按source_group传播标签。reference与admission完全分开。

5项测试覆盖旧版与当前意见冲突、未决共存、多来源缺标/冲突、单极性及重复绑定/
意外准入拒绝，全部通过。另从原metadata/learning/targets重建全量来源绑定，使用
独立集合运算重算55类统计及录音共识、集中度，通过；未重复运行未受影响的NPU测试。

报告：共享datasets下`current-semantic-coverage-v1/report.json`，SHA
`461484de065fc6da3a9e625a4f83ebfed1fcb685fc990d4586b5027c34e08851`。
[独立核验记录](current-semantic-coverage-check-v1.json)保留人口和极性计数。
本轮不改变当前规则、工作包或训练参数，不启动模型任务；范围候选仍未应用。
