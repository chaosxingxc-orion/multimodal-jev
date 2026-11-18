# 统一槽位定义草案与完整既有裁决复核

2026-11-12。将请求相关范围、独立多标签判决和五类角色规则合并为一套完整草案。
入口：[slot-consolidated-design-v1.json](../../configs/tasks/slurp/slot-consolidated-design-v1.json)。
这不是canonical task发布；范围仍为推荐工作假设，未声称用户已确认。

## 本轮集中交付

- 55类型卡v3、55存在政策v3、89共同边界v4统一引用当前规则。
- 历史native边界争议保存在historical字段，不作为当前语义任务互斥优先级。
- 与角色规则有关的卡片把规则与限制写入thinking_requirements和role_rules；不能
  只读取definition短句就宣称完整消费了本体。入口列出runtime与provenance字段。
- household/place与food/ingredient两条共同规则直接改写为本轮明确角色边界；其他
  类型区别保留。原生频率和未共标不建立互斥；真实角色证据不明仍留未决。

其余历史争议被归类为标注兼容性证据，不代表逐项语义覆盖已经完成。独立谓词允许
同词段多true，也不允许用上位类型自动推导标签、用双true掩盖歧义或随意关闭mask。

## 38项参考裁决全部重新对齐

重读16来源的完整参考句，并检查38项已提出的类型判断。5项采用上轮的明确建议，
原31项明确建议均与当前规则兼容，2项真实歧义保留。此次全部定义/政策hash绑定
当前完整版本，不再是新旧规则混合计数。

| 维度 | 结果 |
|---|---|
| 语义建议 | 21 true、15 false、2 unresolved |
| 与native关系 | 28一致、8不同、2未判 |
| 录音绑定 | 185项，涉及78录音 |
| 正式准入 | 0 |

这是同一agent的参考语义复核，不是第二标注者确认或音频听审。8项不同不自动认证为
native错误。旧兼容TSV仍以conflict记录不同值；导出的语义层分开保存建议/native/准入。

共享产物为`slot-reference-consolidated-v1`和`semantic-reference-consolidated-v1`。
全量`annotation-workpack-consolidated-v1`保留40,513录音、9,192来源、1,063参考裁决；
其中意图参考仍使用既有意图定义。只有606来源有任一参考裁决，不将重建等同完成标注。

## 核验与仍缺的证据

独立脚本核对55/55/89数量及引用、全来源—录音身份多重集、38项定义/政策hash、185条
录音绑定和全部产物hash，均通过。见[核验记录](consolidated-slot-design-check-v1.json)。
没有改训练代码、旧gold或旧发布，没有启动模型任务。

后续集中在全量监督、意图定义收敛、独立分歧裁决和统一评测协议。完整类型草案并不
代表所有输入的真值已知，也不代表新增规则在processor/NPU端已验收；工程冻结和
训练应继续遵守任务先行顺序。
