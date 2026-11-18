# 意图校准结果与未决语义

启动后1359秒按约定间隔观察，PID3830498已不存在、总终态记录122/122齐全，未重启。
原始响应均保留在`.local/research/intent-teacher-calibration-v1/run/`。
评分产物为共享`intent-teacher-calibration-observed-v1`。

| 项目 | 结果 |
|---|---|
| HTTP/终态 | 122响应，115 stop且完整、7 length |
| 明确参考 | 130目标，127完整且全部一致，3请求失败；全目标一致率97.69% |
| 正参考 | 65/65返回且一致 |
| 负参考 | 62/65返回且一致，3失败 |
| 歧义参考 | 16目标，12完整；4未决一致、4判正、4判负、4失败 |
| 使用量 | 472554输入、144151输出，总616705 token |

7个失败均耗尽4096上限，不能自动恢复为负例或重试。所有失败目标保留分母。
独立核验全部实际请求wire、原始response的ID/状态、终态及逐目标计数和usage，见
`intent-teacher-calibration-observed-check-v1.json`。report SHA256：
`b5f298813048ae783f4cad1c69d5a82abbdc84c6d693b8b35f4c173c2b840287`。
这些是与选择过的作者参考一致率，不是独立gold准确率、全人口质量或音频模型效果。

## 八项分歧已阅读

四项涉及明确的“当前定义排除”与“参考未决”冲突：edit list→list_create_add；save my
opinion→music_positive_preference；add email/add this new email→contact_record_modify。
教师负判有当前excludes文字依据，不能简单归为教师错误。需统一“未表达目标角色”与
“该角色可能存在但语义未确定”的三值政策，再决定修参考还是修定义，不能按教师多数投票。

另外四项教师选择了具体读法：contact location→地址、save this what a nice song→
保存评价、make sure bread is on my list→添加、help me book→执行预订。现有参考分别
保留位置含义、保存对象、核查/确保、指导/执行歧义。这些须按共同政策复核，不因可行
读法存在就直接准入positive。

逐项记录在`configs/reviews/slurp-intent-calibration-disagreement-triage-v1.json`。
本轮没有修改定义或参考标签，全部resolved=false、semantic_admission=0。
下一先解决四项定义/参考冲突，再复核歧义，保持原校准版本用于比较；槽位89请求仍仅
准备，不借意图明确参考高一致率自动放行全量标注或训练。
