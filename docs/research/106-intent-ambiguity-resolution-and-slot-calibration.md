# 意图歧义条款收口与槽位校准执行

上轮122请求意图校准已终态。本轮逐项阅读8分歧、现有定义及共同政策，并阅读learning
中固定词面命中的28个相关来源。未查询测试集、未按教师结果调整原校准分数。

## 三张卡澄清

共同请求政策已有“真实语义歧义保留未决”；集合政策operation_extent已明确未指定
编辑种类为未决；feedback_scope与非请求政策要求未知意见/评分极性保留未决。
但三张卡的excludes容易被解释为一律negative。本次只澄清：

- list_create_add：明确重命名/排序不是添加；未指定edit仍未决。make sure membership
  没有消歧时不强选核查或添加。
- contact_record_modify：查找/发送/删除等明确不匹配操作为负；add email未明确消息/
  地址及联系人目标时未决；add email to contacts可正，不要求先给地址值。
- music_positive_preference：保存内容但没有评价为负；明确提到意见/评分而极性未给出
  为未决。“不足以判正”不等于明确判负，不推理存储内容的私人真实态度。

目录v5及contact候选v2发布；62张卡与共同request/nonrequest/boundary政策完全不变。
contact_information_lookup、music_feedback_record、transport_passage_acquire的现有
政策足以保留contact location、save this what a nice song、help me book的歧义，不新增
专用样本规则。8项均维持未决，见author-resolution-v1；resolved仍false，因为这只是
作者复核而非独立审阅。旧triage/参考/校准结果完整保留。

新增24来源24建议（5正6负13未决），补充明确正负对照以及未用于本次校准的类似表述。
所有建议是看到校准后作出的研究修订，不称未见泛化或独立gold。记录于
`configs/reviews/slurp-intent-ambiguity-reference-v1.json`。

## 清单与输入迁移

intent-presence-inventory-v4保40513录音/9192来源/65卡，1286评估/1266坐标；442正、
551负、108未决、165需重审、596214未评。65卡均有正负参考，181历史评估因卡identity
变化不再继承；重复评估保留，不投票或最新覆盖。独立重建原metadata/全部录音绑定、
原评估多重集、历史与当前契约和所有来源/录音状态，20相关测试通过。
证据`intent-presence-inventory-check-v4.json`，report SHA256：
`de18e1ab8779a00bf6a938d91f032d1b591b8fc683b928e49a061fe29ccbf65d`。

literal/shared-v3完整工作包已重建并独立核验9192来源/40513绑定/147072日程。
相对v2仅I02/I04/I05 ontology改变，全部7槽位面板及来源绑定逐项相同；正在运行的
槽位校准无需重发。实际builder的INTENT_FIELDS含ambiguity，新文字已完整投影到输入。
shared-v3 report SHA256：
`890f4b4d5a909aa81389cc5a0c6cec79dc0969d940fa8b9c8f7ed9db1f9efb59`。
完整证据见`blind-semantic-annotation-check-v3.json`、`shared-annotation-check-v3.json`
及`blind-semantic-annotation-delta-v3.json`；结构验证不代表NPU上下文或模型理解通过。

## 槽位有限校准已派发

沿用上轮核验的slot-teacher-calibration-v1固定89请求，全部完整返回时700判决，
只对110指定正负参考计分。独立会话PID3831451，启动时检查存活；launch位于
`.local/research/slot-teacher-calibration-v1/launch.json`，run/保存逐请求意图、hash、
PID、响应、终态及总心跳。该启动记录不是当前实时占用声明。

复用既有pin执行器与配置，concurrency2、每请求4096输出、累计364544上限、网络60秒/
总180秒、不自动重试；缺失/截断保留完整分母。输入token没有官方tokenizer全量测量，
不把S00的23013当全部面板上界或费用估计。按600秒间隔检查，不逐分钟轮询。
本轮未宣称请求完成。终态后可复用严格scorer：

```bash
.venv/bin/python tools/jev/score_intent_teacher_calibration.py \
  --packet /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/slot-teacher-calibration-v1 \
  --run .local/research/slot-teacher-calibration-v1/run \
  --output /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/slot-teacher-calibration-observed-v1
```

scorer的历史schema名称带intent，但实际按packet的question IDs验证，不推断任务域。
报告必须明确是槽位参考一致率。新语义准入仍0，未启动训练；下一步独立审阅与全量
监督方案仍须完成，不能用反复规则修订或小参考集替代正式任务数据。
