# 请求相关角色：完整范围草案与旧裁决复核

2026-11-12。异步偏好问题尚无回复；本轮采用推荐的request_relevant_role作为可撤回
的设计工作假设，不将等待时间视为用户同意，不将草案当任务发布。既有整句提及文件
和主任务v2保持不变。独立草案入口为
[迁移配置](../../configs/tasks/slurp/slot-scope-migration-candidate-v1.json)。

## 完整本体对齐

已阅读全部55类型卡及89条共同边界规则，分别发布：

- `type-definition-candidates-v2.json`：55定义显式要求服务某个已表达请求，包含
  约束、目标描述及明确要求传达的消息内容。单类原有歧义继续保留。
- `slot-presence-policy-candidate-v2.json`：55项政策统一请求相关范围，explicit_filler
  不变；否定/取消/排除不等于类型消失。不给gold意图，不按native Final筛范围。
- `type-boundary-rules-candidate-v3.json`：89条规则保留角色区别和未决约定，增加双方
  各自满足相关性的前提；共同出现例不自动准入无关背景。

定义中的通用整句检查要求同步调整，device/order/transport三类的背景措辞单独修改，
避免一处说排除无关背景、另一处又要求全部背景存在。原生标注未修改，旧本体哈希
引用及参考证据保留；新配置全部optimization_allowed=false。

## 逐来源重读38项既有槽位裁决

重新阅读16条完整来源文本，按新范围记录逐来源理由，再绑定新definition/policy hash。
新TSV为`configs/reviews/slurp-slot-decision-reference-v2.tsv`；不是仅改文件版本沿用旧
真值。此轮是同一agent的参考复核，不声称独立标注者裁决，也没有听审音频。

结果仍为16正、9负、7unavailable、6conflict。38项数值均未因范围变化翻转：已有候选
角色均与请求相关，原先缺失的时间/地址等仍缺少表达，而house/place、food/ingredient、
代词角色等边界不因相关性变化得到解决。6项冲突继续明确保留，不原位修原生gold。

共享`slot-reference-request-relevant-v1`包含38裁决、185判决—录音绑定，涉及78录音。
另一脚本逐条核对新旧文本/metadata/值、不同政策与定义hash，以及全部绑定多重集。
验证的是身份与复核记录的完整性，不是独立语义正确性。

## 全量工作包重建

`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/annotation-workpack-request-relevant-v1`
已用新类型/政策和重新复核的槽位证据生成，完整保留40,513录音、9,192来源及1,063项
参考裁决；606来源有至少一项参考证据。与旧包相比，全部来源—录音身份多重集和
意图裁决完全一致，没有新训练标签，没有把未审核项填false。

该工作包的task_contract仍引用v2逻辑算子，仅有本包ontology中的类型和政策采用新
范围；因此它是明确的设计分支，不能作为canonical task v2运行发布。后续任务定稿时
须一次性统一入口及hash，不能在训练时混用两个政策。

证据见[独立核验](request-relevance-migration-check-v1.json)。本轮复用已有构建/绑定
工具，没有修改训练代码。接下来需解决类型重叠和全量监督覆盖，尤其不能将这38项
复核或范围对齐冒称新任务全部数据已经就绪。
