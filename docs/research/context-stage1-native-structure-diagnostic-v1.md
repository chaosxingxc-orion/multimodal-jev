# Stage1完整选择集的原生属性与决策错误分析

使用已归档 Stage1 step634 的全部 9,226 条选择集预测、2,146 个来源组。
预测文件重新核验 SHA256，并绑定原审核收据；冻结 clean release 与原生属性目录重新核验。
完整 top1 混淆矩阵及 set-F1、exact-set、macro-F1、NLL、Brier 重新计算，匹配原评测。
属性错误计数又由原混淆矩阵独立投影复算。没有新模型推理、TEST读取或阈值搜索。

## 1. 最终事件错误不仅是属性消歧

| Final top1 情况 | 录音数 | 占top1错误 |
| --- | ---: | ---: |
| 正确 | 7,572 | — |
| scenario正确，仅action错误 | 540 | 32.65% |
| action正确，仅scenario错误 | 144 | 8.71% |
| scenario和action均错误 | 970 | 58.65% |

全部 top1 错误为 1,654 条。Final top1 投影的 scenario/action 正确率分别为
87.93%/83.63%；这不是过程头的预测成绩。单属性错误说明存在可描述的局部混淆，
但不能证明增加属性Q就能修复；两属性均错也不能据此判定为声学错误或语言理解错误。

较多的有向混淆包括 calendar_set→calendar_query 35条、qa_factoid→general_quirky 34条、
general_quirky→qa_definition 29条、calendar_query→calendar_set 26条、
play_podcasts→play_audiobook 22条。它们是描述性结果，不据此按样本挑Q或改标签。

## 2. 候选排序与阈值决策存在明显差距

固定阈值0.5下，空集1,703条（18.46%），单选7,328条，多选195条（2.11%）。
754条top1正确但返回空集，134条top1正确但返回多选；这888条解释top1正确数与
精确集合正确数之间的差额。188条多选集合包含gold。

因此后续RL需要同时报告排序、集合决策及概率质量。不能只看top1或过程准确率，
也不能将降低阈值作为既定解决方案：本次没有阈值搜索、校准干预或优化，正式阈值保持0.5。
原完整结果为 set-F1 73.8041%、top1 82.0724%、exact-set 72.4474%。

## 3. 本体复杂度不能直接充当模型难度

下述分组在观察模型预测前已由冻结目录定义，不依据本次结果重新划分。

| 本体结构分组 | 录音数 | set-F1 | top1 | exact-set |
| --- | ---: | ---: | ---: | ---: |
| 至少一个属性可唯一确定事件 | 5,945 | 71.70% | 80.45% | 70.51% |
| 两个属性均需与另一属性组合消歧 | 3,281 | 77.62% | 85.00% | 75.95% |

需要属性组合的分组并未更难。分组的类别构成、训练支持和声学条件不同，本表不能作因果比较，
也不能代表逐样本推理难度。继续使用“结构分组”名称，避免称简单/困难gold。
JSON中的分组macro-F1沿用全部60类分母，包含该分组无支持的类别，不宜横向解读为难度差。

## 4. 对既定比较的约束

保持 Direct/Q-input/Q-credit 的预登记定义、lambda0.25、同起点/曝光/选模规则；
本次诊断不改变它们。比较时分别检查上述结构分组与完整人口上的Final变化、
空集/多选及NLL/Brier，不能用过程分数上涨替代Final改善。
当前预测来自原简单模板，不是紧凑Direct或Q模板的step0；新臂仍需各自真实推理起点。

实现：`tools/reports/analyze_context_stage1_structure.py`。
逐记录结果及身份：`context-stage1-native-structure-diagnostic-v1.json`。
未改变训练、过程reward或任何运行包。

## 5. 同时发现的独立探针问题

逐条请求身份核验揭示旧复杂任务parity探针误用带不同JSON序列化规则的content_hash，
与冻结FixedContextRollout的compact wire hash不一致。旧替身测试双方用了同一错误函数，
不能证明实际传输匹配。旧探针/405文件包保留且标为不可派发；406文件v2包新增正确序列化，
原冻结src和CPU包不修改。全部9,226条实际已归档vLLM请求身份通过v2、均被v1拒绝。
另补真实FixedContextRollout._score路径测试，仅engine使用替身。
见 `context-attribute-parity-wire-identity-fix-v2.json`。
这修复独立诊断，不影响正在运行的GRPO；仍未执行新紧凑请求的Qwen/NPU对齐。
