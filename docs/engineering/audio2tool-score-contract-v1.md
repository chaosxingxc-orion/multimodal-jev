# Audio2Tool 评分接口与目标转换边界

新增 `audio2tool_score.py`，接受已经独立认证的目标调用数组及完整terminal预测记录。
它不打开标注、不生成actor输入。当前仅完成评分核心及人工反例，尚未完成真实发布
目标转换，也没有模型评分结果。

预测必须是原始JSON数组，每项恰有name/arguments。拒绝代码围栏、额外字段、重复
JSON键、NaN/Infinity及溢出浮点数，不自动修复。格式有效不代表工具名/参数schema
有效；schema校验仍须结合registry单独处理。

严格指标保留调用顺序、重复次数和嵌套值类型；bool/int/float/string不同，对象键顺序
不影响相等。另有按调用位置+工具名+参数键绑定的micro slot F1，重排与跨重复调用
参数互换会受罚。这是项目明确命名的指标，不冒充论文slot micro-F1。

并列调用固定上游metrics.py原函数（SHA
`b092cc5ba1bac59d58dbfeb56c7f7cf63f1970df59892ea7b56ee24ac2c86802`）：
primary exact、primary逐样本parameter F1、tool-set exact。它们保留官方大小写/
字符串归一化与集合语义，但采用本地JSON解析及完整记录分母，所以不等于未经修改
官方evaluator的历史结果。原官方evaluator排除error行的问题不在本项目复现。

每个target必须对应唯一terminal预测：显式runtime error和格式错误计入分母且不给
exact/F1信用；缺少预测行直接拒绝评测，不能静默丢弃或伪造运行失败。无slot的整体
micro-F1分母为0时返回None，不能仅靠空输出获得成功。目标解析异常必须在转换阶段
记录，不能把未解析目标替换成空调用。

11项测试通过，其中实际加载固定官方源文件，覆盖调用重排/重复、类型差异、
失败分母、空目标格式失败及缺失/重复预测。没有用真实TEST预测选择评分规则。

## 发布目标字段的不一致

随后按固定release字节SHA解析全部30733条记录，静态检查expected_tool_call能否
作为单个命名函数+关键字+有限JSON字面量解析，并与tool_name/extracted_params比较。
只输出计数，没有输出标注值或调用模型。完整计数见
[目标编码审计](audio2tool-target-encoding-audit-v1.json)。

| Tier | 发布记录 | 可解析单调用 | 参数与结构字段类型严格一致 |
| --- | ---: | ---: | ---: |
| 1 | 4292 | 4292 | 4150 |
| 2 | 6320 | 6134 | 4260 |
| 3 | 4292 | 290 | 148 |
| 4 | 4278 | 1776 | 1210 |
| 5 | 4292 | 3074 | 1586 |
| 6 | 4292 | 3644 | 2694 |
| 7 | 2146 | 1451 | 1033 |
| 8 | 821 | 81 | 67 |

不可按上述单调用规则解析，不等于整个benchmark目标无效：存在裸工具名、非Python
字面量语法或其他编码；不能用一个通用literal_eval强行统一。可解析也不保证与
结构字段一致。旧ReleaseDataset源码实际取tool_name/extracted_params，tool_call
取同名字段而非expected_tool_call；官方evaluator主要使用结构字段计分。

下一步目标转换必须显式固定字段优先级并保存冲突，不能默默以expected_tool_call
覆盖结构字段，或相反将不一致擦除。additional_tool_calls的名称/参数编码还需单独
核查后接入完整有序目标。现阶段不宣称目标转换、官方基线复现或Jev评测已完成。
