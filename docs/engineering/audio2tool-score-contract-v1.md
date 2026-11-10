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

## 结构化主目标与附加调用覆盖

已完成scorer-only目标投影 `audio2tool-targets-v1/targets.jsonl`，全部30733条与actor
输入release身份/音频路径逐条对齐。转换器固定八份release字节SHA，目标文件留在
共享实验目录并限制权限，不提交仓库、不传给actor或教师。

官方BaseDataset的all_expected_tools实际是 `[tool_name] + additional_tool_calls`，
evaluator用结构化tool_name/extracted_params评分主调用。投影因此保留该主目标，
原expected_tool_call另存，记录可解析时的完整调用一致性，冲突不被抹除。
这里的一致性包括工具名，故与上表仅参数一致计数不同。

Tier3附加标签5810项、Tier8共1259项，全部经固定官方normalize_tool_name对应独立
registry。它们既有裸工具名也有调用字符串；原始字符串/顺序/重复保留供官方工具
指标使用。仅严格有限JSON字面量调用可提供附加arguments；裸工具名和不支持语法的
参数标记None/unavailable，显式 `tool()` 才对应已知空字典。

全部目标中25969条有完整可投影调用参数，4764条不完整（Tier3 4002、Tier8 762）。
这些是编码覆盖，不能解释为标签正确性。完整集合的严格多调用参数EM目前不可定义；
不得去掉4764条后仍称全量EM，也不得把None换成{}。全量主调用/工具名称仍可按官方
字段评分；若另报参数可用子集，必须标明覆盖和独立分母。

目标工具新增8项测试，与评分核心共19项通过。全量目标哈希、顺序/身份以及未知
标志已核对，见[目标投影结果](audio2tool-targets-v1-verified.json)。评分核心上一节
的evaluate仍要求完整gold calls，不能直接喂入此投影中的未知full_calls；后续需接入
明确的主调用/名称评分入口，不把缺失参数伪装成完整目标。

## 全量投影目标评分入口

`evaluate_projected` 现已接通上述目标：主调用exact/参数F1/工具accuracy及工具集合
exact/F1使用全部声明记录分母，同时提供类型严格主调用exact和有序名称exact。
官方指标调用固定原函数，仍标明本地JSON解析及全错误分母差异。
每个tier及总体分别报告记录数、格式有效数、runtime error数。

目标覆盖声明会重新核验：原始附加标签、归一化名称、显式字面量参数与full_calls
必须一致，unknown不得替换成{}。未知参数不妨碍全量名称/主调用评分；完整参数
子集只在独立 `full_arguments_subset` 下报告分母与严格exact。只要有一条目标参数
不可用，`full_population_strict_calls_exact` 即为None，不将子集分数冒充全量。

26项目标/解析/评分测试通过，并实际运行全30733条两个合成控制：
all_runtime_failures所有主指标为0且runtime error分母30733；primary_only_oracle
严格主调用为1。两者均保持25969可用/4764不可用、全量完整调用exact为None。
合成oracle只用来检验评分身份映射，不产生actor反馈，不作为模型成绩，也不落盘
伪模型预测。见[全量合成控制](audio2tool-projected-score-controls-v1.json)。
正式评分仍须先验收真实模型预测容器、输入/输出覆盖和失败账本；当前没有Audio2Tool
模型输出，不能用这些控制结果替代冻结基线或泛化评测。
