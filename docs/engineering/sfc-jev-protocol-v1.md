# SFC-Jev v1：多轮音频、独立判断与有类型的调用参数

## 目标和边界

从完整 SFC 官方划分构造 Jev 任务：train_id 5534、test_id 2590、test_ood 771 条。
保持原始会话分组和有序音频；第二轮保留官方提供的第一轮 assistant 内容。
这是 supplied-history 离线评测，不是模型自行执行工具的在线交互。
所有任务均直接消费音频与定义，不使用当前参考转写或文本模型中转。

依据 nga `61f244ff9ac0409b6a2a59e4563d4e4e6afdef72` 的
`proposal/00-research-proposal.md` R2 和
`docker/engineering-base/source/native/anyjev/question.py`：
可同时成立的函数需求采用独立 Noul，不能把多个调用压成单个互斥 choice。
nga 原语是 choice/noul/score；本协议的 `typed_call_list` 是本项目扩展，
不是已被 nga 引擎支持的原语，也不能直接塞入当前仅支持 list[str] 的 open_slots。

## 输入与输出

每个原始样本生成一个 bundle。`input.context` 原样保留原始 messages 和 audios；
`input.decisions` 仅从输入中列出的函数定义生成，每个候选固定三个问题：

| 后缀 | 类型 | 目标 |
|---|---|---|
| `.present` | noul | 当前请求是否需要至少一次调用该函数，布尔值 |
| `.absent` | noul | 当前请求是否完全不需要调用该函数，布尔值 |
| `.calls` | typed_call_list | 该函数全部调用的参数对象列表，无调用则 [] |

编号 f00/f01 等按该条输入的函数顺序分配；函数名、描述、参数 schema 随定义提供。
候选数量、问题数量、参数 schema 都只依赖输入，不能根据 gold 选择要问的函数。
不先用 gold 决定正例，也不因预测 present=false 而跳过 calls 问题。

例如函数 f 的两次调用可输出：

```json
{
  "f00.present": true,
  "f00.absent": false,
  "f00.calls": [
    {"x": ["Monday", "Friday"], "n": 2, "person": "NAN"},
    {"x": ["Monday", "Friday"], "n": 2, "person": "NAN"}
  ]
}
```

必须保留重复次数、数组顺序、原始数值类型及同次调用的参数绑定。
不把不同参数分别摊平成集合，否则会把同一函数多次调用的参数交叉配对。
`NAN` 是官方缺失参数字符串，既不是浮点 NaN，也不是 null 或省略字段。
这是对原 schema 的显式 sentinel 扩展；不能直接用未扩展的 JSON Schema
拒绝整数/数组参数上的 `NAN`。省略、null、空数组、NAN 互不等价。

完整 targets 文件单独保存答案、原始 target/provenance 及 request SHA256。
调用列表可以恢复原始调用 multiset；跨函数的原始输出顺序与字符串格式保存在 provenance。
调用顺序若未来承担执行依赖语义，须另建有序计划协议；不能将本版本宣称为交互执行轨迹。

## 模型消费和评分约定

数据转换已发布；现有原生 SFC 推理队列仍用原生格式。
`sfc_jev_runtime.py` 已将一个决策单独渲染给 Omni：原样保留输入函数定义、环境、
有序音频和已提供历史，仅替换 TRAIN 中已人工核验并固定 SHA256 的十种首轮前缀，
以及固定的第二轮输出指令；未知指令直接报错，不静默删除。
当前决策只要求一个 JSON 布尔值或参数对象数组，不引入 tools/XML 输出协议。
全量 136722 个决策通过静态映射审计；尚未完成实际 processor/vLLM 的验证。
完整 context 保留候选菜单，因此本版本不能宣称候选隔离或 IIA 已成立。

Noul 先用 JSON 布尔输出测量；只有实际验证 logit 读出后才能称为原生 typed 推理。
每个问题不读取其他预测答案；合并器显式报告 present/absent/calls 一致性。
typed_call_list 先测格式、参数 schema 和完整调用 multiset 准确率，
再序列化回官方 call-string 格式，以官方评分与严格评分并列横向比较。
不得把格式修复后的成绩冒充原始生成成绩。
数据层 `reconstruct_calls` 检查有限 JSON 类型及分支一致性，并非完整 JSON Schema 校验器。
运行评分由 `sfc_jev_score.py` 检查参数 membership、required、递归类型和 enum；
布尔值不当整数，非有限数、重复 JSON 键、Markdown 包裹和尾随文字均拒绝。
schema 只支持明确列出的关键字，遇到未支持的 schema 报工程错误，不算模型错误。
JSON 参数对象采用闭合参数集合；顶层参数可用 NAN，但数组元素不因此放开类型约束。
评分分开记录格式/类型/正确性、正反 Noul 互补性和 Noul/calls 一致性。
所有 calls 分支都独立推理，合并时不根据 Noul 预测过滤。
任意 calls 分支无效则整个原生预测标记无效，不悄悄改成空调用。

全量 gold 顶层参数审计只用于转换正确性，不能用于选择提示词或输出策略。
初版定义审计仅验证顶层；新增 runtime 审计已递归核对全部 gold 与受支持的 schema。
这不是对任意 JSON Schema 方言的完整实现。
实际参数类型是 string/list/int；bool/object/null 只在合成边界测试覆盖，不能宣称数据已包含。

## 不变性、区分性和训练接口的后续工作

v1 发布完整基础任务，不把尚未实现的扩展写成已交付。
下一版增加函数候选置换、无业务意义 ID 重命名及定义重绑定配对；
每个保持语义的变换必须配套改变语义且答案应变的区分性例子。
有历史的重绑定必须同步处理历史引用并可逆恢复，不能只改当前菜单。
学习/选择按完整 TRAIN 会话分组；官方 ID/OOD 测试不参与提示优化或奖励设计调参。

后续 verl 奖励至少区分：有效格式、schema 合规、函数/参数任务正确性、
配对一致性和定义改变后的区分能力。常量 true/false 和恒空调用作为退化基线。
一致性奖励不能替代正确性；不得奖励“全输出 []”获得高表面一致性。
当前冻结基线与教师提示优化优先，尚不启动权重训练。

## 已完成的评分接入验证

`evaluate_sfc_jev.py` 要求所给发布划分中的全部决策预测，按 request SHA256 绑定输入，
输出逐决策分数、合并后的原生调用，再调用原始 SFC 评分程序和严格评分。
每个预测字段为 example_id、decision_id、request_sha256、raw_text。
后处理仅做确定性结构序列化，不补答案或修复非法原文。

四个完整 TRAIN 会话的六轮、78 个决策使用 gold 作为模拟输出，已通过该桥接及官方评分。
记录见 `sfc-jev-native-bridge-train6-fixture-v1.json`；这是评分器 fixture，不能作为模型成绩。
全量静态映射/递归 schema 证据见 `sfc-jev-runtime-static-audit-v1.json`。
下一步在冻结镜像上运行真实波形 processor 校验，再接入受控 TRAIN 推理；
全量任务上下文长度仍须实际测量，字符长度不能替代 token/audio 特征预算。
