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

本次交付是数据转换；现有原生 SFC 推理队列仍用原生格式。
后续 adapter 须将一个决策单独渲染给 Omni：保留官方输入函数定义、环境、
有序音频和已提供历史，明确说明当前回答采用该决策的输出契约，覆盖原文的
call-string 输出格式要求。不能原样拼接两个互相冲突的输出指令后认定模板已验证。
完整 context 保留候选菜单，因此本版本不能宣称候选隔离或 IIA 已成立。

Noul 先用 JSON 布尔输出测量；只有实际验证 logit 读出后才能称为原生 typed 推理。
每个问题不读取其他预测答案；合并器显式报告 present/absent/calls 一致性。
typed_call_list 先测格式、参数 schema 和完整调用 multiset 准确率，
再序列化回官方 call-string 格式，以官方评分与严格评分并列横向比较。
不得把格式修复后的成绩冒充原始生成成绩。
当前 `reconstruct_calls` 检查有限 JSON 类型及分支一致性，并非完整 JSON Schema 校验器；
推理评分接入前还须实现参数 membership、required、递归类型和 enum 的预测校验。

全量 gold 顶层参数审计只用于转换正确性，不能用于选择提示词或输出策略。
现有审计未递归验证数组元素 schema，不将其描述为完整 JSON Schema 合规证明。
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
