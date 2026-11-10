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
全量 136722 个决策通过静态映射审计；六轮 TRAIN 的 78 个决策已通过实际 CPU
processor 验证，vLLM 引擎内部处理和生成仍待验证。
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
冻结镜像上的真实波形 processor 校验已完成，证据为
`sfc-jev-omni-processor-train-v1-result.json`。四个完整 TRAIN 会话的六轮、78 个决策，
输入 689–1504 tokens，全部满足 4096 上下文加 1024 输出预算。
两个双音频请求交换特征的最大误差均为 1.1920928955078125e-7；
同一音频不同决策的特征完全一致，逐条帧覆盖检查通过。
vLLM adapter 的原始 prompt/波形引用与 HF 输入一致，但未启动 vLLM 引擎或加载模型。
容器退出码 0、应用退出码 0；无 NPU 映射，未读取 target 文件。
下一步接入受控 TRAIN 推理，验证引擎实际 token IDs 和生成路径。
全量任务上下文已另行完成实际CPU测量，结果见下方全量上下文检查。

## 受控 TRAIN 推理队列

`sfc-jev-qwen3-omni-train78-v1` 已准备并启动等待调度器，记录见
`sfc-jev-qwen3-omni-train78-v1-queued.json`。源码固定在
`9055cdebdae30d0dd6ea969bb2d60e90a02d236b`，直接继承原生 SFC 的冻结
Qwen3-Omni、BF16、TP2、seed17、4096 上下文、1024 输出、greedy 和关闭前缀缓存配置。
所有78个决策使用相同输出预算；不根据 gold 或 Noul 预测跳过 calls。

唯一直接前置是已排队的原生 SFC TRAIN6，后者依赖 SLURP 原生 TEST 和迁移 TRAIN 验证。
调度器检查精确进程与容器身份、前置验收和可用 NPU6/7；完成等待不代表已开始推理。
推理窗口7200秒，容器观察7500秒；失败或不明确派发均不自动重试。
推理不读取当前目标答案；对每个决策核对 CPU preflight 的 prompt/token hash 和完整音频帧，
并逐 token 比较 vLLM 实际 prompt IDs 与 HF processor 结果，前后校验全部模型文件。
独立验收成功后才读取这六轮 TRAIN 的目标并评分，不设置答案正确率门槛。
该队列尚无生成结果；不能报告为完成引擎验证或证明模型任务能力。

## 全量上下文检查

`sfc-jev-omni-full-context-v1` 已启动 CPU 容器，固定源码
`6e93bd2856407c8b77a6680d99e0b338ac89539a`，使用与推理相同的镜像和 processor。
对全部 8895 个 bundle、136722 个决策逐条执行真实 HF processor，
包括超过 30 秒的完整音频；不截断、不排除长音频、不读取 targets、不加载权重。
4 CPU/16 GiB 内存、无 NPU 映射、无网络，最大运行窗口21600秒。
每个决策记录 prompt/token hash、音频帧覆盖和4096/1024预算是否溢出，
每个 bundle 保存进度。
启动证据见 `sfc-jev-omni-full-context-v1-launch.json`。

现已确认精确容器 `844aa68b45dcbe41f8ca76c96cf8dbb9d53ced940a422c8a002af52bc63b1073`
退出码0、应用退出码0，并执行独立 `verify_sfc_jev_context.py` 验收。
核对源文件/输入哈希、全量有序决策覆盖、每条帧/预算记录和聚合统计均通过：

| 划分 | bundles | 决策 | 最大输入tokens | 溢出 |
|---|---:|---:|---:|---:|
| TRAIN | 5534 | 86679 | 1883 | 0 |
| ID TEST | 2590 | 38697 | 1837 | 0 |
| OOD TEST | 771 | 11346 | 1438 | 0 |

全部8895条/136722决策保留完整波形，均满足4096上下文含1024输出预算；
TRAIN中的8个超30秒bundle也已覆盖，最长39.65075秒。
结果SHA256 `42ffcdc9e5c89fdae87d2a9ad779f75ff4f89bed05d65940299453dabd772f3e`；
独立验收证据 `sfc-jev-omni-full-context-v1-independent.json`。
该校验器复核记录覆盖与一致性，不重新计算音频特征，也不代替 vLLM 引擎或模型能力评估。

## 教师提示优化的数据边界

当前固定划分 `/mnt/cephfs/xingchao/experiments/multimodal-jev/sfc-jev-prompt-partition-v3.json`，
SHA256 `84df8dc98723a50baa7352be11049cd8c692fb2e2da94ff5de650392abe09330`。
学习区3600会话/4428记录/69210决策；选择区899会话/1106记录/17469决策。
这是完整 TRAIN 的数据边界，尚不是已派发的实验预算或教师请求。
划分按输入 level 分层，以seed17和完整会话成员的SHA256排序，目标约20%会话用于选择。
若不同会话共享音频哈希则作为连通分量整体分配，不为凑数量拆开音频/会话。
已做生成工程验证的四个 Omni 会话，以及早期 ASR 的level_3_1/id_100会话固定进入学习区。
全量输入/类型/上下文审计不使用模型成绩，不能据此挑选更容易的选择样本。

独立重读验证所有4499个TRAIN会话均恰好归属一区、无跨区音频哈希、全部已知生成暴露样本
处于学习区；当前记录见 `sfc-jev-prompt-partition-v3.json`。未采用v1/v2划分做任何推理或教师请求，
v2补齐了id_100暴露来源，v3修正暴露报告路径变量覆盖造成的TRAIN输入哈希记录错误。
v2/v3的所有分组和统计完全一致；旧版本保留作审计记录。后续 teacher只能消费学习区反馈；
候选名单冻结后才运行选择区，官方test_id/test_ood不参与优化。
划分未使用意图/参数gold分层，不保证标签平衡，也不是跨说话人或新工具泛化划分。
待78决策工程验收后，再预登记教师预算、候选规则和学习运行范围，不能把当前分区直接视为训练授权。

本地反馈准备工具 `tools/jev/prepare_sfc_jev_teacher_feedback.py` 已实现：要求指定分区SHA256，
完整探针容器/应用验收、评分结果绑定，并从完整TRAIN输入复现分区。只接受学习区完整会话及
全部决策输出，重新运行严格评分；代表例按类型与错误类别确定性选择，每桶最多两例。
所有决策保留简明汇总，代表例保留未经修复的输出、动态定义和离线gold。
反馈不含音频路径、转写或原始target provenance，教师被明确告知没有听过音频。
私有消息文件限制150000字节，超额报错，不静默截断；输出最多两个1500字符通用指令候选。
仅通用INSTRUCTION可候选化，动态定义/输出契约/参数类型/音频顺序不能修改。
这是ACE式确定性失败反思准备，没有采样组内优势或音频因果对照，不能声称TF-GRPO或RL提升。
当前仅实现并测试工具，78决策真实推理尚待完成，因此未准备真实反馈、未配置教师预算、未调用API。

## 提示候选接入边界

`sfc_prompt_candidates.py` 单独封装候选通用指令替换，原 `sfc_jev_runtime.py` 不变，
已经排队的探针和正在运行的上下文审计仍执行各自固定源码。
选择原INSTRUCTION时要求消息逐字节等同原adapter；候选只替换system前缀，
保留后续函数/环境原文、决策定义、最终输出契约，以及全部user/assistant消息和音频引用。
不能把字符串替换测试误报为候选不会产生语义冲突：新增指令仍需内容审查和实际模型验证。

`stage_sfc_prompt_candidates.py` 只暂存本地proposal和feedback文件，要求显式文件哈希、
固定父提示、TRAIN学习反馈标记、最多两条不同候选、每条1500字符、最多三条诊断及有效证据引用。
候选不得含样本ID、Qwen控制token或伪造的任务分节标题；重复JSON键和非有限常量拒绝。
暂存不会执行文本或派发推理，状态固定为semantic_review=pending、candidate_accepted=false。
该工具**不认证远程API响应来源或SSE完整性**；教师请求执行器、响应认证、预算登记和候选
语义审查尚须接入，不能直接把未经认证的本地文件称为真实教师结果。
候选会改变token长度，必须重新做实际processor预算检查，不能继承父提示的长度证据。
候选文本尚未接入冻结探针运行器，本模块只是受测的输入适配与结构筛选层。
全量8895条/136722决策的静态父提示字节一致性和候选前缀替换检查已通过，
三个划分的父消息哈希均与原adapter全量审计一致。证据见
`sfc-jev-candidate-adapter-static-v1.json`，可用 `audit_sfc_candidate_adapter.py` 重复检查。
该次全量检查使用合成占位指令，不是真实教师候选，也未进行TEST生成或按TEST结果选提示。

## SFC 专用教师执行与响应校验

`configure_sfc_teacher.py`、`run_sfc_teacher_once.py` 和 `sfc_teacher_wire.py`
现已独立实现 SFC 的 `train_id` 反馈适配，不借用 FSC 的 `train`/分组 schema。
配置器要求显式指定准备收据和分区哈希，固定原始教师模板、六轮/四会话/78决策学习反馈、
最多两条1500字符候选；不改动既有排队推理源码。
固定沿用 Omni executor SHA256
`c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4`，
教师 `glm-5.3-flash`、thinking enabled、max_tokens32768、单次读取180秒、总请求900秒。
配置只引用已有凭据环境变量，不复制凭据；配置动作不派发API。

运行器在创建进程前独占写入持久派发意图，以配置和执行器哈希绑定此次请求。
失败、缺少PID或响应、超时均保留尝试且禁止自动重放；进程观察总限960秒。
校验器核对实际 request.json 与固定学习消息一致，再从 response.sse 重建最终正文，
检查 stop/[DONE]、单响应、稳定ID/模型、正文一致性和用量一致性。
教师私有 reasoning 不进入候选文件。报告为 length、缺失结束事件、响应后附加数据、
工具调用或模型变化均拒绝。缺失用量保留 unknown，不能记作零成本；
若供应商报告输出超过预算，仍保留可读的用量证据但不暂存候选。

`wire_authenticated=true` 仅表示本地保存的执行器/请求/SSE证据一致，
不是供应商数字签名或对抗本地恶意篡改的认证。即使校验通过，
semantic_review 仍为 pending、candidate_accepted=false，也不启动候选推理。
真实调用尚未配置或派发，等待 TRAIN78 工程验收后生成实际学习反馈。
当前仅完成合成响应和失败注入测试；语义审查、候选实际processor预算和引擎评估仍未完成。

## 完整人口的会话分片执行计划

已准备 `sfc-jev-complete-shards-v1`，输入计划与实际物化证据见
`sfc-jev-complete-shard-plans-v1.json`。分片只是运行单位，不修改发布划分或任务定义。
目标每片1024决策，完整source_group不可拆分，各会话内按音频阶段排序，
保留每个函数的present、absent、calls全部问题，不按gold或Noul预测筛选。

| 人口 | bundles | 会话 | 决策 | 分片 |
|---|---:|---:|---:|---:|
| TRAIN学习区 | 4428 | 3600 | 69210 | 69 |
| TRAIN选择区 | 1106 | 899 | 17469 | 18 |
| 完整ID TEST | 2590 | 1986 | 38697 | 39 |
| 完整OOD TEST | 771 | 593 | 11346 | 12 |

`sfc_jev_shards.py` 从原始输入重建并核验每个bundle定义、完整会话和音频前缀；
TRAIN必须绑定v3分区哈希并从全部TRAIN复现分区，不能把学习/选择混用。
TEST不接受人口过滤参数。分片顺序仅按source_group排序，与标签和模型成绩无关。
每片固定有序bundle IDs、完整输入哈希和有序决策键哈希；加载或物化前重新生成全计划，
拒绝缺片、重复片、遗漏阶段/决策、分区变化或内容变化。
若单会话决策数超过目标，整组保留并标记超额，不拆会话；本次实际四个人口均无超额组。

计划位于 `/mnt/cephfs/xingchao/experiments/multimodal-jev/sfc-jev-complete-shards-v1/`。
`learning.json`/`selection.json`固定v3分区；`test_id.json`/`test_ood.json`覆盖完整官方测试。
`plan_sfc_jev_shards.py`可重新生成新计划，`materialize_sfc_jev_shard.py`
要求显式计划哈希，只输出所选片requests和来源manifest，不读取或输出targets。
首个学习片已实际物化并校验，未派发推理。它不替代既有TRAIN78的固定输入与队列。

以上是完整执行准备，不是模型成绩、候选选择预算或实测运行时估算。
分片推理器、逐片工程验收、完整合并及官方评分仍须接入；
正式运行预算需结合TRAIN78实测耗时，候选提示还需单独验证processor长度。
仅完成全部注册分片并逐条验证输出后，才能报告对应完整人口的分数。
