# 完整定义任务选择8192上下文候选

2026-11-12。采用现有Span encoder和完整意图定义重新执行90个真实音频processor
组合，全部完成且不超过8192。任务上下文需求登记在
[context-requirement-candidate-v1.json](../../configs/tasks/slurp/context-requirement-candidate-v1.json)。
这选择的是待工程验收的任务预算，不是已放行的训练配置。

## 对此前探针的必要纠正

旧text-budget及panel-context-v2为了中性编号，使用regex在正文中替换本体ID。
time、date、person等也是普通词，该方法会破坏英文定义，甚至匹配update等词的子串。
旧结果仅保留为那些实际输入的测量，撤回其“保留原语义”资格；不能据其token差值
声称语义等价压缩。新版本只替换JSON结构化键和显式类型引用，正文逐字保留。

本轮还把60意图的includes/excludes全部前置；不存在用删边界换取长度通过的情况。
全部15面板逐项核对意图定义三字段、所需槽位定义、共同规则正文及相邻引用完整。
Direct/2Q/8Q的knowledge完全相同。4项设计测试通过，包括普通词与子串不被替换。

## 真实processor结果

| 臂 | 请求数 | tokens范围 | 超8192 |
|---|---:|---:|---:|
| Direct | 30 | 5208–6773 | 0 |
| 2Q | 30 | 5287–6933 | 0 |
| 8Q | 30 | 5524–7412 | 0 |

使用已认证learning人口的中位长度与最长23.36秒音频，15固定面板；定义在首个Span
之前，实际音频展开及唯一读出对齐通过，无截断，容器exit0。记录hash、全部组合及
分组统计独立核验见[证据](panel-context-probe-v3-verified.json)。原有定义争议仍存在，
逐字保留不等于语义已经正确。

本地模型config的text_config.max_position_embeddings为65536，rope_type为default，
8192无需在声明位置范围外扩展RoPE。audio_config.max_position_embeddings=13属于
音频编码器结构，不是整个输入的token上下文上限，不能混用。

## 工程边界

停止以4096和删减知识作为主设计约束，保留8192作为统一基座的任务需求候选；不修改
旧实验快照或旧运行参数。本轮不引入compact renderer，避免任务尚未定稿就增加编码器
分叉。现有schema序列化可完整保留定义，正式代码仍需在任务冻结后统一接入。

90组合不是全人口、所有轮转菜单或NPU证明。正式冻结还需完整任务输入检查、内存及
吞吐、BF16 actor/重载/vLLM概率一致性和三种算法生命周期验收。更长输入可能提高
成本，不从CPU检查推断四卡训练速度。未派发研究训练。
