# SFC-Jev 候选提示的完整划分执行接口

原有候选仅支持结构暂存、教师返回记录校验与 CPU 上下文检查；分片推理、
验收和合并此前只接受 parent。本次将候选通用指令接入同一完整分片执行链路，
没有新教师调用、候选推理或质量结果。既有已排队快照保持原样。

## 候选身份

配置的 prompt_id 为 p1/p2，instruction_sha256 绑定实际候选字节；candidate 字段
包含 staging、staging_hash、candidate_id、review、review_sha256。
execution_prompt 重新验证教师调用的本地 request/SSE/response 证据及 instruction。
本地手工 stage 即使格式正确，也不能作为“远程教师候选”进入此路径。
wire_authenticated 表示既有本地记录链一致，不是提供方数字签名。

语义审阅单独保存，schema 为 sfc-jev-candidate-semantic-review-v1，须绑定候选 ID、
staging 哈希与 instruction 哈希，记录 reviewer、rationale、
decision=eligible_for_evaluation，以及全部八项检查为 true：

- generic_instruction_only
- audio_direct
- history_and_audio_order
- function_environment_unchanged
- decision_output_contracts
- no_noul_gating
- types_nan_duplicates_and_order
- no_sample_answers

这些字段是工程师/助手阅读实际候选后的判断记录，代码不会自动生成肯定结论。
不要求用户再次授权，也不把语义适合评测描述为性能已提升。
暂存文件及其 pending 标记不原地修改，避免破坏已完成上下文审计的身份。

## 上下文、执行和合并

候选必须拥有自己的完整 CPU context identity、result 与 independent acceptance，
三者的 prompt 绑定与当前候选一致，不能借用 parent 上下文。
候选 renderer 与 audio 代码还须匹配审计快照，运行中逐条比较实际 prompt/input IDs
及帧覆盖。整个 shard 完成后再次加载候选，防止运行期间文件漂移。

实际替换位置仅为通用 system 前缀，其余动态函数/环境、决策定义、输出约束、
历史和音频顺序沿用父提示。Noul 与 typed_call_list 均独立生成，不按前一个答案跳过。
全量队列保留完整划分的所有 shard；源码快照及动态 candidate 配置被固定。
只读挂载候选教师根目录、审阅文件以及被证据链引用的 executor 路径。

独立 shard 验收按相同候选调用 context verifier；合并检查 prompt_id、instruction、
全部配置、代码和模型身份跨 shard 一致。不能把 parent、p1 或 p2 混合汇总。
为兼容原队列文件，完整队列 schema 仍保留历史 parent 名字；真实 prompt 身份以
被哈希绑定的 config-template 和每个 shard config 为准，不由 schema 字面名称推断。

当前候选执行仅接受完整 TRAIN learning 或 selection 划分。TEST 仍需独立的
候选选择/冻结协议，尚未实现该路径；不允许把候选直接指向 TEST 来跳过选择阶段。
父提示的既有 TEST 路径保留。

## 验证范围

测试使用合成教师 SSE、结构化审阅、context 及引擎返回；覆盖有效候选确实进入
推理请求，双音频与历史保留，false Noul 不抑制 calls，错误输入 token 被拒绝，
以及 pending/错绑审阅、父提示 context、TEST 候选路径、代码漂移和混合候选被拒绝。
父提示运行器、完整分片队列、合并评分和教师记录链也执行回归测试。

尚未生成真实 SFC 候选或运行其 CPU/NPU 验证；真实教师链依赖原 TRAIN78 完成。
后续须阅读实际候选完成审阅，再执行候选完整 CPU 审计及 learning/selection 推理，
按预先固定的选择指标比较后另行冻结 TEST 提示。当前不作提示优化效果或泛化声明。
