# Audio2Tool Tier7 决策时点与文件映射

固定 release `f1388da9a3189541ab82adac88824a0661670c43` 数据卡明确：
每条记录是一段对话，`user_request` 是触发工具调用的最后一轮用户请求，
`chat_history` 包含双方各轮音频。来源为此前固定的数据卡
`.local/research/sources/audio2tool-card.txt` 的 Subset-specific fields / Multi-turn 节。
当前官方 loader 把各轮当独立音频变体处理的问题见
[官方代码审计](../../../../research/audio2tool-official-code-audit-v1.md)。

## 全量结构证据

`audit_audio2tool_dialogue_protocol.py` 先验证 release 文件的下载清单哈希及字节数，
再将全部 2146 条 flat JSON 记录与原无答案音频索引逐条比较。
这是读取含转写和标签的 TEST 发布 JSON 的结构审计，不能描述为没有读取标注文件。
程序仅使用角色、路径、身份和最后用户文本相等关系；不输出转写/目标值，
不调用模型/教师，不评分、不选择提示或候选工具。

- 2146/2146 的完整有序路径、角色与文件名轮次一致，覆盖全部 7834 WAV。
- 2146/2146 的 `user_request` 等于最后一个 user 轮次的 `content`。
- 音频目录 `query_NNNNN` 在 2146/2146 条中对应发布 `id`，0/2146 对应 `query_idx`。
  两种 ID 都唯一，但含义不同；实际路径应来自发布路径与核验索引，禁止用
  `query_idx` 推造目录。首条 id=1、query_idx=2100，足以区分两种映射。
- 1445 条为 user/agent/user，677 条为五轮交替，6 条为七轮交替。
- 14 条为 user/agent/user/user：保留连续 user，不修补或丢弃其中一轮。
- 4 条为 user/agent/user/agent，发布 id 为 373、1258、1744、2023：
  在最后用户请求后仍有一段 agent 音频。不能把最后一段简单当成当前用户请求。

工具输出保留所有 2146 条结构记录；4 条尾随 agent 和14条连续 user 均单独标记，
没有删除异常对话。见[完整结构结果](../../../audio2tool-dialogue-protocol-v1-result.json)。
6 项测试覆盖两种非标准角色顺序、ID错配识别、路径/角色/轮次篡改，全部通过。

## 用于后续端到端评测的决策时点

本项目应将调用时点固定在最后一个 user 音频结束后：输入完整的此前对话和
该 user 音频，保留角色及顺序。四条尾随 agent 音频保存在原始数据中，列为
决策后材料，不作为该时点 actor 输入；这不是为满足长度预算而裁剪音频。
连续 user 轮次照常保留。对应前缀共 7830 段，仍是完整 2146 条对话分母。
该规则由任务时点决定，不依据模型输出或得分；它必须在正式请求配置中显式固定。

该规则下的音频输入清单现已生成，正式任务 prompt/推理请求尚未固定，未执行推理。
之前全对话的 32768 容量审计
保留为较宽输入的容量证据，不冒充最终决策协议；将来正式 prompt 若变化仍须重验。
同时，不能据此宣称复原了论文实际历史输入方式；论文 v1 与当前发布代码差异仍存在。

正式 actor 只读取核验后的音频、角色与独立完整工具 registry；`query`、
`user_request`、`chat_history.content`、`reasoning`、`functions` 中按样本提供的候选，
以及 gold 调用均不得成为 actor 输入。目标仍只进入独立 scorer，数据全体保持外部 TEST。
当前结构审计不证明目标调用与最后用户请求语义一致，也不解决发布 scorer 的顺序、
重复调用、参数类型和全分母评分差异。

## 已实现的输入投影

`audio2tool_dialogue_input.decision_input` 只接受原无答案索引的明确字段集合，
生成 `input.audios` 和独立 `provenance.post_decision_audios`；不原样复制未知字段。
`messages` 仅接收 `input`、调用者核验后的波形、任务文字及独立 registry 哈希，
不能把 provenance 当输入，也拒绝转写/答案等额外字段。完整有序音频及角色标记
放在一次 user 消息中，音频位置与既有 processor 探针布局一致；不将 agent 声音
转换为文本 assistant 历史。真正的 waveform hash/重采样和任务文字与 registry 的
绑定仍由调用方检查，renderer 本身不声称已完成这两项验证。

准备器仅打开固定 inventory、结构审计投影和独立 registry，不打开标注源文件。
产物为 `audio2tool-tier7-decision-inputs-v1/inputs.jsonl` 和 `registry.json`，全部
2146 条保持 external_test。独立逐条核对 active+post 恰好重建原索引，顺序/哈希/
角色/字节数均保留；actor 7830 段、post 4 段，完整152工具表与原产物一致。
19 项输入/协议测试通过，覆盖未来音频、连续 user、波形顺序和额外字段拒绝。
见[完整准备与复核](../../../audio2tool-tier7-decision-inputs-v1-verified.json)。
这只是输入侧转换，尚未完成 scorer、Jev target 转换、模型推理或论文协议复现。
