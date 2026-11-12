# 语音 Jev 三阶段实验协议 v1

2026-11-09；CPU 资源与数据接口准备阶段。以下假设为前瞻登记，不是实验结论。

**历史协议，已被后续执行路线修订。** 下文ASR→文本、RL/SFT效果比较、
调度循环及200次更新等设置仅供追溯，不是当前派发指令。当前执行遵循
[三条研究主线工作基线](../docs/research/15-rl-three-track-working-baseline.md)
与[单次Span契约](../docs/research/12-single-pass-span-cot-contract.md)：本地
Qwen3-ASR-0.6B直接消费音频与任务配置，BF16 LoRA四卡数据并行训练，
研究RL持续提升、结构过程信用及FSC配置迁移，不新增RL/SFT效果比较。
具体准入与运行证据见当前工程记录，不能将下文旧CPU验证当作训练收益。

## 研究问题与判据

目标是设计 RL 算法，提升语音 Jev 的决策质量与跨任务泛化。分开检验：

1. 固定监督信息及训练预算后，读出 RL 是否优于交叉熵 SFT？
2. 成对 schema/证据约束能否提升未见定义与未见任务泛化，而非仅降低预测变化？
3. 具有获取证据、澄清、回退和执行选择的序列策略，能否改善成功率—成本关系？

NGA `proposal/00-research-proposal.md` 的 R2/R4/R8 与数据覆盖矩阵
`proposal/11-dataset-coverage-inventory-and-selection-2026-11-01.md`
（参考 checkout `61f244ff9ac0409b6a2a59e4563d4e4e6afdef72`）提供设计依据。
借鉴定义优先、nominal/boolean/ordinal 区分、未知状态、成对区分性和来源族隔离；
不继承 NGA 的“仅调度可做 RL”限制。该限制是其研究假设，本项目以同预算 SFT 为必要对照检验它。
未复制 NGA 实现代码；依赖栈通过其已有镜像继承，许可证随依赖保留。

## 阶段 1：复现与诊断

| 路线 | 输入与读出 | 验证条件 |
| --- | --- | --- |
| ASR→文本决策 | 官方 HF Qwen3-ASR 转写；固定文本模型按 schema 输出 JSON | ASR WER、语义指标、两个模型的总成本 |
| 人工转写文本对照 | 原始 transcript；同一文本模型/定义/输出 | 诊断 ASR 误差传播，不声称声学上界 |
| 冻结声学编码器 | projector 输出做有效帧池化，训练任务头 | 同一官方训练集和参数预算 |
| 冻结 ASR decoder | 最后有效 token 状态，候选条件 Choice 或 Noul/Score 头 | 必须验证定义输入真正进入计算 |
| AudioJev / SFC 外部参考 | 按官方方法和固定代码版本 | 官方 scorer 与统一决策指标并列；不以论文表格代替实测 |

CPU smoke 只确认权重精确加载、转写及表征形状；正式基线要跑冻结的 dev 协议。
初轮开发只使用训练与 valid；FSC test 已被构造器读取用于元数据校验，不进行模型选择。

## 阶段 2：任务构造

- FSC：31 个组合意图，保留官方 train/valid/test。候选只从 train 定义，
  `none` 是原始槽位的未指定值；拒识使用 JSON `null`，两者不可混淆。
- 同录音派生原始 Choice、候选顺序变化、候选 id/定义重绑定、移除正确候选、正/反 Noul。
  原录音所有派生视图保持同 split、共享 source_group。缺候选仅是 schema 拒识诊断。
- SLURP：后续增加 scenario/intent 与原始 entity span；开放参数保持文本或原始 span，
  不把整个 test 槽位词典泄露为候选。原始官方划分与来源关系保留。
- SFC/Audio2Tool：后续保留函数定义、参数 schema、对话/说话人/来源族及官方 ID/OOD；
  开始时作为外部 holdout，不混入 FSC 训练。接入其 train 后须另冻结独立来源 holdout。
- SpokenWOZ/τ² voice：后续构造交互状态、缺参澄清、纠正/取消与工具调用；
  工具执行结果来自环境，不接受模型自报“执行成功”。
- Score 仅对具有原生序数/标量标签的任务启用，明确等级定义。
  不以字符长度、槽位数或 ASR 置信度伪造业务评分标签。
- 声学反事实必须有可核验标签；目前未构造，不能以候选重绑定替代声学因果证据。

发布清单包含源哈希、构造器哈希、配置、音频哈希、样本数和跨 split 重叠统计。
生成目录按内容标识命名，已存在即拒绝覆盖。转写与 target 仅在监督/奖励列，
音频模型输入只含 audio 和 schema；参考转写对照须显式选取模式。

## 阶段 3：RL 实验设计

比较 SFT、标准 GRPO、成对约束 GRPO（暂称 pair-GRPO）以及调度 RL。
所有训练从同一 SFT checkpoint 开始；使用 seeds 17/29/43，同样训练样本、token 预算和更新次数。
初轮开发预算：每 seed 至多 200 次更新，rollout group 4；最终预算在 NPU smoke 后冻结修订。

奖励 v1：语义严格正确 + 格式违规惩罚 + 已知问题错误拒识惩罚 + 可选 Brier 项。
权重 1.0 / 0.1 / 0.1 / 0.1；置信度只解释为该次输出正确概率。
独立报告 reward components，缺失置信度不给校准加分。
Pair-GRPO 规划：同 source_group 的视图共同采样，在映射回语义身份后约束置换一致性，
同时对重绑定后正确 id 与正反二元问题给区分奖励；仅一致且正确才允许正奖励。
先验证奖励不能被常量输出、全拒答、输出候选列表或伪造执行字段利用。

调度 RL 的成本来自环境观测（token、延迟、工具调用、回退），错误/过早执行单独罚分。
最终比较必须包含固定规则控制器、总是回退、从不回退、总是拒识。
单决策与成对奖励已接入本地 verl 更新路径并通过受控数值检查；真实 vLLM
采样到更新的端到端路径、算法效果与交互环境仍待验证或实现。

## 指标与停止规则

分别报告 native intent/joint accuracy、Noul balanced accuracy、unknown precision/recall、
coverage/risk、Brier、按 source_group 的置换一致性与区分成功率；不得跨不同指标直接平均。
开放槽位报告原始 scorer、exact match；交互报告任务完成与错误执行。
跨任务报告每个来源的数值及宏平均；置信区间按说话人或对话聚类 bootstrap。

若 RL 不优于同预算 SFT、收益只存在 FSC 派生视图、或一致性改善伴随区分退化，
不宣称泛化收益。测试只在 dev 选定模型后进行；看 test 后改方案须另找未暴露测试证据。

## 当前边界

NGA 基础镜像与 Qwen3-ASR 模型注册已核验，但该 verl 默认数据/rollout 代码未处理 audio。
必须补齐音频 DataProto、processor 输入、vLLM request 和 actor log-prob 输入一致性后才能启动 RL。
默认 ASR 模板可能只处理音频及 system 文本；不能假设 user 文本 schema 会被保留。
NPU 主机目前由已有任务占用，本项目没有抢占或停止它们。
