# Omni 端到端语音 Jev 强化学习设计 v1

本文件补充长期算法设计，不改变当前冻结基线 → 远程教师提示优化 → 完整 Jev 比较
的执行顺序。权重训练继续不派发。`01-rl-research-protocol.md` 中 ASR→文本路线
是历史设计，不能作为本项目当前主任务；现阶段 actor 使用本地 Qwen3-Omni，
直接消费音频与任务定义。SFC 的 typed_call_list 是本项目结构化扩展，不冒称 nga 原生类型。

## 研究问题和证据边界

问题一：在同一初始 checkpoint、训练来源和可核查预算下，结果奖励 RL 相比 SFT
是否提升完整任务正确率？问题二：在严格单决策奖励之上加业务关系奖励，是否提升
完整函数决策质量，而非只提高输出一致性？问题三：收益是否迁移到未用于优化的任务？

参考 nga `61f244ff9ac0409b6a2a59e4563d4e4e6afdef72` 的
`proposal/00-research-proposal.md` R2：不变性必须配区分性，忽略定义的常量策略
不能算成功。本次 SFC 的 present/absent/calls 三元组只支持逻辑及答案正确性约束，
还不能证明候选顺序不变性、定义重绑定、IIA 或音频因果依赖。
SFC 已有官方 ID/OOD 只支持相应划分下的泛化报告，不能直接称为跨数据集迁移。

## 输入和采样单位

沿用已发布 SFC-Jev 原始完整输入：函数定义、环境、官方提供的 assistant 历史、
按时间顺序排列的一或两段波形。gold 在独立奖励数据中，参考转写不进入 actor。
所有函数的 present、absent、calls 独立采样，不以预测 present=false 跳过 calls。
这保证训练和评测都保留完整分母，预测错误不会改变被评测的问题集合。

采样身份至少包含 release、split、bundle_id、decision_id、prompt_hash、policy_hash、
audio hashes 和 replicate。GRPO 的组必须是同一个完整模型输入的 K 个响应；
不能用整段录音、会话或函数三元组代替 prompt 身份。初版 K=4 是待登记实验设置。

给业务关系奖励时，在同一 bundle 内按预先固定的 replicate 对齐三种决策。
不得按奖励/正确性排序后配对，不做 K 的三次方穷举，不丢弃 malformed 响应。
跨 rank 先收集完整 bundle/replicate 和每个 prompt 的 K 个结果，再算奖励/优势，
之后才能切微批。当前 `sfc_jev_reward.bundle_rewards` 只计算一个完整 bundle/replicate，
不负责采样、跨 rank 收集或 GRPO 分组；现有 FSC `JevExample` 也不能无损表达这个输入。

新增 `sfc_reward_batch.compose` 在完整原始 TRAIN 上重现 learning/selection 分区，
检查选定会话所有原始轮次及每决策 K 个固定编号响应；返回与原始记录顺序对齐的
奖励和同输入 uid。uid 绑定 request（含音频哈希和历史）、decision、策略、指令、
实际渲染 prompt 和展开 input IDs 的哈希。同会话不同轮次/问题不会合并为一个组。
它还要求评分文本与 tokenizer 从采样 token 解码的文本一致，拒绝旧策略和缺失副本。
prompt 证据必须由调用方在采样前从真实 processor/engine 登记；此纯函数不能认证
伪造的运行元数据，也没有构造 Omni actor 张量或完成分布式采样。

## 已实现的离线奖励定义

对决策 d 的原始输出 y，直接复用 `sfc_jev_score.score_bundle`：

\[
r_d = \mathbf{1}[y_d\text{ 严格正确}]
      - \mathbf{1}[\text{格式无效}]
      - 0.1\,\mathbf{1}[\text{格式有效但 schema 无效}].
\]

正确性遵循原始调用多重集：重复调用不可丢，参数绑定和类型不可改，数组内部顺序
必须保留；同一函数的调用列表允许按多重集比较。`NAN` 是原数据缺参哨兵，
`[]` 是该函数无需调用，`null` 不是通用拒答。格式解析不去 Markdown、不修复 JSON。
当前没有置信度奖励，也不奖励模型自报工具执行成功。

实验模式 A（independent）仅使用 r。模式 B（function-conjunction）对每个函数 f 定义：

\[
b_f = \prod_{d\in\{present,absent,calls\}}\mathbf{1}[y_d\text{ 严格正确}],
\qquad R_d = r_d + 0.2 b_f.
\]

每个端点最多增加 0.2，每个函数恰好三个决策。报告单决策分量与 bundle 平均值，
不让函数数量直接放大奖励。相同正确端点在其同 replicate 的其他端点错误时拿不到
bonus，形成不同于单题准确率的训练信号；但信号是否改善学习仍需受控实验。
这一模式不命名为已证明有效的 pair-GRPO，也不等同候选变换不变性奖励。

三元组逻辑一致但全错时 b=0。确实无需调用的函数输出 false/true/[] 时 b=1；
不能为避免全空退化而处罚真实无调用目标。因此还需按 active/inactive 函数分别
报告正确率和 bonus 分布，检查大量 inactive 函数是否主导训练。

真实输入的离线核对已验证这一风险，见
[TRAIN78 奖励夹具](../docs/engineering/sfc-jev-reward-train78-fixture-v1.json)。
六条 TRAIN 学习样本含 26 个函数三元组，其中 17 个无需调用。构造的全空策略
保持 26/26 逻辑一致，17/26 三元组答案正确，但完整原生任务 0/6 正确，
获得 61.2/93.6（65.38%）的 oracle 总奖励。该数值来自人工构造输出，绝非模型表现。
当前实现不能被描述成“已消除全空奖励退化”。后续应在仅 learning 的分布审计后
预登记 active/inactive 平衡或整 bundle 完成奖励的独立消融；保留本版单题/三元组
对照，不根据测试结果修改奖励，也不把未经比较的改权重声称为解决方案。

随后完成的 [完整 learning 审计](../docs/engineering/sfc-learning-reward-audit-v1.md)
确认 inactive=16,985/23,070（73.6238%），全空策略获得相同比例的 oracle 总奖励，
完整任务仍为 0/4,428。该完整分布替代六条探针比例用于后续设计依据。
learning 中无整 bundle 空调用标签，整任务拒识仍有覆盖缺口。以上均为人工策略
离线评分，不是模型输出；尚未据此修改奖励系数或启动训练。

## 必须包含的对照与消融

| 条件 | 用途 |
| --- | --- |
| 冻结 parent / 冻结教师候选 | 将提示收益与权重收益分开 |
| 同初始 checkpoint 的继续 SFT | 排除额外监督和训练预算造成的收益 |
| independent GRPO | 相同严格单题奖励、无关系项 |
| function-conjunction GRPO | 相同采样和预算，仅增加业务关系项 |
| conjunction + group centering / 标准差归一化 | 检查奖励尺度及小方差组放大影响 |
| 常量 false、全空调用、格式合法但错误调用 | 退化策略诊断，不参与模型选择 |

需要固定同一初始策略、可训练参数集合、数据次序、学习率及 seeds 17/29/43。
SFT 若作为初始化，各 RL 分支必须共享该确切 checkpoint；另保留同 checkpoint
继续 SFT 对照。不要同时声称“从基座直接 RL”和“从 SFT 初始化”。
最终训练 token、rollout token、更新次数、NPU device-hours 分开报告；
固定更新次数不等于固定算力，固定 token 预算下可能无法固定相同更新数。
预算数值和停止规则须在 Omni NPU 前后向可行性确认后登记，不能照搬旧 ASR 配置。

## 优化器与模型接入前置条件

镜像固定为 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
verl `0.8.0.dev0` / 上游 `e9405d70b77d56a23e72bb2e40a17b906856e913`，
vLLM `0.30.0+empty`、vllm-ascend `0.30.0rc2`。这些版本沿用工程基座，
并不证明默认 verl server 支持音频。

1. 用 Qwen3-Omni 而非 ASR 验证真实单/双音频 teacher-forcing、完整音频帧和
   response mask，检查 HF actor 与 vLLM 的展开 ID、EOS 和逐 token old log-prob。
2. 确定 MoE 模型的 LoRA/参数更新范围，实测 NPU 显存；当前 TP2 推理成功
   不证明两卡训练可行。不得为配合旧 actor 静默换模型。
3. 证明微批和多 rank 下损失归一化与参考计算一致，核验音频张量、MoE 路由、
   梯度同步、参数导出与加载；旧 ASR CPU 或合成 Gloo 结果不能替代这些检查。
4. 全零优势时显式 no-op，包括 AdamW 动量和 step 计数。保留 rollout old log-prob，
   不允许用当前策略概率替代；冻结参考 KL 若启用，需要独立验证和预算。
5. 新策略导出后至少完成下一轮真实采样与更新，验证 checkpoint 身份、优化器状态
   和中断恢复。原生 Ray manager/server/agent-loop 对音频的缺口仍需解决。

当前奖励模块没有接入上述路径。只是实现其可独立验证的语义部分，未触碰运行中实验。

## 选择和泛化协议

SFC TRAIN v3 learning/selection 按完整会话和音频连接组隔离；奖励模块拒绝
非 train_id 或不在调用方 learning 集合的输入。调用方仍必须从原始完整 release
重现分区并核对哈希，不能把任意集合当授权名单。
selection 选模型，官方 test_id/test_ood 不参与提示、奖励或训练超参调整。

跨任务实验必须以整个数据集/业务族为 holdout。当前 FSC、SLURP、SFC 均有开发活动，
不能因模型未读其某次测试标签就称它们完全未见研究流程。可预登记明确的
“FSC 学得策略直接迁移至 SLURP/SFC”的迁移实验；更强的新任务泛化需另留
未用于奖励/提示设计的来源。Audio2Tool/MMAR/真实交互仅在原始访问、标签与 scorer
满足条件后纳入，不能用静态 SFC 离线函数预测替代工具执行成功率。

逐任务报告官方 scorer、严格 call multiset、all-decisions-correct、格式/schema 有效率、
present/absent 各自正确率、active/inactive 分层、逻辑一致且正确比例、成本。
会话是统计聚类单位；同一音频派生的多决策不算独立样本。方差与置信区间需反映
seed 和会话层面不确定性。关系一致性上升而严格正确性下降时，不宣称 Jev 能力提升。
