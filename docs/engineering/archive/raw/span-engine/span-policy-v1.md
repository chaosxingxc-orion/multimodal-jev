# 多 Span 随机策略与 verl 接口

模型一次读出所有 Span 的候选分布；同次推理的前置离散答案不回填。给定完整输入 x，各位置的采样相互独立，因此完整动作 a=(a1,…,aS) 的概率为 Πj pj(aj|x)，联合 logprob 为 Σj log pj(aj|x)。隐藏表示共享不改变该采样分解。

`span_policy.py` 显式采样完整元组，输出 B×G×S 的候选索引和 B×G 联合 logprob。缺少中间 gold 只影响奖励/监督，不改变动作空间，不从 logprob 中删掉对应 Span。候选分布归一化在单个 Span 的合法候选内，不能混用全词表 SFT logprob。

verl 接口将完整元组作为长度 1 的 response action：对同一完整输入的 G 个动作先执行 verl 原生 GRPO outcome advantage，再以联合概率比调用原生 vanilla clipped PPO。此处长度 1 是损失接口的数学映射，不是生成了一个 token，也不是声称已经接入原生生成式 actor。旧 token-mean 或逐 Span clip 均不同于联合概率比，不能混用。

同组 request_id 必须绑定音频、全部问题、思考要求、候选顺序及同一政策/采样配置；跨问题模板不能归为同组。当前函数只检查组形状和唯一 ID，身份绑定、vLLM 采样凭证、actor 重算与准入仍由后续执行层实现。旧生成式 actor 不能直接作为此接口实现。

学习目标仍为：固定 SFT 初始化 → 等预算继续 SFT 对照 → Final 奖励 RL → 可靠中间奖励消融。并列 temporal 任务仅为接口检查，不用三角色联合正确率冒充推理链最终质量。无中间过程 gold 时不能自动造过程奖励。相同输入下与中间随机答案无关的 Final-only 奖励，对中间采样分量的期望直接策略梯度为零；Final 损失仍可通过共享骨干改变前置隐藏表示。这限制了对“RL 学会中间离散推理”的解释。

当前 host 概率测试覆盖完整动作枚举归一化、概率乘积、采样概率重算和梯度。固定镜像的 verl loss 核验另记；没有执行 RL 参数更新，不能声称任务收益或端到端训练框架已经完成。

固定镜像实际 verl 比较已通过：loss=0.0373170562，手工 joint PPO 与 verl 梯度最大误差 0，常量奖励组 loss=0。只验证数学接口；vLLM 原生 pooling 真实音频已跑通，但对 HF 的候选 logprob 误差 0.063258 > 0.02，未准入 actor 更新。

实现补充：`span_rewards.py` 提供 Final 正确性 + 有界辅助正确性的分项回执；Final 必须有可靠监督，中间缺标严格屏蔽，按有效中间项数取均值。当前未启用逻辑一致性项，也未对并列 temporal 任务伪造过程奖励。该函数接受候选索引，合法候选范围仍由采样/契约层检查。

`span_parity.py` 对所有合法候选计算误差，并通过各位置极值求和精确得到所有完整动作的联合概率比范围。基座实测 [0.879938,1.076945]，超过新增 [0.98,1.02] 要求；即使 argmax 全同也不能视为 PPO old-policy 一致。该门槛及历史单项门槛均保留失败证据。

请求绑定已实现为 `span_request_identity.py`：完整问题与thinking、原始及处理后波形hash、前处理配置、输入特征/掩码/token tensor的dtype/shape/hash、Span位置、候选顺序、base+adapter政策身份共同确定request_id。测试验证改变音频特征、问题要求、候选顺序或政策都会改变ID，labels等额外字段拒绝。它在后续probe中接入；已启动的历史快照不回写新凭证。worker仍须独立核验展开token及实际加载adapter，不以客户端hash代替执行证据。
