# 上下文概率头的原生 verl 多进程更新

原生 actor 入口 `src/multimodal_jev/context_native_actor.py` 按每个原始音频 query
传输一行，行内保留完整采样组。一次 microbatch 只执行一次骨干前向，所有采样动作
共享该次 logits。GRPO/RLOO 的优势在切分前按完整组计算；对比排序、二元 NLL、
集合 F1 奖励、联合 PPO 与固定参考 KL 经注册 loss 接入原生 backward/clipping/Adam。
源代码同时核验原生 actor 与 core_algos 的 SHA256。

## 四进程 CPU 证据

固定镜像 `52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
v2 容器 `6930f1f3b176814a75696a94496c0083779b8fea90360fbd3fcf5b0c6979203f`
成功退出。完整证据见 `context-native-actor-dp-cpu-v2.json`。
四个真实 Gloo/DDP rank、实际原生 verl actor、合成音频骨干与真实浅 Transformer 头；
GRPO/RLOO × micro1/2 × 三次更新，每 rank 12 项。第一步为 instruction，随后两步
RL；第二步后保存、扰动、恢复，再与不中断独立参考比较第三步。全局8行中7行有效，
验证零权重尾部及全局 query 分母。错误分母、mask、补齐权重和跨 rank 重复请求被拒绝。

最大误差：梯度5.2154e-8，参数6.3093e-8，Adam一阶矩5.5879e-9、二阶矩1.0914e-11。
优化器与研究基线一致：lr1e-4、betas(.9,.999)、eps1e-8、weight_decay0。
此处使用FP32合成模型、G8、global8，不是Qwen实际音频/BF16/FSDP/global128准入，
也没有任务收益或NPU吞吐结论。

v1保留失败：旧线性头诊断复用了lr.01，attention key bias近零梯度的FP32批量归约
差异经Adam放大，参数差2.2445e-6超过2e-6。无verl/DDP的本地独立重现得到同样差值；
将诊断优化器改为预先定义的正式lr1e-4，比较阈值不变，未改模型方程或冻结偏置。
失败报告 `context-native-actor-dp-cpu-v1-failure.json`。

## 更新前的策略一致性

新增 `context_behavior_parity.py`：完整二元logp按FP64比较概率、logp及0.5阈值翻转，
同时报告任意完整动作元组log ratio的保守上界。概率误差很小不代表罕见动作logp误差小。
actor在现有训练前向后、backward/optimizer之前收集所有rank的检查结果；任何rank
失败则全部拒绝。本轮只有一个optimizer update，微批间参数不变，因此该位置可直接
比较当前actor和采样behavior。明确要求配置本协议阈值，不沿用旧Final60阈值。

18项本地相关测试通过。v3容器
`89e371b192ea1042e2286d26a33cc7a5f6b45b55a1cd501bc9be4f414cc38279`
已exit0/nonOOM，完整收集并逐rank核验48个case；任务阶段所有rank的behavior概率
最大差0。相同policy ID下伪造但内部自洽的behavior被集体拒绝，16个rank-case
检查参数/Adam不变；不一致只注入rank3，其他rank仍参与共同拒绝。
证据 `context-native-actor-dp-cpu-v3.json`。原v1/v2失败与成功现场均保留。
CPU诊断阈值2e-6不授权生产BF16阈值。生产仍待命名runtime/token冻结、更新态协议、
真实DP4 LoRA+head更新恢复和rollout策略激活。
