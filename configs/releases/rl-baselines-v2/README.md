# 统一RL配置基线 v2

本目录登记现有arithmetic-v2的算法角色、共享配置和证据，复用既有训练器。
不替换旧context-prob-v1锁或实际运行包；不因本目录生成而授予训练许可。

| 配置 | 用途 | 已有证据与边界 |
| --- | --- | --- |
| grpo_v1 | 主线，组内中心化及标准差归一化，无critic | 634更新独立审核通过；完整selection收益仍待评测 |
| rloo_v1 | RLOO-clipped对照，无组标准差归一化 | 已有正式包和音频更新验收；不能称全量效果已验证 |
| ppo_critic_design_v1 | 学习value baseline的预留方案 | 未实现、不可派发；需value训练与恢复验收 |

[profiles.json](profiles.json)沿用既有G16、联合动作clip0.2/dualclip3、KL0.01和
辅助排序/NLL各0.3→0.1的配置。算法原始来源见各profile的paper字段；这里是有限动作
Jev适配，不能称论文完整训练配方复现。RLOO-clipped不是原始无裁剪REINFORCE-RLOO。

GRPO/RLOO的共享actor模板仅algorithm不同，统一使用同一宏协议及服务模板。
合作团队使用各实验独立的本地宏快照绑定模型、数据、起点策略、输出根和发布证明。
不能仅修改algorithm便复用GRPO的release/pins启动RLOO；需绑定对应已发布包。
模板只生成执行配置，现有打包、验收及launcher继续负责派发。

```bash
PYTHONPATH=src:. python -m tools.configure \
  --config configs/portable/training/context-rloo-actor-v1.json \
  --local .local/config/macros.json \
  --output .local/config/resolved/context-rloo-actor-v1.json
sha256sum -c configs/releases/rl-baselines-v2/SHA256SUMS
```

先完成任务定义及比较设计，再绑定工程/镜像与实验配置、通过有限受影响链路验收，
最后训练。任务输入、奖励、loss、数值精度、batch或优化器的变更创建新实验身份，
不改变正在运行/排队的依赖或拼接曲线。GRPO/RLOO共用Stage1起点、人口、顺序、
G16、更新数、优化器、辅助项和完整评测点；分别报告NPU时间，不能称计算成本相等。

step-wise首轮继续使用预先登记的三组：Direct60、Q-input124、Q-credit124。
Q-input/Q-credit的输入、动作支持、KL、辅助项一致，仅后者奖励增加
0.25×(scenario_F1+action_F1)/2。过程gold是原生标签投影，不是独立推理标注。
所有组只给Final辅助排序和NLL，不能额外加入过程监督损失使对照失效。
设计详见[原比较配置](../../experiments/context-native-attribute-comparison-design-v1.json)。

一次并行输出Q/Span不构成时间状态转移，不沿问题顺序用GAE。先保留标准估计器，
用完整Final曲线检验过程奖励；若出现组内标准化放大过程奖励、联合裁剪饱和或无奖励
变化，再分别登记分项归一化/信用分配/裁剪变体。每次只改一个机制，不能在现有曲线
中途切换。前置prob未回灌Final，现有方案不宣称已实现答案条件化的决策链。

简单任务出现明确全量selection收益后，复杂任务还需自己的124节点loss/KL、真实音频
训推一致性、更新及恢复验收；PPO不作为简单基线的额外前置条件。

[verification.json](verification.json)记录两模板与现有包的逐字段等价检查（仅派发false），
以及部分关键源码与包清单的hash核验；它不是所有传递依赖或新主机的完整认证。
SHA256SUMS只锁定所列文件；发布后保留本版本，修改时建立后继版本。
