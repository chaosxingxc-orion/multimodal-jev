# 实际音频 GRPO/RLOO 四卡验收

2026-11-14，容器883b000ec8c3603a591635a937ae14c1b271a8e2b9650cef9dc05820f58f96a8
已在175物理4–7运行，固定镜像52ba94e1、Qwen3-ASR-0.6B、BF16 LoRA/head，
FP32 master/Adam、DP4/global128/micro8×4。准确wait12929持评测共享锁，
独立收集驱动PID288265等待同一容器，不重复派发。

## 行为策略与实验范围

初始化采用已有DP4更新后完整joint policy
adc979f192213499e8e444b2db890cfc478b95c16f6636a443a2479607443567，
其256条learning实际音频×60候选HF/vLLM概率全0差已独立验收。
本次复用该次真实vLLM输出，逐请求绑定policy ID、实际输入tensor hash、marker位置
及engine request ID。不是HF伪造rollout，不在更新后继续使用过期行为策略。

`context_rollout_batch.bind_rollout`由单次音频概率向量采样G16完整60bit动作；
CPU seed=317+固定输入序号。同一组动作用于GRPO/RLOO，reward为完整集合F1。
独立sigmoid的factorized Bernoulli仅定义探索分布，不宣称语义事件独立。
边界概率0/1拒绝，不靠clamp改变行为分布。

准备完整256输入，实际更新使用固定前128条：

1. 从固定joint policy、fresh Adam执行一次GRPO更新；
2. 恢复相同初始master/fresh Adam，在相同输入及动作执行一次RLOO更新；
3. 再恢复初始状态，重放GRPO，与第1项逐参数/Adam精确比较。

每个算法只更新一次；这不是GRPO连续两步，也不是正式两阶段训练结果。
KL参考固定为该次验收初始policy，系数.01；辅助ranking/NLL各.3、lr1e-4、clip.2。
该参考只用于单步工程验收，不冒充已完成正式Stage1的参考模型。

## 数值预算

vLLM公开输出是sigmoid概率，behavior用FP32 log1p(-p)/log(p)转换；actor直接
使用log-sigmoid，数学相同但FP32路径可能舍入不同。仅该独立验收预先登记：
probability_max_abs=2e-7，binary_logprob_max_abs=2e-6，0.5阈值翻转必须0。
60位最坏完整动作log-ratio偏差上限因此1.2e-4。不是根据失败结果事后调阈值，
不是生产准入阈值；已有instruction/正式候选的零误差配置不变。

仅CPU对已记录概率作logit roundtrip诊断，max logp差1.1921e-7，FP64归一化残差
2.96e-8；没有实际actor logits，不能把它当HF/vLLM logp一致性证据。实际NPU
每个micro必须通过集体behavior检查后才允许optimizer更新。

## 验证与终态收集

13项本地rollout绑定和behavior parity测试通过，覆盖旧policy、音频/位置变化、
request ID不匹配、饱和/NaN概率拒绝及采样复现。固定镜像原生GRPO/RLOO源码
hash仍fadfafa8；GRPO使用样本std+1e-6，RLOO使用leave-one-out均值，源码记录
`.local/decision-objective-research/context-rl-native-estimators-v1.json`。

collector `collect_context_rl_actor_dp_npu_v1.py`在终态后核对所有文件/pins，
实际输入标签、四rank FP32 master/Adam和BF16导出，重算G16采样、F1、GRPO/RLOO
优势和joint logprob；GRPO重放状态另存并独立逐张量比较。优势代数等价重算
允许明确的FP32求和舍入误差，参数/Adam重放仍要求精确。

驱动`.local/decision-objective-research/continue_context_rl_probe_v1.py`准确wait→collect，
失败保留现场，不重启。状态`context-rl-probe-v1-status.json`，日志同前缀driver.log。
本轮尚未收到终态，不能称实际音频RL验收成功；在线策略导出/激活/新rollout的
连续生命周期、完整Stage1/Stage2收益曲线与FSC迁移仍待完成。

## 实际终态与独立复核

883b000e已exit0/nonOOM，独立collector通过，驱动PID288265进入accepted_bounded_rl_update。
证据`context-rl-actor-dp-npu-v1.json`：128×16采样、F1、GRPO/RLOO优势及joint logprob
独立复算通过，GRPO重放四rank master/Adam精确，128BF16完整导出跨rank一致。
三次执行各rank最大耗时27.148/24.184/22.578秒，最大allocated35.326GiB；
这是验收窗口，不能当稳定RL吞吐。记录的behavior检查最大概率误差8.95475e-8、
binary logp误差1.78814e-7、joint logratio bound4.29153e-6，阈值翻转0。
collector报告的behavior记录为各actor最后micro汇总，代码对每micro均先集体验证才更新，
不能将报告中的最大值称为已保存全micro逐项最大值。

后续b01da14a在175物理7卡运行HF/vLLM256，消费本次GRPO完整BF16导出。
新导出contract明确request_version=jev-context-exclusive_event-v1，runtime为命名
qwen3-asr-context-bf16-packed-row-v1，token仍test_only等待最终准入，历史manifest不改。
准确wait27582持共享锁，独立collector驱动PID292736，状态context-engine-rl-v1-status.json。
新tag context-engine-rl-updated-npu7-v1，本轮未有终态，不预判概率差为0。

新增`context_event_metrics.py`汇总固定60候选exclusive-event指标，
`evaluate_context_event_records.py`只接完整selection人口、单policy和完整engine身份。
它不独立重跑音频processor，调用方仍需验证实际输入/策略资产，不能只靠record哈希
自洽证明真实请求正确。14指标tests过，包含集合与top1区别、空预测、固定顺序tie、
未出现类别宏F1、非法概率及重复请求。尚无新模型完整selection结果。
