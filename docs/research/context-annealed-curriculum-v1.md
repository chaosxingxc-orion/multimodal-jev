# 退火课程 v0.0.4：可导 SFT 到采样奖励

> 状态：`preregistered_not_dispatched`。本文件只登记设计、口径与日程，不是运行包、
> 验收记录或效果结论。v9 双信用 RL 仍在运行且冻结，本设计不改其包、pins 或产物。

## 1. 三段定义

一次 optimizer update 由三个权重控制：

| 段 | 权重 | 是否可导 | 信号来源 |
| --- | --- | --- | --- |
| supervised（SFT） | 0.5 → 0 | 是 | 模板/表征初始化的 NLL/InfoNCE |
| contrast（对比） | 0.3 → 0.4 | 否 | 同一轨迹内 gold×非 gold 配对 + margin |
| ranking（排序） | 0.2 → 0.6 | 否 | listwise 三档排序的标量奖励 |

三项在任意 update 构成凸组合，和为 1，端点精确。对比、排序、概率校准都不是 loss：
它们只在采样结果上产生标量，经 GRPO 组内居中、std floor 和比例裁剪后反传。
只有 SFT 项直接可导。实现入口为 `src/multimodal_jev/context_curriculum_credit.py`。

## 2. 为什么奖励必须带采样随机性

GRPO 的组优势是 `r_i - mean(r)`。若奖励仅由 query 与当前策略的完整概率向量确定性
算出，则同一组的 G 个样本得到完全相同的标量，居中后优势恒为 0，KL/裁剪都不会产生
学习信号。

现在三项奖励统一到**同一条轨迹**：每组从当前策略做一次 Plackett-Luce 无放回抽取，
得到 k 个有序候选；排序、概率、对比都是这条轨迹的函数，共用同一个精确 log-prob。
轨迹的抽样概率依赖当前策略，因此具备 score-function/比例裁剪所需的策略相关路径；
不再使用与策略无关的均匀子集或均匀负样本作为唯一随机源。`trajectory()` 一次返回
`draws / log_prob / rank / brier / margin`，避免三次独立抽样导致轨迹不一致。

## 3. 三项奖励的精确口径

**轨迹（trajectory）**：每组从当前策略权重抽取 k=8 个有序候选（菜单不足 8 时由调用方
显式给出合法 k）。`log_prob` 为
`Σ_j log(w_{c_j} / Σ_{i∉已抽} w_i)`，FP64 计算、形状 `[group_size]`、detached。
组大小 16，seed 317，生成器按 `(seed, update, query_id, channel)` 派生。

**排序（ranking）**：读同一轨迹的前 3 个位置。top-1 命中 gold 记 1.0，top-3 命中记
0.5，否则 0。gold 是完整标签集合，命中其中任意候选都算。

**概率（probability）**：`1 - mean((p - target)^2)`，在**同一轨迹的 k 个候选**上计算；
标签不参与轨迹抽取。奖励落在 [0, 1]。正式评测仍在完整菜单上报告 Brier。

**对比（contrast）**：在同一轨迹内取“抽中的 gold × 抽中的非 gold”配对，奖励为满足
`p_gold - p_neg >= 0.2` 的配对比例。轨迹内没有 gold 或没有非 gold 时该分量记 0，并在
`margin_reason` 返回 `no_gold_in_trajectory` / `no_non_gold_in_trajectory`，不抛异常、
不静默丢弃。

**合成与优势**：`R = w_c * (0.5 * margin + 0.5 * brier) + w_r * rank`，
`w_c`/`w_r` 来自日程；SFT 权重只乘可导 SFT 项，不进入 `R`。优势为
`(R - group_mean) / max(group_std, 0.25)`，算在合成分数上；候选再用同一个
`log_prob` 构造比例裁剪替代损失。模块本身不引入 autograd。

## 4. 日程表（212 updates）

| update | supervised | contrast | ranking |
| ---: | ---: | ---: | ---: |
| 0 | 0.50000 | 0.30000 | 0.20000 |
| 21 | 0.45047 | 0.30991 | 0.23962 |
| 42 | 0.40094 | 0.31981 | 0.27925 |
| 64 | 0.34906 | 0.33019 | 0.32075 |
| 85 | 0.29953 | 0.34009 | 0.36038 |
| 106 | 0.25000 | 0.35000 | 0.40000 |
| 127 | 0.20047 | 0.35991 | 0.43962 |
| 148 | 0.15094 | 0.36981 | 0.47925 |
| 170 | 0.09906 | 0.38019 | 0.52075 |
| 191 | 0.04953 | 0.39009 | 0.56038 |
| 212 | 0.00000 | 0.40000 | 0.60000 |

表内为四舍五入到 5 位；精确值由 `schedule(update, 212)` 给出。曲线点与
`configs/experiments/context-replica-training-dp6-formal-v1.json` 的 11 点一致。

## 5. 与 v9（常数混合双信用 RL）的对照

v9 在当前运行包内使用常数混合的 dual-credit GRPO，并保留 binary NLL 0.3 与
supervised ranking 0.1 两个可导辅助项。v0.0.4 的改动是：

1. 删除 `binary_nll 0.3` 与 `supervised_ranking 0.1`；
2. SFT 权重从 0.5 线性退火到 0；
3. 采样奖励混合从 contrast 0.3 / ranking 0.2 线性长到 0.4 / 0.6；
4. 概率奖励改为同一轨迹 k 候选上的 Brier 分，不再用 binary NLL 作辅助监督；
5. 排序、概率、对比共用一次 Plackett-Luce 轨迹及其 log-prob，不再各自抽样。

共享的是全量 clean SLURP 40513、DP6（训练卡 0–5、推理卡 6–7）、global batch 384、
microbatch 8、seed 317、2 epoch、212 updates 和 11 点 selection 曲线。v0.0.4 必须是
新版本候选包；不得原位修改 v9、其 pins、配置或正在运行的产物。

## 6. 仍未决的事项

- SFT 内部 NLL/InfoNCE 的 0.5/0.5 拆分只是占位，训练前必须冻结；它也决定“可导监督”
  退火时的实际梯度尺度。
- `tools/reports/context_curriculum_candidate.py` 已提供生成器与损失候选，但固定镜像
  CPU 验收、真实音频 rollout 一致性、DP 全局归约和 NPU 更新证据都不存在。
- 候选沿用 dual-credit 的采样算法做行为 rollout，只把注册损失替换为课程目标；轨迹
  仍由当前/行为策略的候选分布抽取，`log_prob` 复用同一条 `draws`。
- 轨迹长度 k=8 不得大于菜单大小；`brier_reward(subset=...)` 与
  `margin_reward(negatives=...)` 保留旧参数名但语义已变为轨迹长度，越界必须报错。
- 无 gold 或无非 gold 的轨迹分支必须计入诊断；`margin_reason` 是审计字段，不能把
  这些 0 分量当作正常配对样本。
- `log_prob` 为 FP64、detached；候选必须复用同一条 `draws` 构造比例，不得重新抽样。
- 因此本配置不可派发、不可进包；它只登记已批准的设计身份与已编译源码候选，等待
  固定镜像 CPU 验收和 NPU 证据。
