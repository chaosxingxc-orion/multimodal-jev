# 上下文长度与推理吞吐：v9 实测归因

2026-11-17。回答“推理慢是不是主要是上下文太长”。结论：**上下文确实长，且直接乘在算力上；
但 139 ms/forward 的 6–10 倍差距主要来自低 MFU（小 batch + eager），不是纯 FLOPs**。
两条都要修，但杠杆不同。所有数字来自 v9 冻结包的实测量，不是估算除非注明。

## 1. 真实 token 构成

从 `cached-context-inputs-cpu-v1` 的真实 SLURP 请求（`audio-1490967510.flac`，2.37 s）取一条：

| 成分 | token 数 | 说明 |
| --- | ---: | --- |
| 总计 `input_ids` | **2831** | 一条 query 一次 forward |
| 系统提示 | 36 | `system` + requirements JSON |
| 音频 | 31 | `<\|audio_pad\|>`（`audio_token_id=151676`，2.37 s） |
| 60 个问题块 | **2764** | 60 × ~46 token |
| 决策 marker | 60 | `[JEV_SPAN_i]`，`positions` 形状 `(1,60)` |

**97.6% 的 context 是 60 个候选问题块**，音频只占 1.1%。所以“上下文长”= 候选清单长，不是音频长。

### 冗余

60 个问题里：
- `id`：60 个互不相同（`alarm_query`、`weather_query`…），必要。
- `definition`：60 个互不相同（`Scenario: alarm; action: query.`…），必要。
- **`question`：60 个完全相同**——`Is the event assigned to this audio under the supplied annotation rules this candidate?`（约 20 token），重复 60 次 ≈ **1200 token，占全部 context 的 ~42%**。

这是可以安全去掉的重复：同一句指令在系统提示里说一次即可，60 个候选只保留各自的 `definition` 与 marker。
去重后 context 从 2831 → **约 1600–1700 token（−40%）**。这是任务模板的结构性修改，
语义等价但改变了 prompt 形态，必须作为新任务定义候选验收，不能混入 v9。

## 2. 为什么 2831 token 要 139 ms

v9 实测（`context-replica-overhead-analysis-v1.md` + 本次观测）：
- 每更新 384 query × 2 策略（reference + behavior）= 768 次 forward。
- rollout 107 s（reference 49.8 + behavior 57.6）→ **139 ms/forward**。

粗算 FLOPs（0.6B，2831 token）：
- FFN ≈ 2 × 0.6e9 × 2831 ≈ 3.4 TFLOPs；
- 注意力 ≈ 2 × 28 层 × 2831² × 1024 ≈ 0.9 TFLOPs；
- 合计 ≈ 4.3 TFLOPs / 0.139 s ≈ **31 TFLOPS 有效吞吐**。

Ascend 910B BF16 峰值约 350–400 TFLOPS，即 **MFU ≈ 8–9%**。所以：
- **算力不是瓶颈，利用率才是。**
- 降低 context 会把 FLOPs 按 L²（注意力）+ L（FFN）一起降，但若 MFU 不变，只拿到约 1.6–1.8×。
- 真正的大头是把 MFU 从 8% 抬上去。

### 低 MFU 的根因

1. **token 预算太小**：`max_num_batched_tokens=16384`，而每条 query 2831 token
   → 每批只装 **5–6 条**（`max_num_seqs=48` 不是限制）。宽核上 5 条序列占不满。
2. **`enforce_eager=True`**：无图捕获，逐 op Python 派发，在小 batch 下 overhead 主导。
3. **上下文长 → 批小**：这就是“上下文长”与低 MFU 的因果链——它通过限制 batch 生效，
   而不是直接吃满算力。

## 3. 修复杠杆（按杠杆排序）

### 3.0 更正（2026-11-17）：token 预算杠杆已被实测证伪

本文早先版本把“抬 `max_batch_tokens` 16384→65536”列为第 1 杠杆，并声称 batch32
对照“两组 `max_batch_tokens` 都是 16384”。**这一说法是错的**，已在权威产物上更正：
`service-batch32-100step-study-v1` 的基线是 `macro-efficiency-v1-prepared/micro8/package/service-0.json`
（`max_batch_queries=4`、`max_batch_tokens=16384`），候选是
`service-batch32-v1-prepared/package/service-0.json`（`max_batch_queries=32`、**`max_batch_tokens=131072`**）。
即 batch32 已经把 token 预算抬到 **8×（16384→131072）**、query 抬到 8×（4→32），
且是在 `gpu_memory_utilization=.4`（`context_switch_rollout.py` 硬编码）下跑通的，
结果只有 **+2.70%**（86.44 → 84.17 s/step）。推论：

- 只改 token 预算/query 容量 **不是** 有效杠杆；131072 已测得只值 2.7%，65536 比它更小。
- `gpu_memory_utilization .4→.6` 也**没有必要**：131072 token 在 .4 下已稳定跑完 100 步。
- 因此基于“抬 token 预算 + 抬显存利用率”的服务容量候选是死路，不再作为 v0.0.5 主线。

### 3.1 真正剩余的杠杆

服务批处理为什么抬不动？因为成本不在被批化的那段：

- **声学塔仍是逐录音串行**：`context_segmented_audio_candidate.segmented_audio_forward`
  明确“intentionally serial in the acoustic tower … never a padded batch”，
  `context_replica_batched_service` 只批量化了 **语言 prefill**（`score_many`），
  音频编码没有一起批量化。所以把 LM 的 batch 抬大，省下的只是 prefill 的一小段。
- **actor 占 48%**：同一 100 步里 actor 每步 ~40.6 s，rollout ~33–36 s。
  actor 完全不随推理 batch 改善，是第二大成本。
- **推理卡在 actor 期间空转**：actor 跑 ~40 s 时两张推理卡闲置。

| # | 杠杆 | 预期 | 风险 | 类型 | 实测状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | **actor 优化**（FSDP/microbatch/Python 派发开销） | 压 48% 的最大单项 | 中（改训练器） | 结构性 | 未测 |
| 2 | **音频编码批处理**（把 segmented 声学塔真正拼批） | 直接压 rollout 主项 | 中高（训推数值一致性） | 结构性 | 未测；必须单独验收一致性 |
| 3 | **reference 与 actor 重叠**（reference 策略冻结，可在 actor 期间预取下一步参考分） | 隐藏每步 ~15–17 s reference | 低（参考策略固定） | 流水线 | 未测 |
| 4 | 模板去重（`question` 只说一次） | context −40%，注意力 −45% | 中（改任务定义，需验收） | 结构性 | 独立任务候选，不并入工程基线 |
| 5 | 关闭 `enforce_eager` 启用图捕获 | 单次 forward 2–3× | 中高（Ascend 正确性未知） | 结构性 | 未测 |

**已证伪/降级**：`max_batch_tokens`/`max_batch_queries` 容量（已测 8× = +2.70%）、
`gpu_memory_utilization` 抬高（131072 已在 .4 跑通）。这两条不再作为效率候选的改动项。

### actor 是另一半

同一 100 步里 actor 占 **48.26%**，rollout 39.65%。actor 不随推理 batch 变好，
且与 rollout 串行——rollout 的两张推理卡在 ~40 s actor 期间完全空转。
所以 1 小时目标不能只靠推理端：必须同时压 actor（FSDP/microbatch/Python 开销）
并把 rollout 与 actor 重叠。

1 小时目标需要 ~8.8× 加速（v9 约 150 s/step，212 步要 17 s/step）。按 3.0 的更正，
单靠容量/服务 batch 已证伪。按 batch32 **实测 DP2 分解**（84.17 s/step：
actor 40.60 / rollout 33.41 [reference 14.96 + behavior+credit 16.51] / 残差 10.16），
单杠杆的保守预期（落在实测分量上）：

| 杠杆 | 命中成本 | 乐观节省 | 步时后 | 加速 | 风险 | 验证顺序 |
| --- | --- | --- | --- | --- | --- | --- |
| actor 优化（FSDP/microbatch/Python） | actor 40.60 s | 减半 20.3 s | 63.9 s | **1.32×** | 中（需先 profile） | 1 |
| reference 与 actor 重叠 | reference 14.96 s | 全隐藏 15 s | 69.2 s | **1.22×** | 低（参考冻结） | 2 |
| 音频编码批处理 | behavior+credit 16.51 s | 减半 8.3 s | 75.9 s | 1.11× | 中高（训推一致性） | 3 |
| 图捕获（enforce_eager=False） | rollout LM forward | 未知 | — | 1.1–1.3×? | 中高（Ascend 未知） | 4 |
| 模板去重 | 全 context 注意力 | — | — | ~1.2× | 中（独立任务候选） | 单独 |

按“预期提升最高优先”的验证顺序：**actor 优化（1.32×）→ reference/actor 重叠（1.22×）→
音频编码批处理（1.11×）→ 图捕获**；模板去重作为独立任务候选单列，不并入工程基线。
组合若相互独立（actor 减半 + reference 全隐藏）约 1.6×，离 8.8× 仍远。
**以上全部是估算，不是实测**；每个模块必须在独立 ≤100 step 候选上分别测量。
旧的“actor 1.5–2×”和“token 预算 4×≈2–3×”估算均作废。

## 4. 子代理模型说明

用户指定“子智能体使用 `deepseek-v4-flash-claude`”。本会话的 `Agent` 工具 `model` 枚举只有
`{sonnet, opus, haiku, fable}`，**接口不支持 `deepseek-v4-flash-claude`**。按 AGENTS.md
“若接口不支持该模型，应说明限制，不静默切换其他模型”，本效率分析与候选设计由主智能体直接完成，
不派发到未知模型。

## 5. 与 v9 的关系

- v9 是冻结的正式包，**不原位修改**；本分析只读。
- 效率候选（v0.0.5 开发态）只能等 v9 释放 8 张卡后独立派发，且每个配置最多 100 个训练 step。
- 本文不改变 v9 的 212 更新预算，也不把效率估计当作效果结论。
