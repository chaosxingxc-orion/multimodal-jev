# Context RL 历史候选归档（2026-11-14 至 2026-11-16）

本文件收编 `context-*` 主线中已被后续版本取代或仅作前置证据的候选。原始 md 已移入
`archive/raw/context-rl-history/`；运行/验收 JSON 收据仍在原路径保留。

## 做了什么

- 原生 verl actor DP4、LoRA 加载、在线 runtime、reference cache、token 契约。
- 批量/分段 rollout、整批 owner、service 候选、checkpoint 策略导出。
- 全日程训练循环、跨进程恢复、初始策略准入、命名 runtime。
- 2026-11-14 的训练效率复查（热缓存、实验周转、吞吐前缀）。
- step256 原生指标语义与独立 checkpoint 审核。

## 有效结论

| 主题 | 结果 | 当前状态 |
| --- | --- | --- |
| 原生 actor / LoRA / 恢复 | DP4 更新循环、LoRA 加载、跨进程恢复链路验证 | 作为当前 runtime 的前置证据保留 |
| reference cache | 固定 reference 缓存可复用；热命中减少重复前向 | 当前仍有 [reference 复用计划](../context-reference-reuse-plan-v1.md) |
| 批量/分段 rollout | CPU 组件与门禁通过；真实 NPU 收益未验收 | 候选未晋升为正式基线 |
| 全日程 / 恢复 | 全日程训练循环与跨进程恢复验收 | 支撑正式训练 runbook |
| 2026-11-14 效率复查 | 热缓存与周转分析 | 已被 2026-11-16/17 声学+KV16 对照取代 |
| step256 审核 | 独立 checkpoint/指标语义核验 | 历史 checkpoint 证据，不是最终效果 |

**结论：这些候选建立了 context 主线的 actor、缓存、恢复和审核基础；具体吞吐候选大多未准入，
不能把 CPU 门禁或单窗口观察当作 NPU 加速或任务收益。**

## 失败/过期路径

- 把热缓存/单窗口吞吐当作整轮 wall 吞吐或算法加速。
- 把分段/批量候选的 CPU 通过当作真实 NPU 收益。
- 使用旧的全量训练输入 v1 代替已发布的 v2 输入。

## 保留证据

- `context-*-cpu-*.json`、`context-*-npu-*.json`、`context-arithmetic-*-independent-*.json` 等原收据。
- 当前活跃文档保留在 `docs/engineering/`，见 [工程索引](../README.md) 第 4 节。
