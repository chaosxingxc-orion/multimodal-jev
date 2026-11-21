# 181：Stage1 两端评测和 Stage2 数据准备

> 分工更新：本轮全 TRAIN 打分及视图构造由 [69 八卡链](stage2-train-view-host69-v0.0.139.md) 接手。181 保留端点评测与后续 Stage2 的 DP6 + 两个 TP1 计划；视图经独立审核后按清单绑定到 181。下表保留原迁移安排供追溯。

2026-11-19，181 成为后续试验默认目标。训练来源固定为审核通过的 Stage1 rc7：40,513 条 clean SLURP TRAIN、两轮、318 步。迁移的是 step 0 和 step 318 的正式对照及 Stage2 训练集构造；69/175 原任务保持暂停。执行在新目录和新容器，旧发布不改写。

| 181 卡位 | 新任务 | 依赖 |
| --- | --- | --- |
| 0–1 | FSC native31：3,793 条 × 2；完成或退出后交给 MINDS native14：8,168 条 × 2 | 全量编码及独立缓存审核；各端点 256 条无标签同宽度数值验收 |
| 2–3 | SLURP clean TEST60：12,991 条 × 2 | 同样的缓存及数值门禁 |
| 4–7 | rc7 step 318 TRAIN 数值验收、全 TRAIN reference（reference 服务用 6–7） | 181 自己的 HF/vLLM 数值及独立收集审核；通过后发布难例+简单回放 |

SLURP/FSC/MINDS 的 CPU 编码可并行；每套数据的两端同机、同卡、同候选顺序使用 `jev-stable-prefix-audio-readout-v1`。推理和数值验收不读取真值。FSC/SLURP 在原机的有限验收不计作 181 准入；旧 MINDS step 0 有 2/256 条 Top‑1 变化、0 条阈值翻转。181 复验若不一致，保留音频输入、logit 和批次身份，在无标签面板定位问题，未通过不得下发全量。

端点主报告使用每条音频最高候选的原始 sigmoid 概率的均值、单标签 Top‑1 macro‑F1，附 Top‑1 准确率。概率均值描述模型给自己首选打出的分数，无需阈值，也不等于正确率。Agent harness 的阈值由源域 TRAIN 校准；冻结评分的固定 0.5 指标仅供诊断。TEST 和迁移集不用于改阈值、选模或挖掘难例。

Stage2 数据：69 负责 NPU 打分，控制端收集后按 DP6 配置采样。只用 step 318 对完整 clean TRAIN 的独立审核 reference；冻结视图按清单部署到 181。采样使用 seed 317、阈值 0.5、概率间隔 0.1、简单回放比例 0.2、每类最少 16 条可用简单例。两轮按冻结人口 N 重算真实曝光 2N、更新 2×ceil(N/384)；尾批零权重补齐，单列真实分母。无需等三套 held-out 评测才开始构造。全量复算预计保留 19,379 条、102 步；正式发布受 69 服务关停故障阻塞，见 [本轮准备及证据](stage2-preflight-host181-20261119.md)。

Stage2 下一轮研究：对同一初始化和冻结视图，预登记基线双信用 GRPO 与集合效用和概率间隔候选。正式试验先做主线；如需把变化归因到某项 loss，再在相同数据、初始化、seed、曝光、LoRA 和 runtime 下做单因素对照。源域以完整冻结 selection 曲线、集合 F1 和候选 NLL 决策，检查 Brier、过度自信负例、漏选和 Top‑1 退化；配对不确定性按同一录音重采样。Stage2 在 181 用 0–5 训练、6–7 推理，global 384、micro8；先做两步有限验收、新进程恢复、第一轮真实尾批及下一轮首批的独立审核，再决定正式派发。本次只迁移评测和数据任务。

运行根用 181 用户家目录下本地 ext4；/data 现场余量约 111 GB 且接近满盘，家目录本地根余量约 221 GiB。模型、共享 TRAIN 音频及特征走只读挂载。固定镜像已安装，181 为 910B4；部署前按实际 Docker 挂载、空间、进程和租约核验。采购镜像或 CPU 测试不视作 NPU 数值证明。

- 本地宏：`EXPERIMENT_TARGET_HOST`、`EXPERIMENT_TARGET_HOT_ROOT`、`EXPERIMENT_TARGET_IMAGE`、`EXPERIMENT_TARGET_DRIVER_ROOT`、`EXPERIMENT_TARGET_LEASE`、`STAGE2_VIEW_TOPOLOGY`、`STAGE2_VIEW_ASSETS`。旧宏中的历史收集路径不变。
- 操作根：`/srv/multimodal-jev-xingchao/operations/stage1-endpoints-host181-v0.0.136/`。三组 `{fsc,slurp,minds}/{prep,score}/run` 记录准确容器，`reference-chain-v137/` 记录 TRAIN 链，`stage2-difficulty-successor/` 记录数据终态。v136 的第一份 TRAIN 驱动缺本地 safetensors，在封包前退出；v137 绑定原已验证依赖，另用新输出。
- 基座收据：`base-source.json`、`cpu/report.json`。181 固定镜像的动态视图/恢复/尾批/DP6 封包测试 9 项通过，无 NPU 挂载。正式评测、reference 和数据发布状态以对应的独立审核结果为准。
- 迁移后首批 NPU 证据（2026-11-19）：FSC 与 SLURP 的 step 0、318 各 256 条通过独立验收，logit 最大绝对差为 0，Top-1 与阈值翻转均为 0，见各自 `score/run/audit-parity-*-collected.json`。TRAIN v137 在 181 的 4–7 卡启动，句柄见 `parity-operations-137/jev-rc7-train-parity-137-launch.json`。这批证据不含全量评测分数或 Stage2 视图结果。

2026-11-19 已收集的 Stage1 端点结果（step 0 → step 318）：SLURP 和 FSC 两端点均完成全量独立审核，见各自 `score/run/audit-full-{0,318}-collected.json` 和 `top1-report.json`。

| 评测集 | 样本数 | 平均 Top-1 概率 | Top-1 macro-F1 | Top-1 准确率 |
| --- | ---: | ---: | ---: | ---: |
| SLURP clean TEST60 | 12,991 | 57.38% → 70.45% | 0.30% → 76.63% | 3.55% → 80.92% |
| FSC native31 | 3,793 | 59.16% → 37.94% | 1.28% → 0.56% | 4.35% → 8.38% |

FSC 的 [Top‑1 独立复算](/srv/multimodal-jev-xingchao/operations/fsc-top1-refresh-host181-v0.0.159/top1-rescore.json) 对每条录音按最高概率取一个标签；两个检查点各 3,793 条均计分，没有空预测。仅在固定 0.5 阈值的集合诊断中，step 318 有 3,134 条空结果（82.63%），step 0 为 0 条。step 318 将 3,379 条预测集中到同一个类别，只有两个类别有正确预测，因此总体准确率上升时，按 31 类平均的 macro‑F1 仍下降。

MINDS step 0 在 256 条无标签面板中出现 2 条 Top‑1 变化、0 条阈值翻转，最大 logit 差 0.03515625，尚未进入全量。应先在原面板定位差异，核验两个端点后再补全量。
