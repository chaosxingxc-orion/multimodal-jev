# Stage2 训练前核对：181

2026-11-19，0.6B Stage2 视图已发布：按 Stage1 rc7 step 318 对完整 clean SLURP TRAIN 的实际打分，采样保留 **19,379 / 40,513（47.83%）**，覆盖全部 60 个意图。原打分容器关停失败后，69 只重放最后一批 65 条，其 logits 和概率逐项一致，其余 40,448 条直接复用。独立恢复审核核对了全部 159 批、40,513 条输入和两次容器的真实退出状态。181 已接收并校验三份 reference 文件。

| 样本分组 | 全量 | 保留 |
| --- | ---: | ---: |
| 错误（严格 Top-1 或固定阈值的集合判错） | 13,425 | 13,425 |
| 边界（判对但概率间隔或双侧约束不足） | 2,078 | 2,078 |
| 简单 | 25,010 | 3,876 |
| 合计 | 40,513 | 19,379 |

采样只消费 clean TRAIN；selection、TEST、FSC 和 MINDS 均不参与。保持目标绑定的 0.5 阈值和 0.1 间隔，以 seed 317 确定顺序。简单回放为 `ceil(15,503 × 0.2 / 0.8) = 3,876`；每类 16 条可用简单样本的保底共需 916 条，一类无简单样本，未抬高总配额。未复制真实样本。13,425 条“错误”中，6,108 条严格 Top-1 判错，7,317 条严格 Top-1 正确但固定阈值集合判错。这一口径用于本轮采样，与 agent harness 的 TRAIN 校准阈值及无阈值端点评测分开。视图保留全部类别，但不对类频率重新加权。

181 按 0–5 训练、6–7 各一个 TP1 推理实例；global batch 384，microbatch 8，两个 epoch 各 51 步，合计 102 步、38,758 次真实录音曝光。两轮尾批各 179 条真实录音、205 个零权重补位。复算验证了每条入选录音每轮恰好一次；当前配置的 selection 检查点为 0、64、102。只在有限 NPU 验收测得真实步时后估算训练时长。

181 现场已核对固定镜像已安装，模型、共享音频及特征缓存可读，用户目录为本地 ext4、当时剩余 214 GiB；现场检查时八卡无在跑任务。跨机封包支持在 `reference_destination` 声明目标端的 reference 目录，仍核对原始 manifest、population、records 的 SHA-256；数据和元数据在训练里只读。此检查和 73 项定向 CPU 回归未占用实验卡。

181 的派发控制器 `tools.reports.context_stage2_view_pipeline` 从已审核 reference 和已发布视图开始：重审父包，装配目标插件，再绑定同一视图到该父包配置。然后按两步前缀、新进程恢复、真实尾批和下一轮首批的顺序执行有限验收；三份独立收据齐备才准入正式 102 步。视图 SHA-256 为 `d9a4036f06d537794670945feb287c2a3c089a8cc139c53746218dc4ebb86230`。历史 DP2 收据不可用于本轮。容器启动与训练进度以现场核对为准。

v0.0.147 的前缀容器退出码为 1：启动器在专有模块尚未进入 Python 导入路径时请求导入视图模块，发生在服务启动和训练更新之前。v0.0.149 从封包内按 SHA-256 加载该模块，在原视图、绑定收据与父包身份保持一致的条件下生成新运行包。固定镜像 9 项检查通过，旧包及失败收据保留。

v0.0.150 复用已收集的 v0.0.149 两步前缀，在同一固定镜像的 NPU 上重放 768 条查询的目标函数轨迹，loss 严格一致（容差为 0）。恢复验收从第 1 步提交的检查点在新容器中重放第 2 步，六个 rank 的参数、Adam、BF16 执行、采样、固定 reference 和目标函数轨迹均精确一致。尾批及下一轮首批分别验收了 179 和 384 条真实查询；尾批 205 个补位不进入分母。三项独立审核见 [前缀](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/pipeline/prefix/audit.json)、[恢复](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/pipeline/recovery/audit.json)和[尾批](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/pipeline/tail/audit.json)。

2026-11-19 18:24 UTC，181 的 102 步正式容器已启动，现场按[派发收据](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/operations/formal/jev-stage2-view-formal-150-launch.json)核对为 running、未 OOM。约 18:30 UTC，[首个正式更新](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/formal-first-update.json)确认六个训练 rank 完成 384 条查询，两个推理实例均确认加载更新后的策略。正式训练结束后，由同一控制器收集并独立审核；首步记录不含效果结论。

2026-11-19 21:40 UTC，六个训练 rank [到达第 102 步并提交最终检查点](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/operations/formal/jev-stage2-view-formal-150-observations/1795124334559799750-result.json)。同一容器的 [wait 收据](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/operations/formal/jev-stage2-view-formal-150-wait.json)与 [inspect 收据](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/operations/formal/jev-stage2-view-formal-150-inspect.json)均确认退出码为 0，未 OOM。控制器随后收集产物，完成量以独立审核收据为准。

2026-11-19 22:16 UTC，[正式独立审核](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/pipeline/formal/audit.json)通过。两轮各 19,379 次真实录音曝光，共 **38,758 次、102 步**。训练容器用时 3 小时 15 分，正式训练中每步最慢 rank 耗时的中位数为 111.54 秒；收集和审核在训练退出后另用约 37 分钟。[最终检查点](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.150/operations/formal/jev-stage2-view-formal-150-collected/run/checkpoint-0102-committed.json)的 SHA-256 与传输清单一致。预登记评测点为 0、64、102；当前结论仅覆盖训练完整性，评测效果尚未验证。

2026-11-19 22:38 UTC：[selection 评测控制器](/srv/multimodal-jev-xingchao/operations/stage2-selection-host181-v0.0.156/campaign/run.started.json)已启动。封存的 [9,226 条评测输入与三个策略](/srv/multimodal-jev-xingchao/operations/stage2-selection-host181-v0.0.156/selection-seal/MANIFEST.json)来自同一 SLURP clean selection 和这轮实际训练。step 0 是 Stage1 rc7 的 step 318，step 64/102 是同一轮 Stage2 的已提交 LoRA/head。先在 CPU 生成并独立校验稳定前缀张量；各点通过 256 条同策略训推比对后，四个单卡推理实例复用完整输入。每点全量预测独立重核后报告平均 Top-1 概率及 Top-1 macro-F1；固定阈值仅用于诊断。此记录是评测启动证据，不含效果。

2026-11-20 00:24 UTC，[完整 selection 曲线](/srv/multimodal-jev-xingchao/operations/stage2-selection-host181-v0.0.156/report/curve.json)已通过独立审核。三个点均覆盖同一批 9,226 条录音、60 个意图，各自的 256 条训推比对逐项一致。收集器核对了原评测容器的终态和各点的预测审核，曲线与支撑收据的 SHA-256 一致。

| 检查点 | 平均 Top-1 概率 | Top-1 macro-F1 |
| --- | ---: | ---: |
| 0（Stage1 终点） | 71.02% | 77.21% |
| 64 | 91.36% | 78.87% |
| 102 | 89.34% | 79.37% |

终点 macro-F1 较 Stage1 终点高 2.16 个百分点。平均 Top-1 概率在 step 64 达到三点中的最高值，到 step 102 回落 2.03 个百分点，macro-F1 同期继续增加 0.50 个百分点。平均概率不能单独证明校准改善；此结果只覆盖 SLURP selection，不外推 TEST、FSC 或 MINDS。

11 月 20 日，step 102 的冻结外域和测试评测另行启动。v158 将 FSC native31 分配到 0–1 卡、MINDS native14 到 2–3 卡、SLURP clean TEST60 到 4–7 卡。FSC 与 SLURP 的 256 条一致性面板均 exact 并开始全量；MINDS 的最大 logit 差为 0.4609375、最大概率差为 0.04342609，Top-1 与阈值翻转均为 0，服务路径停止在门槛。v160 在 2–3 卡上各用独立 HF 进程精确复现原 256 条 HF 面板的一半，随后全量分片评测；报告中明确标识 HF 路径。见 [MINDS 比较收据](/srv/multimodal-jev-xingchao/operations/stage2-minds-hf-host181-v0.0.160/evidence/comparison.json)和 [HF 派发记录](/srv/multimodal-jev-xingchao/operations/stage2-minds-hf-host181-v0.0.160/package/run.started.json)。

FSC step 102 [已审核完成](/srv/multimodal-jev-xingchao/operations/stage2-heldout-host181-v0.0.158/fsc/report/curve.json)：3,793 条，平均 Top-1 概率 **46.80%**，Top-1 macro-F1 **2.78%**（准确率 8.91%）。同人口的 [Stage1 端点](stage1-endpoints-host181-v0.0.136.md) macro-F1 为 0.56%；概率按各自模型的最高候选取均值，固定 0.5 仅供诊断。

SLURP clean TEST step 102 [已审核完成](/srv/multimodal-jev-xingchao/operations/stage2-heldout-host181-v0.0.158/slurp/report/curve.json)：12,991 条，平均 Top-1 概率 **89.15%**，Top-1 macro-F1 **78.25%**；同人口的 Stage1 端点 macro-F1 为 76.63%。

MINDS native14 step 102 [已审核完成](/srv/multimodal-jev-xingchao/operations/stage2-minds-hf-host181-v0.0.160/package/run/audit-hf-shards-collected.json)：独立 HF 路径覆盖 **8,168/8,168** 条，平均 Top-1 概率 **35.51%**，Top-1 macro-F1 **62.01%**，Top-1 准确率 **60.97%**。0.5 阈值仅供诊断。此收据确认完整的 HF 评测；v158 的服务路径仍未通过精确一致性验收。MINDS 是外域评测，不参与 Stage2 样本构造或训练准入。

复跑用 [封存配置](../../configs/portable/operations/seal-stage2-view-selection.json) 与 `tools.reports.seal_stage2_view_selection` 绑定正式窗口、独立审核、终态、原数据发布的 SHA-256。再用 [评测准备配置](../../configs/portable/operations/prepare-stage2-view-selection.json) 与 `tools.reports.prepare_stage2_view_selection` 绑定输出、冻结评测工程和设备，核对生成的 `PREPARED.json` 与归档哈希后，将生成的 `operations-local.json` 传给 `tools.reports.native_seal_operations --detach`。最后用 [曲线收集配置](../../configs/portable/operations/report-stage2-view-selection.json) 与 `tools.reports.report_stage2_view_selection` 等待原控制器终态并核对三点审核；任一点缺失时不发布曲线。封包、容器及审核使用新目录，复用完整 selection 人口。

- [本轮采样预览收据](/srv/multimodal-jev-xingchao/operations/stage2-preflight-host181-20261119/sampling-preview.json)：保留量、各类别计数、固定 reference 身份及预算。用 [预览配置](../../configs/portable/operations/stage2-dataset-preview-dp6.json) 和 `prepare_stage2_dataset --action preview` 复算。该动作仅写汇总收据，不生成视图、训练配置或发布签据。
- [69 的原始打分](stage2-train-view-host69-v0.0.139.md)、[独立恢复审核](/srv/multimodal-jev-xingchao/operations/stage2-reference-replay-host69-v0.0.141/independent-recovery-v145/audit.json)和[181 的正式发布收据](/srv/multimodal-jev-xingchao/operations/stage2-host181-v0.0.147/prepare-view.json)。本次恢复针对结果已提交、关停或终态写入失败的情况；通用中断续跑需独立验证。
