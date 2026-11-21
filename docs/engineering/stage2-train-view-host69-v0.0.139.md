# 69 八卡全量 TRAIN 打分与 Stage2 视图

2026-11-19 用户指定 69 的八张卡统一用于训练集准备。v0.0.139 取代尚未启动容器的 v0.0.138 四卡候选。69 的 0–7 卡各运行一个 TP1 实例；每个完整调度批为 256 条，分成 8 组每组 32 条，每卡一组。冻结的 Stage1 rc7 step 318、stable-prefix 输入、完整 40,513 条 clean SLURP TRAIN 及 60 候选顺序保持一致。共享音频和特征只读，产物和日志写本地 ext4。

先用同一运行时、8 实例配置复验独立选取的 256 条 TRAIN：HF 对照阶段用单卡，随后 8 个推理实例各处理一组，共同完成同一 256 条的精确数值检查。通过独立审核才启动全量打分。最后收集并审核完整记录与批次日志，由控制端 CPU 构造、复算并冻结难例和简单回放视图；不使用测试集或迁移集。采样规则沿用 [181 交接](stage1-endpoints-host181-v0.0.136.md)，未来 DP6 训练日程仍按实际冻结视图人口和 global batch 384 计算。

本地 172 项相关测试通过，包含八卡封包、逐卡绑定、数值与数据审核路径。Ceph 只读修复在 [v0.0.138 的固定镜像检查](stage2-train-view-host69-v0.0.138.md) 验证可读取真实 TRAIN 音频并核对哈希。

操作根：`/srv/multimodal-jev-xingchao/operations/stage2-train-view-host69-v0.0.139/`。`reference-chain/` 记录链状态；`parity-operations/` 与 `reference-operations/` 记录准确容器句柄及独立审核；`dataset-successor/dataset-audit.json` 和 `stage2-difficulty-view/MANIFEST.json` 是视图发布证据。181 仍是后续实验默认目标，这条链不派发 Stage2 正式训练。

启动里程碑（2026-11-19，北京时间）：八卡 TRAIN256 独立数值验收通过后，全量容器 22:59:22 启动。23:04:01 提交首个正式批次，八个实例各处理 32 条。核验记录为操作根下的 `first-formal-batch-observed.json`，与全量结果审核和视图发布分开记录。

全量终态（北京时间 23:37）：准确容器 `cde42bb41233` 已退出，ExitCode 1、无 OOM。159 个批次、40,513 个独立输入全部生成并收集；builder 以 0 退出。服务 h 的 vLLM EngineCore 达到 30 秒关停期限后被强制结束，外层 supervisor 也记录了 `forced_cleanup=true`。现行准入要求成功退出且无强制清理，因此 reference 与训练视图未正式发布。原收据和冻结包保留；独立批次日志复核、采样结果及 181 准备见 [Stage2 训练前核对](stage2-preflight-host181-20261119.md)。
