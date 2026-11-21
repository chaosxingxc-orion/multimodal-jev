# 69 临时补跑 Stage2 训练视图

此四卡候选在派发 NPU 前已停止，由 [八卡 v0.0.139](stage2-train-view-host69-v0.0.139.md) 接替。

2026-11-19 用户临时授权 69 的八张 NPU。本次用 4–7 做 Stage1 rc7 step 318 的 TRAIN256 复验，随后用 6–7 的两个 TP1 服务为 40,513 条 clean TRAIN 打分。181 仍是后续实验默认目标；其 SLURP 评测独立推进。旧 69/175 队列不恢复。

181 的 v137 已通过 TRAIN256 数值验收。全量打分随后退出：reference 构建器误用只支持本地盘的音频镜像类，拒绝了符合规约的 Ceph 只读挂载。v138 使用独立只读入口，校验挂载类型和只读标记，并逐条校验音频哈希；缓存缺失或输入与独立证据不符仍会失败。热状态放 69 的本地 ext4，Ceph 仅供读。数据后继遇到失败终态时保留上游错误。

本地 88 项针对性测试通过。69 的固定镜像用一条实际 TRAIN 音频核验了 `ceph, ro` 和内容哈希，未挂 NPU。v138 冻结了新源码及运行时身份，必须复验本机数值；旧收据只用于追溯。

操作根：`/srv/multimodal-jev-xingchao/operations/stage2-train-view-host69-v0.0.138/`。其中 `audio-probe.json` 记录容器读取证据；`reference-chain/` 记录链状态，`parity-operations/` 和 `reference-operations/` 记录实际句柄、独立收集及审核；通过后由 `dataset-successor/` 自动发布并独立复算 `stage2-difficulty-view/`。采样及预算沿用 [181 交接](stage1-endpoints-host181-v0.0.136.md)：固定 seed、阈值和概率间隔，两轮预算按真实样本数与 global batch 384 计算。不派发 Stage2 训练。
