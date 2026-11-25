# JEV-210：双向 InfoNCE Stage1 与融合 TPO Stage2

当前正式训练暂停。本轮工程验证使用真实 TRAIN 音频与知识卡、有放回采样，配置为 **1 个锚点、1 条额外同类正语音、3 张错卡、3 条异类语音**；保留两个 InfoNCE 方向。统一入口的 `capacity --experiment JEV-210 --capacity-variant real-buckets` 指向[独立有限配置](../configs/portable/operations/card-stage1-real-buckets-8-finite.json)，只封装 prepare 和 finite；需独立的本地包及部署绑定 `REAL_BUCKET_FINITE_PACKAGE`、`REAL_BUCKET_FINITE_DEPLOYMENT`。旧 `run/resume` 配置只供追溯，不用于新配方。

以下为此前正式训练方案。修订 `bidirectional-m16p3-packed-v1` 替代已停止的 `online-ema-v9`，沿用 JEV-210。以[统一入口配置](pipelines/JEV-210.json)和[双向采样与 loss 协议](stage1-card-bidirectional.md)为准。已授权在 181 八卡派发；启动验收须独立复核正式训练前三步。

- **起点与预算：**从 209 同源的冻结初始策略重新训练 0.6B、LoRA r128/α256，不从 v9 的 checkpoint 接续。沿用 clean TRAIN 109,610 条、68 类、两轮；181 八卡，每卡 16 个锚点，无梯度累积，全局 128。每个满批 1,792 个配对，两轮 1,714 步。
- **训练：**每组 1 个锚点＋3 个额外同类正样本＋5 个固定语音错卡负例＋5 个固定卡异类语音负例。有放回，仍排除锚点自配和真类负例。两个方向直接优化成员 logit，各占 0.5；同卡四个正例不互斥。新目标不含旧的概率间隔或双侧阈值项。
- **接续：**只复用身份一致的数据、声学缓存、提示和初始化。先做 4 次推理一致性检查和 2 次训练更新，再正式训练。第 857 步完成第一轮，同一训练进程携带模型、Adam 与 EMA 继续第二轮。轮间不设审核、重启或打分。两轮结束后用**新终点**打分构建 Stage2 难例及回放视图，沿用融合 TPO 训练。
- **验收：**记录双向 loss、同音频真假卡分差、同卡正负语音排序、锚点/秒。Stage1 和 Stage2 都报告 SLURP clean TEST、MINDS native14 的 Top-1 macro-F1 和准确率，并按原计划覆盖其他封存面板。主对照为 209 混合版，逐数据集报告配对差与区间。v9 的半程观察不作为正式终点。阈值只在源 TRAIN 校准。

[实验台账与旧版停点](../wiki/reports/JEV-210.md)。历史 v9 协议可在 Git 提交 `af0aa4c` 的同名文件查阅。

本修订在 CPU 准备阶段一次验证并打包全 TRAIN 声学特征（保留原始 padding 和 mask）。两轮、有限与正式训练复用这一份索引；采样仍用实时 EMA。启动性能按正式前三步每步八 rank 最大准备时间计，同时报告一次性准备成本。与旧每卡 8 前六步相同 384 个锚点比较，并单列首步预取。旧版六步八 rank 最慢准备时间之和为 0.834 秒；新版三步在同一口径应不高于此值。

声学塔复用候选使用统一入口 capacity 变体 tower，181 两组各四卡，每卡 16。普通和长音频窗口各 6 步、另两步预热；两窗口总步时均至少缩短 5%、P90 不退化，逐对 logits、梯度和更新后的 FP32 参数完全对齐，才进入八卡接续验收。正式基座与第 128 步恢复文件保留。
