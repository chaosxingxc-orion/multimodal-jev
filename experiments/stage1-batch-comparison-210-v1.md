# JEV-210 Stage1 微批短测

在 181 上用同两张释放后的 NPU 顺序比较每卡 microbatch 4 和 8。两档梯度累积均为 1，全局 batch 分别为 8 和 16。复用正式 210 的初始化、BF16、LoRA、损失、混淆表、声学缓存、TRAIN 锚点顺序、AdamW 以及学习率规则，分别重建两档的 batch 计划。

各档都执行前向、反向和优化器更新。先用同一批 128 条录音热身，覆盖两档的 8 步学习率 warmup；再跑 320 条普通录音和 32 条长音频压力录音。压力窗口含正式第 51 步的 23.36 秒锚点。总更新数分别为 60 和 30。样本范围在派发前固定并保存顺序哈希。短测仅保留运行证据。

主指标为普通窗口录音数除以逐步最慢 rank 的墙钟耗时之和。另记录热身和长批吞吐、峰值已分配和保留显存及 OOM。计时仅覆盖 Stage1 训练更新。microbatch 8 跑通压力窗口且吞吐改善后，再在六卡上复核容量与首轮耗时。

通过 `run_experiment.py capacity --experiment JEV-210 --local <本机宏>` 派发。[冻结配置](../configs/portable/operations/card-stage1-batch-comparison-210-v1.json) · [主实验报告](../wiki/reports/JEV-210.md)

激活重计算另用 `--capacity-variant recompute` 接续。该配置只重跑 microbatch 8。封包逐项比对基础包的冻结训练配置，仅增加 BF16 语言层的非重入式激活重计算。使用同一两张卡、同一租约、相同三个窗口，并等待原对照完成。共同报告峰值显存、普通窗口的录音吞吐及长批 OOM。若原版 microbatch 8 未跑完，按长批容量与 microbatch 4 的吞吐判断是否值得做六卡复核。[重计算配置](../configs/portable/operations/card-stage1-recompute-210-v1.json)

## 五卡效率对照

181 释放 2 号卡后，用 3–7 号卡运行 `dp5`。microbatch 4 和 8 对应全局 batch 20 和 40，各用一次微批完成更新。两档复用同一审核通过的 TRAIN 数据、混淆表、声学缓存、初始化和目标。先跑相同的 320 条热身录音，再测 160 条普通录音和 80 条长音频压力录音。压力窗口保留此前观察到的 23.36 秒录音。两档分别执行 28 和 14 步。

普通窗口以锚点每秒为主指标。长音频压力窗口要求完成全部更新，另记录显存峰值和分配失败。按每个更新中最慢 rank 的耗时统计。对两档相同曝光复算，记录实际使用的卡数和批量。五卡测试沿用已发布的 CPU 准备证据，按现场设备检查重新派发。配置从本机宏解析设备和路径。

运行入口为 `run_experiment.py capacity --experiment JEV-210 --capacity-variant dp5 --local <本机宏>`。[配置](../configs/portable/operations/card-stage1-batch-comparison-210-dp5-v1.json)
