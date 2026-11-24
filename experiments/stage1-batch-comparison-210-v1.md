# JEV-210 Stage1 微批短测

在 181 上用同两张释放后的 NPU 顺序比较每卡 microbatch 4 和 8。两档梯度累积均为 1，全局 batch 分别为 8 和 16。复用正式 210 的初始化、BF16、LoRA、损失、混淆表、声学缓存、TRAIN 锚点顺序、AdamW 以及学习率规则，分别重建两档的 batch 计划。

各档都执行前向、反向和优化器更新。先用同一批 128 条录音热身，覆盖两档的 8 步学习率 warmup；再跑 320 条普通录音和 32 条长音频压力录音。压力窗口含正式第 51 步的 23.36 秒锚点。总更新数分别为 60 和 30。样本范围在派发前固定并保存顺序哈希。短测仅保留运行证据。

主指标为普通窗口录音数除以逐步最慢 rank 的墙钟耗时之和。另记录热身和长批吞吐、峰值已分配和保留显存及 OOM。计时仅覆盖 Stage1 训练更新。microbatch 8 跑通压力窗口且吞吐改善后，再在六卡上复核容量与首轮耗时。

通过 `run_experiment.py capacity --experiment JEV-210 --local <本机宏>` 派发。[冻结配置](../configs/portable/operations/card-stage1-batch-comparison-210-v1.json) · [主实验报告](../wiki/reports/JEV-210.md)
