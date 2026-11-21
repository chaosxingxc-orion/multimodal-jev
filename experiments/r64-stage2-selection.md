# R64 Stage2 SLURP selection 补评

复用同轮 68 步 Stage2 正式训练的审核收据。源域 selection 固定为 9,226 条录音、60 个原生意图。保留已审的 step 0，对 step 68 补齐全量推理。保留此前的有限 HF 参考，复用目标机已验证的同人口缓存，并通过本次运行时对照后执行全量。评测使用源域 selection 的固定用途，报告 Top-1 macro-F1、准确率和平均原始 Top-1 概率。
