# 连续意图学习曲线与证据边界

2026-11-12 UTC。新增`intent_learning_curve.py`与`summarize_intent_learning_curve.py`，按固定12步协议汇总连续检查点。输入必须从step0起为连续前缀，所有点使用相同基座、学习名单、标签和准入；每个训练点核验实际训练/验收容器退出0、数据/配置哈希、确切前序策略及固定批次日程。不允许跳过失败或缺失点后拼接曲线。

每点记录模型身份、累计样本使用、累计不同来源数、重复次数分布、全学习集指标、曾更新来源及尚未更新来源。后两种子集随日程变化，不能当作固定开发集或零样本泛化。完整曲线报告末点相对起点、相邻回落次数、按更新步归一化的准确率曲线面积，不挑最优点。空子集指标为null，不填0伪造退化。

6项测试通过，包括累计覆盖、空子集、缺失检查点、标签/人群变化、日程越界和基座变化。已实际重读step0/1独立验收概率生成首个真实前缀：`intent-finite-curve-step01-v1`。D/Q准确率变化均为0，目前仅首步被纳入已验收前缀，第二步训练完成不提前纳入。证据见`intent-finite-curve-step01-v1.json`。已冻结运行快照不因新增汇总工具而改动。

同时检查开发集是否可复用omni既有听音证据。公开clean-v7 README、annotation-audit及Confirmed-Defect-Text-Audit明确没有音频听审，不能以清理发布名义替代音频语义裁决。固定300条selection来源逐一绑定原TRAIN recording元数据，300条status均为correct，191条WER=0、109条非零，无缺失WER；尚未核实该WER的生产方法，因此不据其筛选、删除或自动准入。此处未打开新的TEST样本；读取的omni汇总文档含历史测试清洗统计，非本次测试反馈。

原始逐条证据保存`intent-development-review-v1/publisher-recording-evidence-v1.json`，仓库凭证为`intent-development-publisher-evidence-v1.json`。独立开发评测的音频/本体审查仍未完成。学习曲线、研究基线复测和FSC跨域结果是三类不同证据，前者不能替代后两者。
