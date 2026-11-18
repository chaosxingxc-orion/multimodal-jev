# 旧单选分类分支：全量选择集终态

旧分支634次监督更新后的全9226条内部选择集评测已完成。HF/vLLM各正确7599条，
accuracy82.3651%，固定60类macro-F1 79.5706%，支持类macro recall81.0041%，
invalid/failed均0；两后端argmax零分歧，报告最大logprob差1.90735e-6。

原评测cd263613与CPU归档2e10e82b均exit0/nonOOM，既有完整输入/权重/后端审核
证据保留。另从绑定的原始9226条预测重新构造混淆计数、复算accuracy与macro指标，
与归档结果一致（浮点误差<1e-12）。证据`slurp-categorical-full-selection-final-v1.json`。

这是clean SLURP TRAIN内部selection、旧单Final60分类/CE任务的终点成绩，
不是官方TEST、无训练基座、新context概率头或RL收益。与Omni既有不同输入协议/
人口的分数不能直接排名；没有相同起点完整曲线，不能归因算法提升。
本结果作为历史工程与任务适配参照保留，不改变当前概率决策/RL主线。
