# 已验收采样的奖励信号分析

`analyze_sfc_sampling_rewards.py` 用于冻结 TRAIN78 K2 采样完成后的离线分析。
真实采样仍排队，尚无该分析的真实模型数值；不得用 greedy 或人工 false+EOS
响应填充 K 组，也不重放缺失请求拼成完整结果。

先核对 sampling root、独立验收哈希、完整 156 条响应、容器/源码身份和 tokenizer，
再从 intent/raw/result 独立重建采样记录，与验收凭证逐字段比较。通过后才打开
固定 TRAIN target 文件，选择六条 learning 回合。完整 TRAIN 文件被解析，
selection 行不进入奖励，不打开 TEST 目标。compose 再次验证完整学习会话、
原始分区及同 prompt 身份。调用方仍须确认原容器和验收进程的实际终态。

分别计算不变的 independent 和 function-conjunction（bonus=0.2），报告：

- 有奖励差异的组、等分全对组、等分全错组及等分混合正确性组。
- 相同文本/相同 token 的组数，区分答案多样性和奖励差异。
- present/absent/calls 各自的正确、格式/schema 错误、截断、token 数和奖励总量。
- 完整 bundle/replicate 的原生调用精确匹配与全部决策正确数。

等分全错说明当前 outcome GRPO 没有组内对比信号，等分全对可能已解决该输入；
两者不能合并为同一种问题。关系奖励能增加正确 calls 端点的奖励差异，不代表
calls 语义正确率提高。K2 零差异也不证明整个策略分布退化。该分析不计算模型
梯度，不调整奖励、不自动选择新算法。

33 项本地相关测试通过。区分性夹具只翻转一个 absent 输出：独立奖励有一个
对比组，关系奖励有三个，但原生调用正确数不变；全错同分仍为零对比组。
这些是人工夹具证据，真实报告须等待原采样及独立验收完成。
