# 简单任务固定终点的配对不确定性分析

新增只读入口tools/reports/context_simple_endpoint_uncertainty.py；不改原比较工具或队列。
现有GRPO/RLOO比较给出描述性分差，本入口补充固定step634相对step0的来源组配对区间。
不选择checkpoint、不读取TEST、不自动放行复杂训练。

输入为一个完整简单RL实验目录、算法ID和clean发布目录。先调用原load_run，重新审核
完整634步训练及11点完整selection；随后重验发布标签/候选、两端预测SHA和策略身份、
完整有序9226人口以及实际input/positions一致性，并用原评分器重算端点指标。
缺完整曲线立即拒绝；不把已归档TRAIN128诊断当成selection。

复用已有context_attribute_paired_bootstrap：按来源语句分组、整组配对抽样，保留录音权重，
seed317、PCG64、2000次、百分位95%区间。输出set-F1、exact-set和top1分差区间；
NLL/Brier另列原始点值和无穷标志，不伪造其区间。
只比较预先固定终点，不报告挑选最佳checkpoint后的未校正置信区间。
这里的不确定性以固定模型和来源组抽样假设为条件，不代表训练随机种子方差、说话人独立性
或纯RL因果效果。跨算法/指标多次查看也不能据一个区间自动宣称普遍显著。

本轮15项CPU测试通过/2.04秒，包含既有聚类重采样穷举、配对顺序、策略和输入变更拒绝。
真实当前未完成GRPO调用被拒绝，未读取部分结果冒充成功。完整真实产物成功路径仍待训练
及评测完成；没有新增调度器或设备任务。

完成后的命令为：

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:tests:. python -m tools.reports.context_simple_endpoint_uncertainty \
  --directory .local/decision-objective-research/context-rl-grpo-arithmetic-v2-experiment \
  --algorithm grpo_v1 \
  --release /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v1 \
  --output docs/research/context-simple-grpo-endpoint-uncertainty-v1.json
```

复杂训练仍需要结合完整曲线、Final增益、概率质量退化和本分析作独立书面判断；区间正值
不是全部准入条件。当前尚无正式简单RL完整selection收益，不新增复杂训练。

实际人口补充验证：已认证Stage1终点9226条预测、2146来源组与自身配对，原候选顺序、
标签、预测SHA、input/positions均核对，三个决策分差及区间全为零，五项原评分重算一致。
用时2.55秒。见docs/engineering/context-simple-uncertainty-full-population-control-v1.json。
这是实际输入上的零差控制，未覆盖完整RL成功审核路径、非零增益或选模后区间覆盖率。

迭代预算补充：两次历史Stage1全量selection推理分别2473.70秒、2313.90秒，
按相同路径估计11点纯推理约7.07–7.56小时，另有加载、parity和收集开销。
数据身份已按原proof重验；不是未来RL评测耗时实测或完成承诺。当前固定11点不删减，
后续吞吐候选应同时测评测端收益。见docs/engineering/context-selection-cost-reference-v1.json。
