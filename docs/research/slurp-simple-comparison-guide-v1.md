# 简单 Final60 的比较口径与退化基线

2026-11-14更新。本文补齐现有实验的结果解释，不修改训练任务、奖励、采样、选择集或
任何运行配置，也不增加正式Stage2工程发布条件。

## 已完成旧Stage1与概率质量（2026-11-14）

旧版634步Stage1已完整验收，同一9226内部selection上正确1864条，accuracy20.2038%、
macro F1 8.7107%；基座154条/1.6692%/0.0547%。HF与vLLM完整候选logprob最大差
1.90735e-6，argmax变化0。原证据`slurp-stage1-selection-accepted-v1.json`，非官方TEST或RL收益。
只预测24/60类，37类零召回，calendar_set/play_music占8300/9226=89.96%。

复用原始已验收概率，按完整输入/策略/hash绑定离线计算如下；未重跑模型或拟合温度：

| 指标 | 未训练基座 | 旧Stage1 |
| --- | ---: | ---: |
| 未加权gold NLL（nats，越低越好） | 6.58462 | 5.27722 |
| 多类Brier（60类平方差求和，范围0–2） | 1.29889 | 1.33314 |
| Top-label ECE（固定10个等宽bin） | 0.53958 | 0.61946 |
| 错误预测平均置信度 | 55.63% | 80.36% |
| 置信度≥90%且错误的录音数 | 0 | 3176 |

NLL改善与Brier/ECE变差可以同时发生：这些指标衡量分布的不同方面。不能只凭accuracy
或NLL上升宣称概率可供harness可靠使用。ECE依赖分桶，这里是事后描述性10-bin诊断，
不代表预注册校准结果，也不据selection调温度、阈值或训练配置。旧接口是互斥60类
categorical概率，不能将这些数值直接当作新sigmoid适用概率分支的表现。

工具`tools/jev/diagnose_final_selection_probabilities.py`可复用到后续同人口已验收配对产物；
要求完整预测/候选分布和原验收hash，不过滤失败后评分。结果
`docs/engineering/slurp-stage1-probability-quality-v1.json`；独立NumPy公式重算全部指标，
见`.local/simple-final-v1/stage1-probability-quality-independent-check.json`。14项相关CPU测试通过。

新版class-spread全量训练与旧版不同，同时调整了类别时序及批量执行profile，结果必须
另列；即使改善，也不能将全部变化因果归因于某一项修改。终态后按既有流程依次收集
checkpoint、独立state/export审核、完整9226 HF/vLLM评测及finalize，再与基座和旧Stage1
做同人口配对统计，并报告上述概率质量、类别覆盖和两大类预测占比。当前无新版终态
评测或持续RL效果，不把旧结果标作新配置结果。

## 同人口控制

从冻结的40,513条learning计算频数，最多类别为`calendar_set`（2,557条）。
先固定这一预测，再对完整9,226条selection评分；不使用selection或TEST选择类别。
另列固定候选首类及已观察到的基座常量输出，避免把类别偏置视作任务理解。

| 方法 | 固定预测 | 正确/全部 | accuracy % | 固定60类macro F1 % |
| --- | --- | --- | --- | --- |
| learning多数类 | calendar_set | 641/9226 | 6.9478 | 0.21655 |
| 固定候选首类 | alarm_query | 138/9226 | 1.4958 | 0.04912 |
| 已有未训练ASR基座输出 | qa_definition | 154/9226 | 1.6692 | 0.05473 |

前两行是确定性控制，不是模型推理。第三行的原评测计数已核对，全部输出同一类别；
重新构造该常量预测得到的accuracy及macro F1与原分数完全一致。这个结果证明当前
任务接口下未训练基座表现退化，不证明该模型在所有语音任务上都没有理解能力。
typed输出的格式有效率为100%也不足以说明识别有效。

控制使用现有`simple_final_metrics.score`，再以正确数/总数及单类F1解析式独立核对。
结果JSON保留原始比例值，上表转为百分数。新增控制在已有基座结果之后计算，属于
事后描述性诊断；不能声称在基座运行前预注册。没有依此选择训练超参。

## 已闭合基座概率与采样信号诊断

2026-11-13：核对原始9226条完整候选logprob文件hash、policy/request身份、冻结人口与
候选顺序，再计算描述性统计。全部top1相同，但9226行分布各不相同；平均最高类别
概率55.63%，分布对总体平均分布的平均KL为0.02264 nats。这只是输入间变化，不能
单独证明音频语义被理解或使用，也不应把相同argmax说成完全相同的模型分布。

| 诊断 | 结果 |
| --- | --- |
| gold概率均值 / 中位数 | 1.5056% / 0.1251% |
| 平均gold负对数概率 | 6.58462 nats（未加类别权重） |
| top3 / top5 / top10准确率 | 3.3709% / 6.8610% / 12.7574% |
| G16预计混合奖励组比例 | 9.3082% |
| G16预计全错组比例 | 90.6914% |

G16数字由每输入gold概率p按`1-p^16-(1-p)^16`计算，假定温度1下16次独立有放回
采样。以二项分布1至15正确的概率和交叉验证，并用独立Python标量归约复算均值、
熵、topk和混合组比例。这是selection上的解析期望，不是实际learning采样，不与
两步CE起点的GRPO工程更新直接作提升比较。概率归一化只消除原logprob最多2.22e-7
的求和残差；原始top1和指标未改。

该诊断解释未适配起点为何组内RL信号稀疏；保留既定Stage1及辅助CE，但不据selection
调整采样G、温度或loss。topk也不替换已冻结的top1与macro F1。正式RL仍从最终Stage1
开始，其有效组比例与Final曲线需要实际验证。
结果：`../engineering/slurp-base-probability-diagnostic-v1.json`；独立复核收据：
`.local/simple-final-v1/base-probability-independent-check.json`。没有新增模型运行。

## 横向比较边界

| 参照 | 人口与条件 | 当前能否与ASR selection直接计算差值 |
| --- | --- | --- |
| ASR基座→Stage1→RL | 同一9226内部selection、相同Final60口径 | 基座及旧Stage1已完成配对；新版Stage1与RL待完成 |
| omni项目P节点 | FoldA排除后1075条，历史Top1=909/1075 | 不能；需匹配人口及任务供给后复测 |
| 本项目冻结Omni native | 原始官方TEST13078，clean排除子集12991；意图+槽内容联合生成 | 不能；划分、输入模板和输出接口不同 |
| SpeechBrain HuBERT | 官方监督任务权重/论文TEST指标 | 本地权重复测尚未完成；文献不能冒充本地复现 |

本次逐ID检查确认：learning/selection录音ID、源组、音频文件SHA交集均为0；
二者与Omni clean TEST的录音ID交集均为0。后者仅是ID检查，未在这里证明跨官方划分
音频内容或语义来源完全无重叠。Omni历史参考绑定的5份原文件hash均未变化。
不能把不同人口的accuracy、SLU-F1或槽位类型F1相减，称为RL提升。

简单主线完成后，先报告全部预定selection时点的accuracy、macro F1、逐类召回与
失败率，展示末点及相对Stage1起点的变化。超过多数类只是排除一种退化解释，不是
新的学术成功阈值，也不等于超过业内基座或已经证明跨任务泛化。

后续同人口Omni复测和官方学术权重复测保持独立证据轨道，不改当前RL学习集或奖励。
当前只有内部selection发布；官方最终测试的冻结模型与评价协议仍须独立落实。

## 可复核证据

- `docs/engineering/slurp-simple-comparison-population-v1.json`：完整计数、控制分数和输入hash。
- `.local/simple-final-v1/audit_comparison_population.py`：只读数据核对与控制计算脚本；
  实现hash写入上述JSON。
- `docs/research/omni-slurp-jev-reference-lock-v1.json`：历史项目参照及限制。
- `docs/research/speechbrain-history-v1.md`：学术论文、发布权重与历史配方差异。

没有新模型推理，没有用TEST结果调参；候选rc1源码及两份运行配置保持原hash。

## FSC历史产物状态核对（2026-11-13）

两条旧FSC矩阵/外部比较控制器的v1→v7链已按共享盘替代收据及当前原PID缺失核实，
最终因全量SLURP两阶段转向而停止，旧矩阵目录未生成；不能继续描述为等待中，
也不能据此声称跨数据集收益。旧队列不重启。未来迁移须绑定新主线验收模型与当前任务协议。

SpeechBrain官方权重的3793条已闭合预测hash与原收据完全一致，既有正确3774条的
结果与音频对齐报告保持完整；此次只核产物身份，没有读错误样本改模板或重新推理。
详见`../engineering/fsc-history-reconciliation-v1.json`。

## Stage1结果的离线配对分析入口（2026-11-13 19:44 UTC）

`.local/simple-final-v1/compare_selection_pair.py`复用已冻结的输入绑定、Final60评分和
来源组bootstrap。输入为完整selection输入产物、两个同后端原始结果及其独立验收SHA256。
拒绝缺样本、重复ID、输入/音频/策略绑定错误和不同后端；不自动选择checkpoint。

主报告保留accuracy、固定60类macro F1、逐类召回和失败率，并给出正确→错误、
错误→正确及配对accuracy差值。2000次、seed317按source_group有放回重采样，组内全部
录音一起进入，仍以录音数加权。区间仅为描述性95%百分位区间，不含训练seed不确定性，
不校正多checkpoint比较，也不是说话人聚类区间。新统计仅用于结果解释，不改优化目标。

完整9226基座自比较已执行：2146来源组、154正确、9072错误，改善/退化均0、差值及区间
均0，全部原评分精确一致。另以不等大小组、改善/退化并存合成例独立复算非零宽区间；
缺失/重复人口及缺来源拒绝、单组不报告区间。这不是Stage1或RL效果结果。

执行时，`--inputs-sha256`取独立输入验收记录；两个结果hash取各自独立评测验收记录，
不能仅计算待比较文件的hash就当成模型执行证明。分别用`--base`、`--candidate`指定原始
后端JSON，`--output`必须为新路径。当前基座为HF结果，Stage1比较也使用HF结果；
训推HF/vLLM差异仍由原独立finalizer单独报告。

准备收据：`../engineering/slurp-selection-paired-analysis-ready.json`。
此入口在本地分析目录，不进入冻结训练源码；rc2的1360文件和两份配置hash均未改变。
