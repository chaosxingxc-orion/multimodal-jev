# 固定评测人口、定义诊断与学术横比

2026-11-12。任务设计阶段，未调用模型。当前协议集中于
[evaluation-protocol-candidate-v1.json](../../configs/tasks/slurp/evaluation-protocol-candidate-v1.json)，
任务v2已引用。沿用已有论文/官方源码核查记录，本轮不冒称重新检索或完成学术复现。

## 固定选择集与实际请求量

完整official-clean-intent-v1选择集9,226录音/2,146来源组，是official TRAIN内部
来源隔离划分，不是官方DEV/TEST。重新核验manifest及所有输入hash，确认与learning
在录音ID、source_group、音频SHA上无交集；不声称说话人隔离或语义近重复完全隔离。
评测清单只包含录音身份与面板/变换坐标，不含reference文本或gold选题逻辑。

固定15面板、三臂Direct/q_small/q_full、三种布局。q_small在14面板有2Q、最后
面板只有1I；q_full为13个8Q、1个7Q、1个4Q。最后没有S的面板不补造问题或转移
奖励预算，结果不能统一标作8Q或2Q。各臂对同面板供给相同定义知识；Direct的面板
槽位知识也会不同，所以不能在没有输入身份等价证明时只跑一次并复制15份成绩。

每检查点canonical完整评测415,170请求；三个布局全量诊断1,245,510请求。来源与
独立音频数不随面板增加。三个布局全量诊断只在Stage1与预定terminal checkpoint
登记，canonical曲线频率待训练总预算和吞吐验收后确定具体update IDs。此次是成本
透明的设计，不代表已有资源承诺或允许运行。不同模型若可安全复用音频编码，需单独
实测，不把请求数直接当音频编码次数或设备小时。

共享目录`slurp-jev-benchmark/selection-evaluation-design-v1`含roster、requests、
design。完整坐标和身份独立复核见
[selection-evaluation-design-check-v1.json](selection-evaluation-design-check-v1.json)。
这不是完整processor验收，C00等符号是逻辑身份；绑定到现有候选token的渲染映射还须
通过完整输入及概率一致性检查，不能直接以字符数推测token长度。

## 指标与失败处理

原生Final主指标为60个canonical scenario/action组合上的自然录音准确率，所有
面板分别报告并给等权均值。另报固定60类macro-F1、macro recall、每类支持数、
learning频数小于100的尾类和来源组等权结果。非法输出计错，不删除困难录音；缺失
预测保留全分母，但未完成推理只能称partial，不能冒称完整运行验收通过。

语义Q只用独立准入的positive/negative，缺标/冲突mask，逐定义报告覆盖、正例F1、
正例召回及误报。某极性无支持时相应指标为null并给分母。宏正例F1聚合范围和定义数
公开，不以补零或静默忽略制造可比性。非法二分类输出计为错误，正类计FN；负类非法
单列，不能仅凭低FPR掩盖无效输出，balanced accuracy把负类非法也计错。
槽位集合EM只在面板真值完整时计算，部分标签只能报masked决策指标。面板集合不等于
全55类型一次输出，更不等于含槽值的官方SLU-F1。

原生标签投影和语义定义真值分表。cooking完整审核已证明两者会不同；原生指标能用
不自动授权新Q监督。相同mask/人口用于各方法，不依赖其预测是否成功。

## 定义诊断与统计

候选重排只改变位置、保留符号语义映射；符号重绑定只改变符号映射、保留卡片位置与
含义。评分前还原语义ID，同时报告映射一致率、两次都对、准确率差及无效率。常量
错误也能高一致，因此一致率不能代替任务正确性。这两个测试仅支持等变性结论。
同义改写需独立审核等义卡；改变语义定义需重新确定真值，不能沿用旧gold。两项仍
待完成，明确保留为冻结缺口，不能以符号实验代替定义理解。

配对bootstrap按source_group重采样，组内所有录音/面板/布局一起保留；重复抽到组
按倍数计入，重算录音加权比例，10,000次、seed317、95% percentile区间。训练种子
登记17/29/43，各自曲线、均值和sample SD分别报告；来源bootstrap不覆盖训练种子
波动。所有预定检查点报pointwise区间，不挑最优检查点声称统计显著。
选择集只提供开发证据；正式泛化结论需要冻结后独立TEST协议。

## 横向比较分表

| 结果条件 | 能回答的问题 | 不能替代 |
|---|---|---|
| 原始官方13,078 real TEST + 官方scorer | 公开学术原生任务定位 | clean或新语义任务表现 |
| 同一版本clean TEST + 相同清洗及mask | 相同数据条件下模型比较 | 直接和原始论文分数相减称收益 |
| 本项目内部selection + 固定菜单 | 训练开发曲线及机制诊断 | 独立官方TEST泛化结论 |
| 新语义定义与独立监督 | 定义消费、类型判决 | 原生60类或含槽值SLU-F1 |

SpeechBrain HuBERT直接音频满足端到端约束；作者75.10 SLU-F1、87.54% intent在
real+synth训练条件下，只列文献参考。现有仓库有接口/解析核查，本轮未加载权重或
复测；正式复现仍缺完整资产与运行验收。不能把解析往返测试当模型成绩。

Omni历史P 84.56%来自1075条筛选人口；S类型77.22%来自545条、gold intent条件，
且混有learning与heldout。S1虽不提供intent，也使用P成功筛选的TRAIN人口。它们
不作本项目的达标线。matched Omni需同音频、本体、定义知识和mask，不给gold intent；
原生生成式解析与Jev typed读出分别记录，模型规模/训练暴露差异公开。

FSC已有官方SpeechBrain公开权重3774/3793复测证据，但它用目标任务监督训练，与
SLURP训练后冻结权重迁移的比较是能力定位而非同预算RL算法胜负。FSC新迁移协议v2
保持独立；本轮不重启旧矩阵或沿用其checkpoint选择规则。

下一步仍是完成语义准入、Stage1目标及定义理解干预，之后才冻结工程。官方/clean
TEST清单与scorer身份也需补齐；本轮不新下载模型，不派发推理或训练。
