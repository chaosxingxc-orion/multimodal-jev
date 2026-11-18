# 已有Omni原生预测的clean-v7基线复算

2026-11-12。本轮完成原生基线的一项实际交付：复用已验收的完整TEST预测，在固定
clean-v7排除规则下重算。没有重新推理、调整提示或解析器，没有新模型下载或训练。
该工作不依赖新增语义Q标注，但结果也不能作为新语义Q的准确率。

## 冻结条件与结果

历史模型为本地Qwen3-Omni-30B-A3B-Instruct，直接音频，BF16、seed17、greedy，
完整原生taxonomy输入，输出scenario/action/entities JSON；4096上下文、256输出预算。
采用既有严格parser，不修复Markdown等格式失败。clean-v7固定删除27来源/87录音，
其余原生标签逐条核对不变，全部预测与旧预测对应行完全相同。

| 人口 | 录音 | joint intent正确 | 准确率 | 官方SLU-F1 | 格式失败 |
|---|---:|---:|---:|---:|---:|
| 原始官方TEST | 13,078 | 5,219 | 39.91% | 31.41% | 5,993 |
| clean-v7 TEST | 12,991 | 5,207 | 40.08% | 31.43% | 5,958 |

clean人口有2947来源。失败记录保留在完整分母，格式有效7033条；没有把仅成功解析
样本的条件准确率当作总体成绩。两行差异来自固定数据排除，**不构成算法提升**。
该历史完整JSON输出协议与omni项目的选择人群/提供意图等协议不同，不能与其84.56%
意图或77.22%槽位类型F1直接排名。类型F1与包含槽值的SLU-F1也不是同一指标。

## 验收与限制

工具`rescore_slurp_clean_native.py`先核对历史acceptance及prediction/identity哈希、
成功终态、完整13,078预测覆盖，随后证明clean元数据恰好对应固定排除后的集合，
保持原生target与请求绑定，最后调用固定官方metrics及固定依赖环境。
全量数据、排除表、scorer和实现hash记录于共享
`experiments/multimodal-jev/frozen-qwen3-omni-slurp-clean-test-rescore-v1/result.json`。

[独立复核](../engineering/omni-slurp-clean-native-rescore-v1-verified.json)从clean元数据
另行重建gold，以旧规范化预测过滤后逐行比对新预测，重新调用官方metrics核对所有
指标项。最大浮点差1.82e-12，在既有relative/absolute 1e-12容差内；整数意图计数
精确相同。仍使用同一官方指标实现，不宣称独立算法评分器或新音频校验。
16项人口与原生评分测试通过，包括错来源、标签漂移、重复/未知排除和格式失败。

结果加入academic-baseline-lock.json的local_observations，文献记录不变。
旧协议末尾“尚未验收”是历史状态；其后验收见既有native-test-v1-verified.json，
本轮不覆写历史验收。此次clean成绩仅作固定外部定位，不用于训练、教师反馈、
提示优化或checkpoint选择。

核心学术复现仍有缺口：FSC SpeechBrain完整权重复测已完成3774/3793；SLURP
SpeechBrain HuBERT缺核心资产，当前不新增模型下载；本结果不能代替其任务训练权重
复现。新Jev任务同人口、同定义的Omni对照也尚未完成，不能用历史原生结果顶替。

## 2026-11-13 固定预测的不确定性补充

复用同一12,991条预测，按来源和说话人分别作2,000次整组有放回bootstrap，
seed317、线性分位数95%区间；统计量始终按录音加权，失败保留分母。

| 指标 | 点估计 | 按2,947来源聚类区间 | 按142说话人聚类区间 |
|---|---:|---:|---:|
| 原生联合意图准确率 | 40.08% | 38.48%–41.69% | 37.95%–42.22% |
| 格式失败比例 | 45.86% | 44.56%–47.22% | 43.19%–48.67% |

有样本的59个原生类别宏平均召回为44.45%；当前人口没有`cooking_query`，
分母不是完整60类或65个语义谓词，不为无样本类填0。
这是结果产生后的描述性统计，不是训练种子不确定性、来源与说话人联合相关模型、
显著性检验或新的SLU-F1区间，不能用于TEST选模板、规则或checkpoint。

报告位于共享`experiments/multimodal-jev/omni-clean-native-uncertainty-v1/report.json`，
SHA256为`4d70ab7f1a2dba686769c24adac8ffd55efbf3dc21b4b94a99dcc1e114492285`。
[独立复核](omni-clean-native-uncertainty-check-v1.json)逐录音重建结果，并用标量
整组重采样重新计算全部四个区间，最大差0；3项测试覆盖权重、顺序和无效输入。
此次没有新推理、标签修改或训练，也不改变原有基线的可比范围。
