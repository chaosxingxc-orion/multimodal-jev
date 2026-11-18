# clean SLURP Omni参考：同人口身份与来源聚类区间

本次复用历史Qwen3-Omni原生复合JSON输出，没有新推理或权重更新。
沿当前已冻结TEST scoring contract重新核验12项评分资产和归档结果的16项依赖，
由原生targets和clean metadata重建gold、87条排除名单以及完整12991条输入顺序，
验证source_group与音频身份字段对齐。没有重新读取全部音频波形，实际推理身份继续依赖
已归档的原生验收收据；本次不把存档核验称为新的模型执行。

| 项目 | 结果 |
| --- | ---: |
| clean TEST录音 | 12,991 |
| 来源语句组 | 2,947 |
| 原生joint intent正确 | 5,207 |
| 全分母accuracy | 40.0816% |
| 来源语句聚类bootstrap 95%区间 | [38.4782%, 41.6860%] |
| 原生格式失败，保留计错 | 5,958 |

bootstrap使用PCG64、seed317、2000次，整组有放回抽样、保留组内全部录音，
每次按实际抽到的录音总数计算准确率，百分位采用linear插值。来源组大小为1–56，
不能把组均值直接平均来替代录音加权指标，也不能按12991个独立录音估计区间。
所有组的原生gold一致性已检查。

这是在历史参考分数已知后新增的描述性区间，不是事前注册的主假设检验。
来源语句聚类不能覆盖不同语句共享说话人的相关性、训练种子波动、多重比较或选模偏差。
未来Jev与Omni的差值需要同录音配对计算；不能用两个边际区间是否重叠判断差值显著性。

当前TEST队列已冻结同一clean人口的初始/来源选择策略及Omni比较，保持原评分契约不变。
本报告不改变阈值、模板、选模或排队包，不提供Jev TEST成绩。Omni原生生成接口包含
格式失败且训练条件不同，该参考只用于能力定位，不能称同预算RL优越性或业内SOTA复现。
当前Jev独立概率意图任务也不能沿用Omni的开放槽值SLU-F1。

证据：[资产身份和统计](slurp-omni-clean-reference-cluster-v1.json)。
工具：`tools/reports/audit_slurp_omni_cluster_reference.py`。
6项本地测试通过，含不等大小来源组、记录重排一致、重复/缺组/非布尔/单组拒绝。
已有FSC SpeechBrain复测、待补SLURP HuBERT资产和AudioJev CUDA资源仍分别记录在
[学术比较总表](../../experiments/academic-baseline-comparison-v1.md)，本报告不替代其复现工作。
