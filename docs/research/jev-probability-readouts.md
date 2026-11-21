# Jev 可以输出哪一种概率？

本项目的 Jev 直接读取音频与候选规约，逐候选给出判决。先比较 2026 年三篇决策模型原文的输出契约，再讨论本地训练集如何标定分数的位置。

| 2026 年模型 | 模型输出的量 | 对 harness 的含义 |
|---|---|---|
| [Qwen-Audio-Jev，§2.3](https://arxiv.org/html/2610.21614v2) | Choice 只在有效候选上做 softmax。Noul 用一个标量表示给定命题为真的概率。Score 输出有序评分档位的分布，并按档位值求期望。 | 我们对每个意图单独问「这条语音是否表达它」，最接近 Noul。Noul 的执行阈值由应用确定。Score 的期望描述评分水平。 |
| [AudioJev](https://arxiv.org/html/2610.01293v1) | 从一次前向的下一 token logits 读取当前候选位置标签，在这些标签上归一化，得到给定候选集的选择分布。 | 调换候选顺序后，把概率映射回同一语义候选以检查一致性。顺序稳定性和分数对应的实际正确率需要分别验证。 |
| [Intern-Decision](https://arxiv.org/html/2610.22131) | 文本/图像模型在每个决策字段的合法选项内做温度 softmax，一个前向可以返回多个字段的分布。 | 在单独的校准集拟合温度。在论文的 96 例已知分布试验中，作者报告多类 Brier 从 0.628 降到 0.550。正温度保持同一字段内的排序。 |

**源域训练样本上的中位数。** 2026-11-21 对完整 clean SLURP TRAIN 的 Stage1 缓存逐录音统计，每条录音等权，共 40,513 条。设同一条录音的候选成员分从大到小为 p₁、p₂、…。三个数代表不同问题。

| 每条录音先计算 | 跨录音取 P50 | 读数含义 |
|---|---:|---|
| p₁ | 0.893309 | 典型录音的最高成员分 |
| p₁−p₂ | 0.750975 | 典型录音第一名领先第二名的幅度 |
| 所有候选成员分的中位数 | 0.000287 | 典型录音中处在候选中间位置的分数 |

数据身份为策略 `0d62139bbc2dc5ee064b9f7e18af944bb31aba105844bd4f247a731a024ff03b`，reference SHA256 为 `ddd6325117857b1598651bc59c5a802fd828293a36000c6fbbf183b6a7c89130`。可用 `PYTHONPATH=src:. python3 -m tools.reports.train_score_medians configs/portable/operations/jev-train-confusions.json .local/config/stage1-train-score-medians.json` 重算。本地宏绑定该策略的发布集与 reference。这里使用参与模型拟合的 TRAIN，仅描述这个 checkpoint 的分数分布。

每个请求可返回候选原始分数、赢家与 p₁−p₂。P50 属于策略元数据，按 checkpoint、模板、候选定义和源域校准人口固定。若需要分数的相对位置，可用同一人口的经验分布计算百分位。若需要「本次赢家判对的概率」，须用真实标签，以独立留出的源 TRAIN 录音组或折外预测拟合。执行阈值按误接成本或可容忍的错误率选择，并报告覆盖率。Stage2 的难例过采样改变了样本权重，直接求 P50 描述的是难例视图。对 FSC/MINDS 做迁移评测时冻结源域规则。

**Qwen-Audio-Jev 的数据口径。** [原文 §3.1.1、§3.2.1、§4.1–4.2](https://arxiv.org/html/2610.21614v2) 区分训练标注和评测配置。训练例是音频、问题、答案规约和已观测目标；只监督该问题激活的类型，缺标不产生监督。文中以 MELD 和 VoxCeleb 说明异构标注，未给出足以复原训练混合的逐库划分、条数及比例。下表只登记评测口径：

| 类型 | 评测配置 |
|---|---|
| Choice（8） | CochlScene、NSynth、NonSpeech7k、DiDiSpeech、MELD、MMAU、MMSU、MMAR。 |
| Noul（4） | VoxCeleb1-O 同说话人验证、Clotho-AQA 二元问答子任务、PodSarc 讽刺检测、In-the-Wild 语音真伪。 |
| Score（6） | NISQA 的总体质量、噪声、音色失真（coloration）、不连续性、响度五项；VoiceMOS/BVCC 合成语音总体质量一项。对照人工平均评分。 |

论文可选 RLCD 阶段的采样是同题一次前向后，从探索分布有放回抽取 32 个动作；它不等于抽取训练难例或改变候选集合。项目 Stage1 的全候选 InfoNCE 已学习集合内竞争，v0.0.195 TPO 则将排序等信息写入每候选是/否的奖励差。建议先用现有 logits 在完整原生意图集上比较 Choice 类别概率与 Noul 成员分；纯读出变换不会改变同一组 logits 的 top-1。只有具备独立有序评分标注时才增加 Score 任务，不把 TRAIN 中位数转成评分监督。

以下为可对照的其他读出方式及相邻文献。

| 概率或证据 | 论文依据 | 适用条件与边界 |
|---|---|---|
| 单个命题的 Bernoulli 概率，采用 probit 链接 Φ(f) | Rasmussen & Williams, [Gaussian Processes for Machine Learning，第 3 章](https://www.gaussianprocess.org/gpml/chapters/RW3.pdf) | Φ 是标准正态 CDF，同 sigmoid 一样严格单调。若潜变量具有已建模的高斯后验 N(μ,v)，预测均值为 Φ(μ/√(1+v))；只有换链接没有引入 v 时，没有多出不确定性信息。 |
| 封闭候选集合上的类别分布，softmax(相似度/T) 或 softmax(−距离/T) | [CLAP](https://arxiv.org/abs/2206.04769) 提供音频文本对比匹配；[Prototypical Networks](https://arxiv.org/abs/1703.05175) 明确采用对负距离的 softmax | CLAP 的对比归一化服务于匹配；部署时归一化得到的是给定候选集下的相对选择分布，不能把训练 batch 匹配概率解释为部署后验。 |
| A 相对 B 的胜率 | [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) 的轨迹回报差归一化 | 二选一 softmax 数学上仍是 sigmoid(sA−sB)，共同分数偏移不可识别。训练排序有用，不能独立决定执行。 |
| Yes/No 词表读出 | [monoT5](https://arxiv.org/abs/2003.06713)、[Generative Verifiers](https://arxiv.org/abs/2408.15240) | 区分整词表的 P(Yes) 与只在 Yes/No 中重归一化；后者等于 sigmoid(zYes−zNo)。复用词表 head 是参数化选择，不要求 Jev 生成文本。 |
| 概率分布的分布：Dirichlet，二元时为 Beta | [Sensoy et al., Evidential Deep Learning](https://arxiv.org/abs/1806.01768)；[Bengs et al., 局限分析](https://arxiv.org/abs/2203.06102) | 正参数 α、β 给出均值 α/(α+β)、浓度 α+β。只对预测均值计算现有 Bernoulli KL，浓度不可辨识；需要额外学习假设和独立验证。不能将浓度直接当作未知事件的真值。 |
| 输出似然比证据，按应用先验转换后验 | 语音验证 [BOSARIS](https://arxiv.org/abs/1304.2865) | 对同一判别任务，其已校准对数似然比 ℓ 与部署先验 π 组合：posterior odds=exp(ℓ)·π/(1−π)。误接和漏接成本决定执行阈值。原始网络 logit 通常不是校准似然比，跨域似然变化也不是只改先验能解决的。 |
| 冻结分数的非参数概率校准 | [Venn–Abers](https://arxiv.org/abs/1211.0025)；温度缩放对照见 [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) | Venn–Abers 输出一对概率，不是对任意一次查询的置信区间。校准假设不保证跨域成立。可预测“赢家是否正确”作为单独事件，无需改动候选读出。 |

本项目 v0.0.195 的 TPO 目标 q 是冻结的 Bernoulli 参数。换成 probit 时保留 q，用 KL(Bern(q)||Bern(Φ(z)))；若仍用原来的 sigmoid KL 训练后直接替换推理激活，训推协议会改变。改为完整候选 softmax 则改变事件空间，需独立构建归一化类别目标。对于冻结分数，逐项施加相同的严格单调变换不改变候选排序；跨样本的概率间隔和阈值表现可以改变。

第一优先的对照：在源 TRAIN 内按录音/语句组划出独立校准部分，避免拿训练内拟合分数冒充未见校准样本；用一个另行定义的顶选正确性校准器读取 p1、p1−p2 和候选中位数，输出 p_top_correct。对比仅用 p1 及 p1+margin 的对照，在固定覆盖率比较误接率和风险覆盖曲线，另报顶选正确性的 NLL/Brier。独立候选自身的 NLL/Brier 和 macro-F1 分开报告。规则在源域冻结，再对新域做盲测，不以 FSC/MINDS 测试数据选择变换或阈值。未验证到目标域时不承诺跨域后验或自动执行。
