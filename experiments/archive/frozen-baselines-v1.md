# 冻结基线实验归档（2026-11-09 至 2026-11-12）

本文件收编冻结 Omni / Qwen3-ASR / FSC / SLURP 直接音频基线、远程教师提示优化和早期迁移试跑。
原始 md 已移入 `archive/raw/frozen-baselines/`；模型、输入、预测和验收 JSON 收据仍在
`docs/engineering/` 与共享实验目录原路径保留。

## 做了什么

- Qwen3-ASR-0.6B/1.7B 直接音频 JSON 与有限标签读出。
- Qwen3-Omni-30B 冻结权重在 FSC 六视图、SLURP 原生 TEST 上的完整推理。
- FSC 冻结提示的远程教师 TFRL 候选与 p1 完整 TEST 对照。
- FSC-Jev / SLURP-Jev 冻结读出、早期 SFT 与迁移试跑。
- SpeechBrain FSC 官方权重复测与 AudioJev/MMAR 资源核验。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| Qwen3-ASR 原生 FSC | 3,793 条直接 JSON 0 正确、0 有效；1.7B TRAIN64 同样 0 有效 | 说明该 checkpoint/提示/接口失败，不说明 ASR 能力为零 |
| Qwen3-ASR 直接 SLURP | 固定 60 条探针 0 条严格任务正确 | 历史接口诊断，已被 `context-prob-v1` 概率头路线取代 |
| SpeechBrain FSC | 官方权重 TEST 3,774/3,793 = 99.4991% | 任务监督模型，不是零样本；与作者 99.60% 差 -0.1009 点 |
| FSC Omni p1 提示优化 | native 46.1904% → 91.7743%；六视图全对仅 0.6327%；正确候选缺失成功率 <1% | 冻结权重提示收益；不等于权重训练、完整 Jev 或跨任务泛化 |
| SLURP Omni 原生 TEST | 联合意图 5,219/13,078 = 39.9067%，官方 SLU-F1 31.4127% | 原始未清洗标签诊断；不得用于训练、奖励、教师反馈或选模 |
| FSC-Jev 冻结读出 | native 1/64、permutation 2/64、rebind 1/64、missing 0/64；全零对照 native 2/64 | 有输出变化但无有效语义判别证据 |
| SLURP SFT 迁移 | 格式 168/168，但非空目标正确 0/81，低于全 null 控制 87/168 | 只有格式学习，没有语义迁移；不是当前端到端基线 |
| AudioJev/MMAR | 资源与协议已登记，无可用 CUDA 运行证据 | 未复现，不能引用论文数字当本地结果 |

**结论：冻结基线提供了接口失败、学术参考和提示优化三类证据；它们不是当前 Qwen3-ASR
概率头主线的训练收益，也不能与 `context-prob-v1` 的 Stage1/Stage2 成绩直接相减。**

## 失败/过期路径

- 用 ASR 转写再调用文本模型替代在线音频 actor。
- 把 Omni 历史 84.56% 与当前小样本分数直接比较：人口、候选和知识供给都不同。
- 把固定 `language English<asr_text>` 前缀读出或提示格式改善算作任务能力。
- 用原始 SLURP 标签诊断值做训练、奖励、教师提示优化或模型选择。
- 把 FSC p1 提示收益表述为权重训练收益或跨任务泛化。

## 保留证据

- `docs/engineering/` 下的 `frozen-*-verified.json`、`omni-*-verified.json`、
  `fsc-*-result.json` 等原收据。
- 工程侧总结见 [Omni / 冻结基线归档](../../docs/engineering/archive/omni-baseline-v1.md)。
- 当前学术横比入口仍为 [学术基线比较 v1](../academic-baseline-comparison-v1.md)。
