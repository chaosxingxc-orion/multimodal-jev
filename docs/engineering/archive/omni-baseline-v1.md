# Omni / 冻结基线归档（2026-11-10 至 2026-11-12）

本文件收编 Omni、冻结 Thinker、FSC-Jev 冻结读出和 K/V 压缩探针。原始 md 已移入
`archive/raw/omni-baseline/`；模型、输入、预测和验收 JSON 收据仍在原路径保留。

## 做了什么

- 在 Qwen3-Omni-30B-A3B Thinker 上做 CPU/NPU 加载、FSDP 布局、概率审计、采样与
  `compute_log_prob` 对齐。
- 验证不可达 K/V 移除候选，尝试降低 Omni actor 的显存/计算。
- 对 Qwen3-ASR 与 FSC-Jev 冻结接口做六视图读出诊断。
- 复核 Omni 历史 P 基线与当前 ASR-Jev 的可比条件。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| Qwen3-ASR 原生 FSC | 3,793 条直接 JSON 结果 0 正确、0 有效 JSON；1.7B TRAIN64 同样 0 有效 JSON | 说明该 checkpoint/提示/接口失败，不说明 ASR 能力为零 |
| SpeechBrain FSC | 官方权重 TEST 3,793：joint 3,774/3,793 = 99.4991%；CPU 推理 13,721.23 秒 | 任务监督模型，不是零样本；与作者 99.60% 差 -0.1009 点 |
| FSC-Jev 冻结读出 | 384 条真实音频：native 1/64、permutation 2/64、rebind 1/64、missing 0/64；全零对照 native 2/64 | 有输出变化但无有效语义判别证据；固定前缀不是教师收益 |
| Omni 历史 P 基线 | 909/1075 = 84.56%，可作历史项目能力参照 | 仅 3 条音频与当前 89 条交集；候选集与知识条件不同，不能与当前 1/89 相减 |
| K/V 压缩候选 | CPU 测试与候选门禁通过；完整 actor 对照未完成 | 未派发 NPU/训练，不作为当前发布依赖 |

**结论：Omni 是历史冻结基线，保留其工程与数值诊断；FSC-Jev 冻结读出没有证明语义判别能力。
当前主线已转为 Qwen3-ASR + 共享概率头的 `context-prob-v1`。**

## 失败/过期路径

- 直接把 Omni 历史 84.56% 与当前小样本分数比较：人口、候选和知识供给都不同。
- 把固定 `language English<asr_text>` 前缀读出算作任务能力或教师收益：只是输出格式约束。
- 继续派发 Omni FSDP / K/V 全量对照：当前阶段不需要，且未完成验收。

## 保留证据

- `omni-*-verified.json`、`omni-probability-audit-v1.json`、`frozen-*-progress-*.md` 等原文件。
- 当前仍活跃的跨数据集音频核查：`native-cross-dataset-audio-v2.md`、`external-baseline-compatibility.md`。
