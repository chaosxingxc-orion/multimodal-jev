# JEV-210　1.7B r256 知识卡与融合 TPO

沿用 JEV-209 的混合数据和两阶段配方，检验 Qwen3-ASR-1.7B 与 q/v LoRA r256 的组合是否改善源域与跨数据集表现。两个阶段均使用 r256、alpha512，保持 alpha/r = 2。相对 209 同时扩大基座与适配器，结果归因于这组改动。基座使用官方 [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3)，隐藏维度 2048，28 层共 56 个 q/v 投影。

| 阶段 | 配方与预算 |
| --- | --- |
| 数据与输入 | 109,610 条 clean TRAIN，真人 40,513 条、合成 69,097 条，68 类。复用 JEV-209 的冻结人口、种子 317、知识卡与随机示例分配。核对 processor、音频 token、卡片编译结果后复用输入缓存。 |
| Stage1 | 181 八卡。每条模型输入包含一张卡和一段音频，三元组 InfoNCE 加概率间隔与双侧约束。每卡每步 96 个锚点，全局 768，每轮 143 步，两轮 286 步。先尝试 microbatch 24，通过四次推理和两次更新锁定宽度。 |
| 第一轮交接 | 用 210 初始权重建立负例采样的混淆表。第 143 步落盘后，使用本轮权重对全 TRAIN 重新打分，再按原 Zipf 混淆分布采负例。保留优化器状态，恢复验收后执行第 144–286 步。 |
| Stage2 | 从第 286 步初始化并固定参考。按 209 规则从本轮全 TRAIN 分数构建难例与正确回放。沿用双向 TPO 与动态概率间隔，6 卡训练加 2 个单卡推理实例，global batch 384，跑一轮，步数为实际视图人口除以 384 向上取整。 |
| 评测 | Stage1 与 Stage2 端点在原封存面板上配对。覆盖 SLURP selection、clean TEST、完整官方 TEST，MINDS native14、SIB-FLEURS topic7、MMSU 多义词。以 Top-1 macro-F1、准确率、平均最高原始概率为主，MMSU 报准确率。附候选级概率误差与配对区间。1.7B 重新验证 HF 概率一致性，MINDS 同形状合批。 |

用相同面板比较 209 和 210 的对应端点，并在 210 内比较 Stage2 相对 Stage1 的增量。模型选择在源域完成。源 TRAIN 校准 harness 阈值，冻结后只用于补充诊断。

[209 配方](stage1-cards-tpo-209.md) · [210 模型配置](../configs/experiments/jev-210-context-design.json) · [派发配置](pipelines/JEV-210.json) · [实验台账](../wiki/reports/JEV-210.md)
