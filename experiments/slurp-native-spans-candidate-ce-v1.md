# 候选内 CE 与全词表 CE 的固定预算对照

启动前固定：对照 `slurp-native-spans-balanced-v2`，相同 reviewed release、seed17、40 条选中学习录音、相同 20 个互补 batch，每任务每批正负 1/1，120 项监督，原始音频。相同本地 Qwen3-ASR-0.6B-hf、语言 q/v rank4/alpha8 LoRA、lr1e-4、AdamW、clip1、单 epoch。只将 loss 从全词表 masked CE 改为候选内 masked NLL；两者均保留完整 LM head 前向，便于计算预算比较。

正式检查：训练前全部 48 条输入 token/位置/波形 receipt/候选概率与基线一致；batch schedule 完全一致；参数配置除目标和源码快照外一致。记录更新有限性及全部学习人口的 before/after/silence 正确数、每类召回、balanced accuracy、完整三项全对数。按 source recording 配对重算预测变化。新增 8 条未参与此轮更新的录音仍属于既有学习人口，不能转称独立测试集。

当前基线为 after=70/142、silence=70/142，三个任务 balanced accuracy 0.5。没有任务收益的预设成功阈值；负结果也保留，不按学习诊断最佳结果选择正式模型。此实验是初始化/分类目标对照，不是 RL、CoT 或学术泛化证明。

运行根 `/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-native-spans-candidate-ce-v1`，物理卡 4，固定镜像，离线，硬截止 3600 秒。

完成退出0，独立比较确认两臂配置除目标/源码外相同，全部输入 token/位置/波形凭证/标签相同、before 概率完全相同、40 条选择及20次更新顺序完全相同。

| 学习人口诊断 | 全词表 CE | 候选 CE |
| --- | --- | --- |
| 正确数 /142 | 70 | 92 |
| 三项全对 /46 | 5 | 10 |
| 等长度静音正确数 /142 | 70 | 80 |
| 等长度静音三项全对 /46 | 5 | 10 |

候选 CE 相对全词表 CE 修正53项、破坏31项，净增22项。三任务 balanced accuracy 分别为 date=.5208、time=.7576、timeofday=.6667；静音为 .5/.7064/.5。没有独立测试证据，时长等捷径仍可能存在，不能将学习人口改善表述为泛化、CoT或RL收益。全组合全对仍与静音一致，也不能凭总正确率声称已解决音频依赖问题。

见 [同条件比较](../docs/engineering/slurp-span-objective-comparison-v1.json) 与 [独立指标验收](../docs/engineering/slurp-native-spans-candidate-ce-v1-verified.json)。
