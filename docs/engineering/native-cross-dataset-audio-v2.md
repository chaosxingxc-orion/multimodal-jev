# 原生 FSC / SLURP / SFC 音频哈希交叉核查

对三个固定 native release 的全部九个划分核查请求清单哈希、完整请求数和
source group 数，再比较预先登记的音频文件 SHA256。工具为
`tools/jev/audit_cross_dataset_audio.py`，9 项测试覆盖重复 ID、错 split、
请求/manifest 变化、覆盖不足、非法哈希和多音频计数。实际结果见
[完整核查](native-cross-dataset-audio-v2-result.json)。

九个划分合计 111,333 个请求；36 对划分之间共享的音频文件哈希均为零。
其中跨数据集 27 对、同数据集跨划分 9 对。未打开独立 target 文件或解码音频；
哈希继承已固定的输入 release，本次不重新读取波形验证文件内容。
SLURP manifest 本身包含既有 TRAIN 重复文件的标签说明，工具只使用其版本和
划分计数/请求哈希；因此准确口径是“未读 target 文件”，不是“未读任何标签元数据”。

| 数据集 | 划分请求数 | 发现 |
| --- | --- | --- |
| FSC | train 23132 / valid 3118 / test 3793 | train 有两条请求共享一个字节哈希，unique=23131 |
| SLURP | train 50627 / valid 8690 / test 13078 | 使用已冻结的官方 scorer 按 filename 去重版本；train 原始 50628 条目有一条重复，不更改既有处理 |
| SFC | train_id 5534 / test_id 2590 / test_ood 771 | 后续回合重用同会话音频，分别有 1035/604/178 个哈希被多请求引用，符合 supplied-history 输入结构 |

进一步核对 FSC 当前提示优化的固定 learning64 / selection256：两侧各有
64/256 个唯一音频哈希，交集为零；上述两条重复录音均不在该分区内，见
[分区核查](fsc-prompt-audio-overlap-v1.json)。无需修改当前实验。
未来全量 FSC train-derived 分区必须按音频连接组聚合，不能只按旧 source_group
区分两条重复录音；完整原生官方划分和测试分母保持不变。

初版 v1 对 SLURP manifest 错用统一的 identity.dataset 字段而退出，未生成结果。
修订为识别明确的 slurp-native-release-v1 schema 和 recordings 计数后以新输出目录
完成 v2；原 plan 和失败 terminal 保留。本次未清理或改写任何数据集。

这些结论只排除三个 release 之间的完全相同文件字节。转码、重采样、裁剪、
同源不同录音、文本/语义近重复以及预训练暴露均未覆盖；Audio2Tool/MMAR 等
未纳入本次分母。FSC/SLURP/SFC 已用于开发，仍只能称已知任务间迁移，不能因
本检查为零就改称未见数据集泛化。
