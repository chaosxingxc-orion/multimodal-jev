# 原生意图评测封存 v0.0.129

共享入口：`/mnt/cephfs/xingchao/experiments/multimodal-jev/native-evaluation-seal-v0.0.129/MANIFEST.json`。
SHA256：`31f5b64b70f48227932828fa0c775f264973479cbeb66eacbfbe10313970adea`。
69 已核对主清单和工程清单身份。共享副本逐文件核验通过，含 8,177 个数据文件和 1,988 个工程文件。封存时核验 24,952 条音频的源路径和内容哈希。

| 数据 | 划分 | 录音 | 有序候选 |
| --- | --- | ---: | ---: |
| SLURP clean TEST | 官方 TEST、既定质量清洗分母 | 12,991 | 60 |
| FSC native31 | 官方 TEST | 3,793 | 31 |
| MINDS native14 | 固定跨数据集评测人口，上游 split 为 train | 8,168 | 14 |

`inference/` 仅有音频引用、内容哈希和稳定前缀任务面板；`scoring/` 单独保存与样本和音频绑定的真值与分组。SLURP 和 FSC 引用共享原音频，MINDS 封存核验后的 16 kHz 音频。固定阈值为 0.5，按清单报告 Top1、macro-F1、集合 F1、exact set、NLL 和 Brier 等；目标集不用于选模、调阈值或训练。

`engineering/` 封存共享编译器、评测和评分工程源码及配置。生产入口为 `tools.reports.seal_native_evaluation`，模板为 `configs/portable/operations/seal-native-evaluation.json`。本包是数据及工程快照；既有运行器的旧布局缓存和旧源策略准入不能直接派发本轮评测。正式评测仍需绑定源域锁定策略、生成本轮稳定前缀编码缓存，并完成有限的设备数值和运行验收。未执行 NPU 评测，无目标指标。
