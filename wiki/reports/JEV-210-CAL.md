# JEV-210-CAL

[实验台账](../experiment-ledger.md)

## 评测效率基线

181 同一镜像、同一检查点、同一批 3,000 条音频，两组各四张卡。样本覆盖 SLURP clean TEST
（1,183 条）、SLURP clean selection（800 条）、MINDS（736 条）、SIB-FLEURS（160 条）、
MMSU（66 条）及官方 SLURP TEST 补集（55 条）。两组各输出 133,968 个候选概率，
与审核后的归档逐候选比较，差异均为 0。

| 每卡评分进程 | 评分墙钟 | 含容器启动墙钟 | AI Core 平均 | HBM 平均 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1145.3 秒 | 1184.2 秒 | 19.80% | 22.30% |
| 6 | 307.4 秒 | 353.3 秒 | 86.16% | 78.31% |

锁定每卡 6 个进程：同样本评分吞吐 **3.73 倍**，计入容器启动 **3.35 倍**。
固定样本倍率与正式评测的全量墙钟分开报告；旧版不再补跑全量。正式评测沿用冻结源 TRAIN
校准，复用审核通过的 Stage1 预测，对 CAL Stage2 的完整留出集评分。

收据：`jev-card-eval-210-concurrency-screen-v0.0.4/ops/screen-audit.json`，
SHA-256 `4956ede3e5d79b40aa11adcbbf5dce513396d824b5fd06ee9c53167555e57557`。

## 与 209、210 正式评测的耗时对比

三轮均在 181 使用八卡，Stage2 末点各重新评分同一批封存留出输入（30,890 条独立音频）。本轮
CAL 末点为 step 217，初始化取自 JEV-210 Stage1 step 1362。检查点不同；同检查点的并发收益见上节。

| 实测范围 | JEV-209 | JEV-210 | JEV-210-CAL | CAL 对 209 / 210 的速度倍数 |
| --- | ---: | ---: | ---: | ---: |
| Stage2 全部留出评分（30,890 条） | 1:56:37 | 1:34:48 | 0:23:50 | 4.89× / 3.98× |
| 其中 SLURP clean TEST（12,991 条） | 1:41:51 | 1:31:03 | 0:22:12 | 4.59× / 4.10× |
| 整轮容器运行窗口 | 4:00:03 | 5:41:35 | 0:31:04 | 工作范围不同 |

评分窗口从对应任务的首个输入到最后一块提交；数据集在同一队列交错执行，行间耗时不能相加。
整轮容器窗口从首个容器启动到最终评分容器退出。CAL 从封包到收集、评分与审核完成的完整墙钟
为 **31:24**。209、210 重新评分两个端点，210 另含源 TRAIN 阈值拟合；CAL 复用已审核的
Stage1 预测和冻结校准。整轮节省包含上述复用收益。

CAL 正式八卡 AI Core 平均 **89.70%**，HBM 平均 **86.40%**，评分吞吐 **1,017.31 候选/秒**。
复用 969 个 Stage1 块，重算 969 个 Stage2 块并产出 1,455,224 个概率；全量覆盖审核通过，另与归档
同检查点的 93 块共 141,760 个概率逐项比较，差异为 0。

计时证据根：181 `/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/`。
209 对应 `jev-card-eval-209-v0.0.8/`、210 对应 `jev-card-eval-210-stratified-v0.0.2/`，
使用各自的评分进程收据与 Docker 启停记录。CAL 对应 `jev-card-tpo-calibrated-replay-v0.0.2-eval-v0.0.5/`，
正式核验见 `formal-throughput-audit.json`（SHA-256
`ff35097745763d521c4579a75ef56c87b74fb63671e8880b803c3bf4dc3f21e1`）。

<!-- experiment-package:begin -->
## 实验入口执行记录

队列状态 **complete**。领取提交 `07865ef2df70ecca1a5f341cbb94dfac0811f4ac`，台账行 SHA-256 `f881d8a0aa25f5dc04b26170407efe2fa4313a7756c43a0b860c1f24f59cec1d`，协议 SHA-256 `f74220269763240aba7471a58a694c81b27439f2db156f9ad3121f8cca0fc8ed`。
本次配置 SHA-256 `1a586e815de14b806a3fc95200153a6f2e468b3fc3cb825c82ba819db68d158a`。

| 阶段 | 步骤 | 状态 | 执行方式 | 证据或待办 |
| --- | --- | --- | --- | --- |
| data | build-calibrated-view | audited | reused | `MANIFEST.json` SHA-256 `93d986c5a5e4f9b5cf7879bc21b00c72f5aeb3478ef2ec07da342aa3c7192b51` |
| train | bind-fused-stage2-parent | audited | reused | `parent.json` SHA-256 `f7786ce899efabe1c02b08e1e11d18a8a64f4631805cf6cbdfa1dad5e96e5d7a` |
| train | freeze-fused-stage2 | audited | reused | `operations-local.json` SHA-256 `65d93175101364a2b0f1d8b491bb30fcbd3e96e4083b6bc07222b6de7738d95e` |
| train | train-fused-tpo | audited | reused | `infer-0-collected.json` SHA-256 `c45a9fd4271a9d25c731de8ce5700135f18c32b815c4b53f0b4b9dde0be3b801`；`infer-1-collected.json` SHA-256 `3fec99c791e1c2999683c6c2200cdc6aab0595783062146aa58aa87901edc422`；`finite-collected.json` SHA-256 `7bdd96d885129a8fc496fe0f6a785271b29f5722268eb3bb5f92c7b6d1cf005e`；`tpo-collected.json` SHA-256 `5b9e36dc5aafd1cf225e45a757d0846623a173729a4d712ffd9e8ad8b4d3d166` updates=217、endpoint=d1aeb45dfd39cd4b50b4f5f10f2a2db3df0b626259acbb36a6b5b62551079b5e；`terminal.json` SHA-256 `5d7e7862ce75250b6f2c13901d5bdacc7352a92c4db3868670a02c9afd78c5e5` |
| evaluate | evaluate-stage1-and-fused-stage2 | audited | executed | `score-collected.json` SHA-256 `f1e742c874186d9518b83a4894dfdcba1eeca050d78f36ea94a5460353272b0b` SLURP_clean_TEST60 stage1 vs 209 ΔF1=0.06386122269102745、SLURP_clean_TEST60 stage1 95% CI=[0.04506317166588324, 0.07597817552952221]、SLURP_clean_TEST60 stage1 结论=改善、SLURP_clean_TEST60 tpo vs 209 ΔF1=0.01988023875644196、SLURP_clean_TEST60 tpo 95% CI=[-0.001095364657565373, 0.03206554825027925]、SLURP_clean_TEST60 tpo 结论=不确定、MINDS_native14 stage1 vs 209 ΔF1=0.2334545614279977、MINDS_native14 stage1 95% CI=[0.2106278377698914, 0.2547287211490495]、MINDS_native14 stage1 结论=改善、MINDS_native14 tpo vs 209 ΔF1=-0.013640594100520154、MINDS_native14 tpo 95% CI=[-0.03131395557576655, 0.0037077022027189083]、MINDS_native14 tpo 结论=不确定；`tpo-collected.json` SHA-256 `5b9e36dc5aafd1cf225e45a757d0846623a173729a4d712ffd9e8ad8b4d3d166`；`receipt.json` SHA-256 `872258d6471788aef39d66532e8b261e773597fee1f4050e3831b257b4a3d742`；`terminal.json` SHA-256 `5d7e7862ce75250b6f2c13901d5bdacc7352a92c4db3868670a02c9afd78c5e5`；`formal-throughput-audit.json` SHA-256 `ff35097745763d521c4579a75ef56c87b74fb63671e8880b803c3bf4dc3f21e1` 八卡 AI Core 平均占用 (%)=89.70317388283068、八卡 HBM 平均占用 (%)=86.40358395380261、固定样本评分吞吐倍数=3.7254365474431967、固定样本容器耗时倍数=3.3520879657213083、正式评分吞吐（候选/秒）=1017.3059048590338、正式评测墙钟（秒）=1884.3899492695928、CAL 逐项比对候选数=141760 |

### SLURP_clean_TEST60

SLURP clean TEST。配对人口 12,991。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 85.90%、平均最高概率 96.82%、Top-1 macro-F1 82.30% |
| Stage1 full epoch endpoint | Top-1 准确率 85.53%、平均最高概率 96.69%、Top-1 macro-F1 81.04% |

相对 Stage1 full epoch endpoint，Top-1 准确率 +0.37 个百分点、平均最高概率 +0.13 个百分点、Top-1 macro-F1 +1.25 个百分点。

相对主对照，SLURP clean TEST的 Top-1 macro-F1 上升。

### SLURP_clean_SELECTION60

SLURP clean selection。配对人口 9,226。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 86.62%、平均最高概率 96.82%、Top-1 macro-F1 83.47% |
| Stage1 full epoch endpoint | Top-1 准确率 86.84%、平均最高概率 96.67%、Top-1 macro-F1 83.77% |

相对 Stage1 full epoch endpoint，Top-1 准确率 -0.22 个百分点、平均最高概率 +0.15 个百分点、Top-1 macro-F1 -0.30 个百分点。

相对主对照，SLURP clean selection的 Top-1 macro-F1 下降。

### MINDS_native14

MINDS native14。配对人口 8,168。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 55.30%、平均最高概率 95.59%、Top-1 macro-F1 49.80% |
| Stage1 full epoch endpoint | Top-1 准确率 57.82%、平均最高概率 95.13%、Top-1 macro-F1 54.96% |

相对 Stage1 full epoch endpoint，Top-1 准确率 -2.52 个百分点、平均最高概率 +0.46 个百分点、Top-1 macro-F1 -5.16 个百分点。

相对主对照，MINDS native14的 Top-1 macro-F1 下降。

### SIB_FLEURS_eng_topic7

SIB-FLEURS 英语话题。配对人口 320。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 79.06%、平均最高概率 95.85%、Top-1 macro-F1 76.42% |
| Stage1 full epoch endpoint | Top-1 准确率 77.19%、平均最高概率 95.22%、Top-1 macro-F1 73.79% |

相对 Stage1 full epoch endpoint，Top-1 准确率 +1.88 个百分点、平均最高概率 +0.63 个百分点、Top-1 macro-F1 +2.63 个百分点。

相对主对照，SIB-FLEURS 英语话题的 Top-1 macro-F1 上升。

### MMSU_polysemy

MMSU 多义词。配对人口 98。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 92.86%、平均最高概率 96.56% |
| Stage1 full epoch endpoint | Top-1 准确率 91.84%、平均最高概率 96.55% |

相对 Stage1 full epoch endpoint，Top-1 准确率 +1.02 个百分点、平均最高概率 +0.01 个百分点。

相对主对照，MMSU 多义词的 Top-1 准确率 上升。

### SLURP_official_TEST60

SLURP 完整官方 TEST（历史口径诊断）。配对人口 13,078。

| 策略 | 指标 |
| --- | --- |
| Stage2 末点 | Top-1 准确率 85.55% |
| Stage1 full epoch endpoint | Top-1 准确率 85.18% |

相对 Stage1 full epoch endpoint，Top-1 准确率 +0.37 个百分点。

相对主对照，SLURP 完整官方 TEST（历史口径诊断）的 Top-1 准确率 上升。
<!-- experiment-package:end -->
