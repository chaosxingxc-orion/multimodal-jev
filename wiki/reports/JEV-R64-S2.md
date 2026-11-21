# JEV-R64-S2　双信用 GRPO 末点评测

本轮正式训练为 `next-simple-r64-dp8-v1`。68 步训练独立审核通过，初始化是同轮 Stage1 step 160。Stage2 末点策略 ID 为 `9b1507151124d1477ad94e4605410fa99ed3d15792f3fa06cd8c311196835b6f`。SLURP TEST 和 MINDS 的历史评测在 69，selection 末点在 181 补齐。

SLURP clean TEST 的 12,991 条录音使用同一封存清单和候选顺序。256 条入场对照的 HF 和服务输出完全一致；全量评测经独立审核，并从原始预测复算。

| 检查点 | Top-1 macro-F1 | Top-1 准确率 | 平均 Top-1 原始概率 |
| --- | ---: | ---: | ---: |
| Stage1 step 160（Stage2 step 0） | 80.97% | 84.54% | 75.87% |
| Stage2 step 68 | 80.43% | 85.19% | 88.80% |
| 变化（百分点，按未舍入值） | −0.53 | +0.65 | +12.93 |

准确率增加 85 个正确样本，但 macro-F1 降低。平均最高概率升高不代表概率校准改善。

MINDS native14 是源模型的跨数据集迁移评测，共 8,168 条、14 个语言地区，来自上游 TRAIN；这不是 MINDS 的官方测试集。两点使用同一封存清单与候选顺序，由独立 HF 分片直接读取音频，结果经审核与复算。

| 检查点 | Top-1 macro-F1 | Top-1 准确率 | 平均 Top-1 原始概率 |
| --- | ---: | ---: | ---: |
| Stage1 step 160（Stage2 step 0） | 71.64% | 72.00% | 32.58% |
| Stage2 step 68 | 66.05% | 66.47% | 49.07% |
| 变化（百分点，按未舍入值） | −5.59 | −5.53 | +16.49 |

Stage2 在 MINDS 少判对 452 条；平均最高概率更高的同时，分类效果退化。

首轮卡时截止后，于 2026-11-20 16:00:56 UTC 核验释放 8 卡。获得新卡时后，SLURP TEST 沿用已审核结果；MINDS 和 SLURP selection 复用已核验的有限 HF 对照，从新运行包继续。按用户指定范围，selection 只评 step 0/68，step 64 未派发。本轮队列未包含 FSC。

SLURP selection step 0 完成 9,226 条的全量审核和原始预测复算：Top-1 macro-F1 83.05%，准确率 85.53%，平均 Top-1 原始概率 76.66%。step 68 已通过有限对照；按用户释放指令停止全量推理，最终进度文件记录 896 / 9,226 条，无全量成绩。

在 181 上接续 step 68。首版尝试将新版评测入口装入旧快照，有限对照因缺少该入口依赖的模型接口而失败。第二版完整复用先前通过对照的运行时快照，并在 181 上复用同一缓存封存清单和 256 条 HF 参考，重新核验有限对照后再跑全量。

181 的有限对照逐条一致，概率和 logit 最大绝对差均为 0。全量容器用八卡运行约 12 分 39 秒，退出后释放租约；CPU 独立审核又用约 2 分 30 秒，确认 9,226 条全部到齐。step 68 的 Top-1 macro-F1 为 82.78%（−0.27 个百分点），准确率为 85.67%（+0.14 个百分点），平均 Top-1 原始概率为 88.66%（+12.00 个百分点）。与 SLURP TEST 的方向一致：最高概率显著升高，准确率略增，macro-F1 下降。

续跑的最后一个推理容器于 2026-11-20 16:46:28 UTC 退出。随后现场核验本轮无运行容器、NPU 0–7 无运行进程、69 的共享锁可取。

本地操作根目录为 `/srv/multimodal-jev-xingchao/operations`，证据相对路径：

- Stage2 训练审核：`next-simple-r64-dp8-v1/stage2-windows/formal/audit.json`。
- SLURP TEST 独立审核与复算：`jev-stage2-r64-host69-193/slurp/run/audit-full-68-collected.json` 与 `jev-stage2-r64-host69-193/scoring-evidence/slurp_test/`。原始预测 SHA-256 为 `9c7f62778433b58571d2f3a817ebd715720ddd2823f8cff32b6ce61a935e653d`。
- MINDS 独立审核与复算：`jev-stage2-r64-host69-196-minds/run/audit-hf-shards-collected.json` 与 `jev-stage2-r64-host69-197-selection/scoring-evidence/minds/`。原始预测 SHA-256 为 `18b202371376cb8fa7111135af1ed145d8b2c85c2a22049c5c4c111ddf7b6eec`。
- Selection step 0 独立审核与复算：`jev-stage2-r64-host69-197-selection/run/audit-full-0-collected.json` 与 `jev-stage2-r64-host69-197-selection/scoring-evidence/slurp_selection_0/`；停止与释放收据为同一运行根目录的 `RELEASED.json`。
- 181 上的 selection step 68：`jev-stage2-r64-selection-181-v2/run/audit-full-68-collected.json` 与同目录下的有限对照、容器退出及终态收据。原始预测 SHA-256 为 `ed3082dae452d7f21f1ee0f9b56a3132173071068aa68d6b92451ac648aea015`。
- Stage1 对照：`jev-stage1-r64-host69-endpoint-186/scoring-evidence/slurp-top1-rescore.json`、`jev-stage1-r64-host69-minds-hf-188/scoring-evidence/top1-rescore.json`。
- 复用的有限对照：`jev-stage2-r64-host69-194-minds-hf/`、`jev-stage2-r64-host69-195-selection/`；新包的 `REUSED.json` 钉住从首轮复用的输出。

## 复算

准备报告列出的原始预测、独立评测审核收据和封存面板 `native-evaluation-seal-v0.0.129`。面板清单 SHA-256 为 `31f5b64b70f48227932828fa0c775f264973479cbeb66eacbfbe10313970adea`。在仓库根目录运行以下命令，把三个变量绑定到对应证据文件或目录。

```bash
python3 -m tools.reports.summarize_audited_top1 \
  --predictions "$PREDICTIONS_JSON" --audit "$AUDIT_JSON" --seal-root "$SEAL_ROOT"
```

MINDS 的独立 HF 分片复算还需加 `--hf-policy-id`，值为上文对应检查点的冻结策略 ID。输出中的 `top1_metrics_independently_recomputed` 应为 `true`。两个数据集的末点应分别得到 0.8043405512 和 0.6605126048 的 Top-1 macro-F1。

<!-- experiment-package:begin -->
## 实验入口执行记录

队列状态 **complete**。领取提交 `15012e18537789ce649724f0f704dcf7593026b4`，台账行 SHA-256 `77571a9c14cbe94d4d4b6bd6c56d536290656711a0a25f16c9aaf51dacc56990`，协议 SHA-256 `0d44407cef85c7e020b844e55654da73da55fa02f8fd538ab006f59c29e26ca8`。
本次配置 SHA-256 `bc83230b6411b361969683470a360f20af1730c25b0a0e75c2dfb4dc51e5b1c5`。

| 阶段 | 步骤 | 状态 | 执行方式 | 证据或待办 |
| --- | --- | --- | --- | --- |
| data | sealed-selection | audited | reused | `stage2-selection-host181-v0.0.156/campaign/entry/selection/MANIFEST.json` SHA-256 `4995beb4588e7e98143223623632dc8bb1dc3db3c84dadd31bdcb4ec6fe5d26a` |
| train | audited-stage2 | audited | reused | `next-simple-r64-dp8-v1/stage2-windows/formal/audit.json` SHA-256 `e89e60f06a101f425ccfc593053a55bc53d4b28f2b1a94710d652f49208f1168` |
| evaluate | selection-step68-v2 | audited | executed | `jev-stage2-r64-selection-181-v2/run/terminal.json` SHA-256 `5d7e7862ce75250b6f2c13901d5bdacc7352a92c4db3868670a02c9afd78c5e5`；`jev-stage2-r64-selection-181-v2/run/audit-parity-68-collected.json` SHA-256 `dbdc17a2b1be1d1ba5892587f9af7daa960bbe61d60d72550d3b5c7e1ebda14a`；`jev-stage2-r64-selection-181-v2/run/audit-full-68-collected.json` SHA-256 `a007657a750d0a61ac677810753f16b1df1ac47d1a160aae09b672e41c141034`；`jev-stage2-r64-host69-197-selection/run/audit-full-0-collected.json` SHA-256 `eefbf482de1a5377662509c862963e9103adf0bcdb99d16dd4faa7744fab142b` |

### SLURP selection / Noul

SLURP selection（源域诊断）。配对人口 9,226。

| 策略 | 指标 |
| --- | --- |
| Step 68 | Top-1 macro-F1 82.78%、Top-1 accuracy 85.67%、平均原始 Top-1 概率 88.66% |
| Step 0 | Top-1 macro-F1 83.05%、Top-1 accuracy 85.53%、平均原始 Top-1 概率 76.66% |

相对 Step 0，Top-1 macro-F1 -0.27 个百分点、Top-1 accuracy +0.14 个百分点、平均原始 Top-1 概率 +12.00 个百分点。

相对主对照，SLURP selection（源域诊断）的 Top-1 macro-F1 下降。
<!-- experiment-package:end -->
