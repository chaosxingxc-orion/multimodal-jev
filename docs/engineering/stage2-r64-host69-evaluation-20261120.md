# 0.6B r64 Stage2 step 68：69 评测与释放记录

本轮正式训练为 `next-simple-r64-dp8-v1`。68 步训练独立审核通过，初始化是同轮 Stage1 step 160。Stage2 末点策略 ID 为 `9b1507151124d1477ad94e4605410fa99ed3d15792f3fa06cd8c311196835b6f`。本轮评测仅在 69 派发。

SLURP clean TEST 的 12,991 条录音使用同一封存清单和候选顺序。256 条入场对照的 HF 和服务输出完全一致；全量评测经独立审核，并从原始预测复算。

| 检查点 | Top-1 macro-F1 | Top-1 准确率 | 平均 Top-1 原始概率 |
| --- | ---: | ---: | ---: |
| Stage1 step 160（Stage2 step 0） | 80.97% | 84.54% | 75.87% |
| Stage2 step 68 | 80.43% | 85.19% | 88.80% |
| 变化（百分点，按未舍入值） | −0.53 | +0.65 | +12.93 |

准确率增加 85 个正确样本，但 macro-F1 降低。平均最高概率升高不代表概率校准改善。

截至 2026-11-20 16:00:56 UTC，按卡时截止要求停止了 MINDS 的全量分片和 selection 后续队列，核验本轮无存活容器、NPU 0–7 无运行进程、69 的共享锁可取。MINDS step 68 的 8 个有限分片精确复现对照，全量未完成。SLURP selection step 0/68 完成 HF 对照，全量未执行；step 64 未派发。FSC 为历史参考。

本地操作根目录为 `/srv/multimodal-jev-xingchao/operations`，证据相对路径：

- Stage2 训练审核：`next-simple-r64-dp8-v1/stage2-windows/formal/audit.json`。
- SLURP TEST 独立审核与复算：`jev-stage2-r64-host69-193/slurp/run/audit-full-68-collected.json` 与 `jev-stage2-r64-host69-193/scoring-evidence/slurp_test/`。原始预测 SHA-256 为 `9c7f62778433b58571d2f3a817ebd715720ddd2823f8cff32b6ce61a935e653d`。
- Stage1 对照：`jev-stage1-r64-host69-endpoint-186/scoring-evidence/slurp-top1-rescore.json`。
- 未完成的评测封包及有限对照：`jev-stage2-r64-host69-194-minds-hf/`、`jev-stage2-r64-host69-195-selection/`。
