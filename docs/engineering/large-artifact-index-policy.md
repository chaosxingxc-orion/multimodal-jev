# 大体积证据文件的索引策略（GitHub 基座）

日期：2026-11-16
范围：本仓库工作树（212 开发机），不涉及 175/181 主机磁盘，也不删除任何本地文件。

## 背景

用户决策：**GitHub 不传大体积文件**。工程基座由 GitHub 统一管理，但逐条推理记录、
`docker inspect` 快照、全人口审计明细这类机器生成的大 JSON 不进入远端索引。

仓库当前状态（2026-11-16 实测）：

- `.git` 14 MB，历史 713 个提交，历史中最大 blob 1.9 MB
  （`docs/engineering/omni-fp32-decoder-divergence-v1-verified.json`，已跟踪，保留）。
- `HEAD` 跟踪 2715 个文件、共 19.5 MB。
- 工作树共 190 MB（不含 `.git`/`.local`/`.venv`），其中 9 个未跟踪 JSON 合计约 115 MB。

风险点不是历史，而是**下一次 `git add -A`**：这 9 个未跟踪文件会被一次性索引进远端。

## 策略

1. **阈值**：单个文件 > 1 MB 的机器生成证据不进入索引；≤ 1 MB 的正常跟踪。
   纯二进制产物（已由 `.gitignore` 的 `*.pt`/`*.safetensors`/`*.ckpt` 等规则覆盖）不受影响。
2. **以指纹代替内容**：这些文件仍保留在本地磁盘，供审计复算；GitHub 侧只保留
   路径 + 字节数 + SHA-256 + 内容说明（下表）。任何依赖它们的下游文档已经按 hash 绑定，
   指纹表让远端读者知道"本地有一份、内容是哪一份"，而不必搬运 payload。
3. **不删除、不改写**：本策略只做索引排除，不改动文件内容，因此不影响既有
   `proof.sha256`、`manifest` 引用与冻结运行包。
4. **新增同类产物**：生成脚本把大 JSON 写到 `.local/` 或 `experiments/`（均已在
   `.gitignore` 中）时无需额外操作；若写到 `docs/`，须同时在此表登记。

## 当前豁免清单（>1 MB，未跟踪）

| 字节数 | SHA-256 | 路径 | 内容与价值判断 |
| --- | --- | --- | --- |
| 71,418,006 | `34defc0ae54c9362e51ce31aa408f85774c5e5ab00b4ddbc27f55869bc0341c5` | `docs/engineering/context-admitted-full-npu-v1.json` | Stage1 全量 634 步终态记录。体积 99.5% 来自 `rank_updates`（逐步 × 四 rank 遥测）。原始 payload 不入库，结论已在 `context-admitted-full-npu-v1` 相关 md 汇总。 |
| 12,767,519 | `157f11d26419eeb7b901a96a5a1bacc8cfb3afd606bdae07efbe3bdc8b3f4457` | `docs/engineering/context-checkpoint-shared-npu-v3.json` | checkpoint 生命周期 NPU 验收记录，含完整 `docker inspect` 与逐 rank 明细。 |
| 12,767,420 | `170d427120b84bb92102f29035f0fd7bda85c818df29c751bde39c0b2942fbe5` | `docs/engineering/context-online-shared-npu-v4.json` | 同容器在线运行验收记录，同上结构。 |
| 6,386,789 | `c9f589d0c75b5c2d963efc1fe0419b38b1588dab00a2f513f9411a6850e0940c` | `docs/engineering/context-online-shared-resume-npu-v4.json` | 恢复重放验收记录。 |
| 4,967,311 | `d022f32e4a7a362495d3911d35fd44fa90933a544a51236f505be138740c2a37` | `docs/research/minds-local-audio-audit-v1.json` | MINDS 8168 条音频全量审计明细；汇总结论在 `docs/research/64-minds-local-transfer-readiness.md`。 |
| 3,372,902 | `d14d6ca2b8e4894a4a2b5ffa906e057f530fb70646d0086ca585e3634ff15120` | `docs/research/context-stage1-native-structure-diagnostic-v1.json` | 9226 selection 逐条结构诊断明细；结论在 `context-stage1-native-structure-diagnostic-v1.md`。**注意**：该文件通过 `proof.sha256` 绑定下方 step0634 记录。 |
| 1,312,238 | `9b8360a67e0574271bf701f5b01648c7fd7aa65c396dcd55513c90523f113323` | `docs/engineering/slurp-batch-updated-components-v1.json` | 批量更新组件隔离诊断逐条明细。 |
| 1,080,417 | `7e501aaa90177a4e85ff286ba64e9fe5a35c06eac103d4a34b72ddc237ace088` | `docs/engineering/context-stage1-selection-step0634-v2.json` | Stage1 终点 9226 全量概率记录，被上表诊断文档按 hash 引用。 |
| 1,079,070 | `cdc71f70211a0223041f561629e1f540b782ae723118c8bb389843402af6cdc4` | `docs/engineering/context-stage1-selection-step0000-v2.json` | Stage1 起点 9226 全量概率记录。 |

合计 115 MB。全部为本地证据快照，删除或改写会破坏已有 hash 绑定，因此只做索引排除。

## 校验方式

本地复算（不联网、不删除）：

```bash
sha256sum docs/engineering/context-admitted-full-npu-v1.json   # 与上表对比
git check-ignore -v docs/engineering/context-admitted-full-npu-v1.json
```

`git check-ignore` 命中即说明该文件不会被远端索引。

## 边界

- 本策略不是"清理"，不回收磁盘空间；主机侧清理见
  `host-disk-cleanup-20261116.md`。
- 不改变任何运行包、冻结配置或 `pins`；被排除的文件仅是不进入 GitHub，仍在本地可读。
- 未来若要真正瘦身（例如把 `rank_updates` 转成汇总表后删除原文件），必须另外走
  版本化流程并更新引用它的文档，不能借本策略静默替换。
