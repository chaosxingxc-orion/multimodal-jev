# ceph `datasets/` 与 `models/` 归一（v1）

**日期**：2026-11-16 　**根**：`/mnt/cephfs/xingchao` 　**范围**：`models/`、`datasets/`

用户要求（2026-11-16）：*「数据，模型要归一，不要散落在各地；其他数据先放一放，
交给其他任务去修复。」*

本文交付**归一契约**：一份声明式清单 + 一个只读工具 + 一层唯一入口视图 + 一份实测目录。
**没有移动、复制、重命名或删除任何既有 ceph 目录**——原因见 §1。唯一的写操作是新增
44 条符号链接（`datasets/_families/`、`models/_families/`）与 2 个空目录。

## 1. 为什么是"视图"而不是"搬迁"

仓库里有 **1,595 个文件**硬编码了 ceph 绝对路径，其中大量是 hash 绑定的冻结证据与
`pins`：

| 被引用路径 | 引用次数 |
| --- | --- |
| `datasets/slurp/slurp_real` | 116,836 |
| `datasets/slurp-jev-benchmark/official-clean-intent-v1` | 1,263 |
| `models/Qwen__Qwen3-ASR-0.6B-hf`（含 7 个逐文件引用） | 616 + 1,588 |
| `models/qwen3-omni-30b-a3b-instruct` | 155 |
| `datasets/fsc/extracted` | 149 |
| `datasets/audio2tool` | 102 |
| `datasets/slurp-clean/clean-v7-a3cd74482c32b74f` | 96 |
| `datasets/jev/fsc` | 61 |
| `models/multimodal-jev/fsc-sft-init` | 40 |
| `models/local-training/{verl_v030_20261029,nga}` | 27 + 27 |

物理搬迁会直接破坏复现与既有审计（AGENTS.md 明确禁止改冻结运行包 / `pins`）。
因此归一的做法是：**旧路径永久保持有效**，在其之上叠加唯一命名入口。

## 2. 归一后的布局

```
/mnt/cephfs/xingchao/
  datasets/
    _families/<family>/<stage>/      # 唯一入口（符号链接视图，可为空）
    <family>-<stage>/...             # 现有物理位置，保持不变
  models/
    _families/<org>__<Model>/        # 唯一入口（符号链接视图）
    <org>__<Model>/                  # 现有物理位置，保持不变
  artifacts/<experiment>/<run>/      # 本项目训练产物（尚未建立）
```

规则三条：

1. **一个数据族只占一个顶层目录**，版本/阶段用 `stage` 名区分，不再出现
   `slurp/`、`slurp-native/`、`slurp-clean/`、`slurp-jev-benchmark/`、
   `slurp-jev-full/` … 各自为政。
2. **`models/` 只放上游权重**，本项目训练产物属于 `artifacts/`。
3. **顶层只允许清单声明过的条目**；出现未声明的一级目录，`--check` 直接失败。

## 3. 交付物

| 文件 | 作用 |
| --- | --- |
| `configs/ceph-assets-v1.json` | 声明式清单：7 个数据族（64 条 stage 路径）、6 个模型组、错位登记、可回收登记、顶层契约 |
| `tools/ceph_asset_catalog.py` | 只读目录：`--check` 校验 / `--report` 出目录 / `--sizes` 单次 `du -sb` 计量 / `--materialize-views` 幂等建视图 |
| 本文 | 结论与迁移边界 |

```bash
python -m tools.ceph_asset_catalog --check                 # 契约校验，0 problems
python -m tools.ceph_asset_catalog --check --sizes --report -   # 带体积的目录
python -m tools.ceph_asset_catalog --materialize-views --dry-run # 先看会建哪些链接
```

当前结果：`OK 64 declared assets, 7 dataset families, 13 hardlink mirror groups`，
0 个问题、0 个未声明顶层目录。

## 4. 实测体积

`models/` **427.6 GiB**、`datasets/` **228.2 GiB**。

> 之前 `host-disk-cleanup-20261116.md` 里那张表的数字（models 116.9 GB、
> datasets 168.2 GB、omni 478.7 GB）来自 `getfattr -n ceph.dir.rbytes`，
> **该属性在本挂载上返回空**，那些数字是错的。本文数字全部来自**一次** `du -sb`
> 调用（必须一次，见下）。

### 4.1 13 组 hardlink 镜像（不是重复，不能删）

`models/` 与 `datasets/` 下的 13 个一级目录，其**文件**与 `omni/` 同名目录是
同一 inode（`nlink=2`）。抽样比对命中率 **100%**：

```
models/{jev, lco-embedding-omni-3b-2605, lco-embedding-omni-7b,
        omni-embed-nemotron-3b, qwen3-asr-1.7b, qwen3-embedding-0.6b,
        qwen3-omni-30b-a3b-instruct, qwen3-omni-30b-a3b-thinking, serving-views}
datasets/{audio2tool, earnings21-22, jev, slurp}
```

所以 `du -sb models … omni/models …` 必须**一次**跑完：`omni/models` 记 0 字节、
`omni/datasets` 只记 125,681,519 字节，因为 inode 已经在 `models`/`datasets` 下
计过一次。分两次跑会重复计算，`ceph.dir.rbytes` 也没法用。

### 4.2 `models/` 明细

| 目录 | GiB | 角色 |
| --- | --- | --- |
| `local-training` | 111.98 | **错位**：训练产物（`nga` 属其他项目 + `verl_v030_20261029`） |
| `jev` | 78.25 | 上游 jev/decider 参考模型 zoo（44 个） |
| `qwen3-omni-30b-a3b-instruct` | 65.69 | 上游权重 |
| `qwen3-omni-30b-a3b-thinking` | 59.09 | 上游权重 |
| `serving-views` | 34.20 | **派生物**，见 §5 |
| `shlv__AudioJev` | 17.53 | 上游权重 |
| `lco-embedding-omni-7b` | 16.65 | 上游权重 |
| `multimodal-jev` | 11.70 | **错位**：本项目训练产物 |
| `lco-embedding-omni-3b-2605` / `omni-embed-nemotron-3b` | 8.78 / 8.78 | 上游权重 |
| `qwen3-asr-1.7b` / `Qwen__Qwen3-ASR-1.7B-hf` | 4.38 / 3.81 | 同一模型两种序列化 |
| `Qwen__Qwen3-ASR-0.6B` / `-0.6B-hf` | 1.75 / 1.47 | 同一模型两种序列化 |
| `Qwen__Qwen3-ForcedAligner-0.6B` / `qwen3-embedding-0.6b` | 1.71 / 1.12 | 上游权重 |
| `speechbrain__*` | 0.64 / 0.03 | 上游权重 |

### 4.3 `datasets/` 明细

| 目录 | GiB | 目录 | GiB |
| --- | --- | --- | --- |
| `slue-phase-2` | 95.96 | `mac-slu` | 1.77 |
| `jev` | 41.29 | `fdb-v3` | 1.57 |
| `slurp` | 12.63 | `mmau-mini` | 1.13 |
| `spokenwoz` | 11.94 | `tau2-voice` | 0.86 |
| `voicebench` | 10.43 | `minds14` | 0.53 |
| `mmau` | 9.84 | `fsc-jev` | 0.47 |
| `audio2tool` | 9.72 | `slurp-jev-full` | 0.44 |
| `mmar` | 6.26 | `versions` | 0.27 |
| `earnings21-22` | 6.16 | `sfc-jev` | 0.15 |
| `fsc` | 3.51 | `slurp-native` | 0.11 |
| `mac-slu-spans-train-16k-v1` | 3.08 | `fsc-native` | 0.10 |
| `mmsu` | 2.94 | `slurp-clean` | 0.09 |
| `sfc-bench` | 2.59 | `audio2tool-jev` / `sfc-native` | 0.04 / 0.04 |
| `mac-slu-spans-train-diagnostic-v2` | 2.31 | 其余 10 项 | < 0.01 |
| `slurp-jev-benchmark` | 1.92 | | |

## 5. 散落与重复的判定

### 5.1 一个族散在多个顶层目录

| 族 | 散落处 | 数量 |
| --- | --- | --- |
| SLURP | `slurp`、`slurp-clean`、`slurp-native`、`slurp-jev-benchmark`、`slurp-jev-full`、`slurp-jev-reviewed`、`slurp-jev-{rule,selection-rule,balanced-rule}-controls-v1`、`jev/slurp` | **10** |
| mac-slu | `mac-slu`、`mac-slu-binding-information-v1`、`mac-slu-spans-train-{16k,diagnostic-v1,diagnostic-v2}` | 5 |
| FSC | `fsc`、`fsc-jev`、`fsc-native`、`jev/fsc` | 4 |
| SFC | `sfc-bench`、`sfc-jev`、`sfc-native`、`jev/sfc-official` | 4 |
| MMAU | `mmau`、`mmau-mini`、`jev/mmar-official` | 3 |
| audio2tool | `audio2tool`、`audio2tool-jev` | 2 |

`datasets/jev` 尤其混用：28 个第三方 HF 数据集（`org__name`）**和**我方 pipeline
阶段（`fsc`、`slurp`、`sfc-official`、`mmar-official`、`evaluation`、
`fsc-semantic-review-packet-v1`、`fsc-semantic-transfer-train-inventory-v1`、
`fsc-span-transfer-test-population-v1`）同处一级。清单里已按 `family` 拆开登记。

### 5.2 不是重复，别删

- **ASR `-hf` 与 repo 布局**：`Qwen__Qwen3-ASR-0.6B`（`vocab.json`/`merges.txt`、
  分片 `model-0000N-of-0000M.safetensors`）vs `Qwen__Qwen3-ASR-0.6B-hf`
  （`tokenizer.json`/`processor_config.json`/`chat_template.jinja`/`VERIFICATION.json`、
  单文件 `model.safetensors`）。checkpoint 数不同，是同一模型的**两种序列化**。
- **`models/jev` 的 Base/Instruct 成对**：Qwen3.5 9B/4B/2B/0.8B 各两份
  （18.00/8.70/4.26/1.65 GiB），是可选择起点，不是冗余。

### 5.3 唯一的真·派生物（可回收 34.2 GiB，但需跨项目确认）

`models/serving-views/*-v1` 与源权重逐项相等——tensor 数量、dtype、shape 全同，
**总字节差恰好为 0**，差异只有键名前缀 `audio_tower.*` → `thinker.audio_tower.*`
和重新分片：

| 派生视图 | 源权重 |
| --- | --- |
| `serving-views/lco-embedding-omni-7b-v1` | `lco-embedding-omni-7b` |
| `serving-views/lco-embedding-omni-3b-2605-v1` | `lco-embedding-omni-3b-2605` |
| `serving-views/omni-embed-nemotron-3b-causal-v1` | `omni-embed-nemotron-3b` |

理论上可由源权重 + 前缀重写脚本重新生成，删除回收 34.2 GiB。**但该目录由 omni
服务侧消费，属其他项目**，按用户决策 2 不动，只登记在
`configs/ceph-assets-v1.json` 的 `reclaim_register`。

## 6. 错位的训练产物

| 路径 | GiB | 现状 | 目标 |
| --- | --- | --- | --- |
| `models/multimodal-jev` | 11.70 | 本项目训练产物（`container-audio-rl-two-step-v1` 5.85、`vllm-cpu-chain-sft-v3` 2.93、`fsc-sft-init` 2.93） | `artifacts/` |
| `models/local-training` | 111.98 | `nga`（**其他项目**，2026-11-09 后无更新）+ `verl_v030_20261029` | `artifacts/` 或交回 nga |

迁移前提：先改 52 处引用，且对冻结包保留兼容链接。本轮只登记，不动。

## 7. 已做 / 未做

**已建立**唯一入口视图（44 条符号链接，全是新条目，未改动任何既有路径）：

```
datasets/_families/{slurp,fsc,sfc,audio2tool,mac-slu,mmau,jev-zoo}/<stage>
models/_families/<name>
```

```bash
ls /mnt/cephfs/xingchao/datasets/_families/slurp/
# balanced-rule-controls  benchmark  clean-v7  full  legacy-root  native
# raw  reviewed  rule-controls  selection-rule-controls
```

幂等已验证：再跑一次 `created 0 / already_present 44 / conflicts 0`。指向错误时工具
拒绝覆盖并以 1 退出，不会静默改链接。

- **未建立** `artifacts/`（见 §6）——需先改 52 处引用。
- **未移动、未删除**任何 ceph 文件。`archive/`、`omni-retained/`、`experiments/`、
  `docker/` 一律按用户决策 2 交给其他任务。

## 8. 其他项目：175 的 NGA

`/home/xingchao/workspace/nga` 已停更（最后写入 2026-11-09，15 个
`engineering_k1_*` / `formal_*` 运行目录），属**另一个账号**的 NGA 项目。
175 上 `sudo` 需要密码，本会话无法自行删除，请手动执行：

```bash
ssh npu 'sudo rm -rf /home/xingchao/workspace/nga'
```

同目录下还有 `/home/xingchao/workspace/multimodal-jev`（本项目）和
`omni → /mnt/cephfs/xingchao/omni`（软链），**不要**一起删。

## 9. 边界

- 本文对应的写操作仅限新增 44 条符号链接与 `_families/` 两个目录；未移动/删除/
  重命名任何既有 ceph 目录，未改冻结运行包、`pins` 或配置，未停止任何容器。
- 体积数字来自单次 `du -sb`（§4.1），不是 `ceph.dir.rbytes`。
- `--check` 的顶层契约是**回归护栏**：将来出现未声明的一级目录会失败，而不是
  静默通过。
