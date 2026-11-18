# 175 / 181 主机磁盘清理记录（2026-11-16）

范围与授权（用户决策，逐条执行）：

1. 只清理 **175、181**；212 不动。
2. 超过 1 天未修改的临时文件删除；ceph 侧 `archive` 类同样规则，但**延后**执行。
3. 其他项目仍在驱动的目录**不动**。
4. 仓库变更**分批 PR**。
5. 主要目标是回收**本地**可用空间。

清理前对每个候选目录都做了存活检查：`docker ps` / `docker inspect` 挂载点、
`/proc/*/cwd`、`fuser -m`、`find -newermt -1 day`；并在删除循环内对每一项重新
`stat` 判龄，`mtime` 距今不足 86400 秒的一律跳过，避免误删正在写入的目录。

## 175（k8s-node1，账号 `x00930083`）

| 位置 | 处理 | 结果 |
| --- | --- | --- |
| `/home/x00930083` | 删除 `s-c175-input-stage-v1.tar.gz`（481,836,571 B）、`s-c175-protocol-v2-delta.tar.gz`（104,016,072 B），均为 2026-11-09 | `/home` 可用 27,710 MB → 28,275 MB（+561 MB） |
| `/var/tmp` | 删除 `CLOSURE.json`(4 MB)、`own.tar.gz`(2 MB)、`source-manifest.json`(2 MB)、`MAIN-C-QUEUE-RELEASE-c6a.private.json`、`verify-release.py` | `/var` 可用 579,415 MB（余量充足） |
| `/var/tmp/s2_next_strategy` | 部分清理，57 MB → 1 MB | 剩余 `fixed01/output/check/*` 属 root，本机 `sudo` 需密码，**保留** |

保留项（决策 3）：

- `/var/tmp/omni-npu175-experiment-v1` 2,196 MB、`/var/tmp/omni-npu175-b175-minus-r01-v2` 573 MB
  —— 其他项目在驱动。
- `/var/tmp/mkinitramfs_*` 665 MB —— 系统 initramfs 残留，非本项目。
- `/home/xingchao/workspace/nga` 3.4 GB —— NGA 项目（另一项目），属主 `root`、
  非 git 仓库、内容仅 `docker/ experiments/ tools/`，最后写入 **2026-11-09**，
  自那以后无更新。**待删除**，因 175 `sudo` 需密码，交由用户执行：
  `ssh npu 'sudo rm -rf /home/xingchao/workspace/nga'`。
  同目录下 `multimodal-jev`（本项目）与 `omni → /mnt/cephfs/xingchao/omni`（软链）
  **不可一并删除**。（此前记录写成 `/home/x00930083/workspace/nga`，路径有误。）
- `/var/tmp/omni-confirmed-only-c-overlay.tar.gz` —— 实为指向 ceph
  `omni-retained/npu175/var/tmp/` 的符号链接（0 字节），非占用来源。
- `/data` 下 `jev-macro-train-v1/v2/v4/v5`、`jev-macro-v1/v2` 等历史目录约 600 MB ——
  `/data` 仍有 3.1 TB 空闲，回收价值低于误删风险，**保留待确认**。

结论：175 的 `/home` 已用 96%（620 G / 680 G），但绝大部分属于其他用户。
本项目已自行回收约 0.56 GB；另有 **3.4 GB 的停更 NGA 目录**待用户执行 `sudo`
删除（见上）。**175 不是本项目的空间瓶颈。**

## 181（A800I-03，账号 `x00930083`）

`/var/tmp` 是本次唯一的实质回收点，共清理 1,003 个条目、合计约 31.1 GB：

| 批次 | 对象 | 条目 | 释放（du 求和） |
| --- | --- | --- | --- |
| 2a | `p-only-*`、`p-teacher-*`、`p-binding-*`、`p-node-*`、`p-test-*`、`p-integrated-*`、`codex-*`、`tfgrpo-*`、`efficiency-manifest-*`、`gepa-*`、`local-teacher-*` | 96 | 10,086 MB |
| 2b | `psd-*`、`saea-*`、`omni-*`、`ace-*`、`activation9*`、`a64k-*`、`a-*/A-capture*` 文件 | 365 | 15,796 MB |
| 2c | 其余全部 >1 天条目（逐项判龄，root 属主回落 `sudo`） | 542 | 5,227 MB |
| — | **合计** | **1,003** | **≈31.1 GB** |

磁盘效果：`/var/tmp` 35,682 MB → **760 MB**；`/var` 可用 121 GB → **158,168 MB（+37 GB）**。

保留项：

- `p-teacher-candidate-activation-v1`（92 MB）—— **删除后由其他项目的实时流程在
  11:58:48 重新创建**，属活跃工作，按规则保留。
- `p-only-eval-data-v1`、`p-only-rework2-prepared`、`p-node-release-consumer-test`、
  `local-teacher-completion-replay` —— `mtime` 为当天，判龄跳过。
- `systemd-private-*`、`mkinitramfs_*` —— 系统目录。
- `/var/lib/omni-hot-runtime` 244 GB、`/home/xingchao/omni-hot-runtime` 156 GB
  —— 其他项目存活挂载（已由 `docker inspect` 与运行中 `row_operator_probe.py`
  pidfd 等待确证），**未触碰**。

> 更正（2026-11-16T16:0xZ）：本节早先版本把 `/data/xingchao/s2-dev1850-hot-preparation-v1`
> 也列为「其他项目存活挂载」，**该判断是错的**。复核证据：`docker ps` 全部运行容器的
> bind 清单中没有任何 `/data/*` 路径；`/proc/*/cwd` 无进程位于 `/data` 下；该目录
> `mtime` 为 2026-11-13T02:31:29Z，远超 24 小时判龄线。早先据以判定的
> `row_operator_probe.py` 等待的是 `/var/lib` 下的路径，不是 `/data`。
> 该目录及其余我方 `/data` 条目已于 16:06:44Z 删除（见下节）。

结论（更正后）：181 的 `/data` 仍 99% 满，占用主体是**其他账号的模型权重**
（`Qwen3-VL-235B-A22B-Instruct` 942.7 GB、`GLM-5.2-w8a8c8` 764.2 GB、
`wanghongliang` 740.3 GB、`DeepSeek-V3.2-Exp-w8a8` 694.5 GB 等），
与 omni 无关 —— omni 的 244 GB + 156 GB 分别在 `/var` 与 `/home`，不在 `/data`。
本项目在 `/data` 的占用本来就只有约 482 MB，删掉也改变不了 99% 的占用率；
能真正释放的 omni 容量在 `/var/lib` 与 `/home`。

### `/data` 专项执行（2026-11-16T16:06:44Z）

用户问「/data 我们占了多少，是否有清理的空间」，并要求 omni / jev /
multimodal-jev 的容量释放掉。先量后删：

| 项 | 值 |
| --- | --- |
| `/data` 总量 | 6,390,492,389,376 B（约 5.8 TiB） |
| 清理前已用 | 5,950,562,897,920 B（**99%**），可用 117,790,167,040 B（约 110 GiB） |
| 我方占用 | ≈ 482 MB（**约 0.007%**）；`/data/xingchao/` 下 7 个一级条目 |
| 删除后已用 | **5,950,080,491,520 B**，可用 **118,272,573,440 B** |
| 实际释放 | **482,406,400 B ≈ 460 MiB**，`RM_FAIL=0` |

删除前的存活检查（三项全过，缺一不删）：

- 全部运行中容器的 bind 清单里**没有任何 `/data/*` 路径**（逐容器 `docker inspect`
  的 `Mounts` 逐条过滤）。
- `lsof +D /data/xingchao` 为空，`/proc/*/cwd` 无进程落在 `/data` 下。
- 7 个一级条目 mtime 全为 2026-11-13，远超 24 小时判龄线。

删除的 6 个目录（第 7 个 `/data/xingchao/omni` 是指向 ceph 的软链，0 本地字节，保留）：

- `/data/xingchao/s2-dev1850-hot-preparation-v1`（此前误判为"其他项目存活挂载"）
- 其余 5 个我方历史输出目录，逐项登记在 `manifest.txt.gz`。

收据：`/var/backups/data-own-cleanup-20261116T160644Z/`
（`df-before.txt`、`df-after.txt`、`manifest.txt.gz` 409 行两级元数据、
`manifest.sha256` = `c3e733c837d01f3b93c613d7aa72cd818c08abbd50139c3de4bce183096648eb`）。
只存元数据，不存 payload。

**`/data` 的 99% 不可由我方回收到合格水位**：剩余占用是其他账号的模型权重，
即使把我方 482 MB 全部清空，可用空间仍是 118 GB / 5.8 TiB。要真正腾出 `/data`，
只能由那些权重的属主处理。

## `/opt` 专项（175 + 181）

用户要求「继续清理 175 和 181 下面 `/opt` 的临时文件」，随后明确指定
「`/opt/omni-hot-runtime` 就是这个目录，要清理掉，后续已经不会往这个目录下写东西了」。
结果：175 无可自行删除项（root 属主，`sudo` 需密码，交用户执行）；181 的
`/opt/omni-hot-runtime` **已于 2026-11-16T15:39:41Z 删除**，释放 **21,330,111,740 B**。

> 本节 15:30 的版本曾判定「登记不删」，理由是该目录仍在推进、有其他项目在驱动。
> 用户随后明确该目录可清理、且不会再写入；该判定被新指令取代。下方记录执行结果，
> 并把原判定依据与「判定反转所依据的实测证据」分开写，避免看起来像事后改口。

### 175 `/opt`（全部条目，快照 2026-11-16T15:30:22Z）

| 条目 | 属主 | mtime | 体积 | 判定 |
| --- | --- | --- | --- | --- |
| `containerd/` | root | 2025-01-22 | 4 K | 系统组件，保留 |
| `os_start_run.sh` | root | 2026-09-29 | 860 B | 主机启动脚本，**保留**（内含 ceph key，且不得复制进仓库） |
| `vllm/` | **shenlu** | 2026-03-12 | 28 K | **其他账号**的文件，保留 |
| `nga-rc1-protected-assets-probe-20261106/` | root | 2026-11-06 17:56 | 4,106 B | 过期探针，**待用户 sudo 删除** |
| `nga-rc1-protected-review-probe-20261106-v1/` | root | 2026-11-06 18:02 | 8,204 B | 过期探针，**待用户 sudo 删除** |

两个探针目录内容确认（只有一个 `file`、以及 `file` + 空 `directory` 的权限测试存根，
9 天无更新，`du -sb` 合计 **12,310 B**）。175 `/opt` 全树 **没有任何 24 小时内变更的条目**
（`find -newermt "-1 day"` = 0）。删除需 root 且 175 `sudo` 需要密码，请手动执行：

```bash
ssh npu 'sudo rm -rf /opt/nga-rc1-protected-assets-probe-20261106 \
                    /opt/nga-rc1-protected-review-probe-20261106-v1'
```

**不要**连带删除 `/opt/containerd`、`/opt/os_start_run.sh`、`/opt/vllm`（前者是
containerd 运行时，后者属于另一账号 `shenlu`）。收益 12 KB——175 的 `/` 本来就只占
24%（45G/198G），这一项纯粹是卫生清理，不是空间回收。

### 181 `/opt`（全部条目，快照 2026-11-16T15:30:33Z）

| 条目 | 属主 | mtime | 体积 | 判定 |
| --- | --- | --- | --- | --- |
| `cni/bin/` | root | 2025-09-18 | 232 M | CNI 插件（tuning/portmap…），保留 |
| `containerd/` | root | 2025-09-16 | 4 K | 系统组件，保留 |
| `data/usr/` | root | 2026-01-28 | 8 K | 空目录树，无引用，体积可忽略，暂留 |
| `os_start_run.sh` | root | 2026-09-29 | 868 B | 主机启动脚本，**保留** |
| `omni-hot-runtime/` | root | 2026-11-16 15:29（末次写入） | **21,330,111,740 B**（36,497 条、47 个 generation） | **已删除**（15:39:41Z，用户指令） |

删除前 `/opt` 共 5 个一级条目：前四行是系统项，第五行 `omni-hot-runtime` 是 omni
跨后端诊断的产物树。删除后 `/opt` 只剩前四项。上表体积取 `du -sb`；15:18 的一次
分目录统计读到 32 G，差异来自 omni 自身按代回收（见本节末快照说明）。

### 删除前的判定证据（2026-11-16T15:39，删前 2 秒）

15:30 的初判是「其他项目的活跃任务，不能删」，依据是它每 20–30 分钟新增一代、最新写入
就在观察前 3 分钟。用户随后明确该目录可清理且不再写入，于是按**「写入已停止」**重新取证：

| 检查项 | 结果 |
| --- | --- |
| `find -xdev -newermt "-5 min"` | **0**（间隔 24 分钟连续两次） |
| 最新条目 mtime | **15:29**（`reclaim-diagnostic61-copy.jsonl`、`reclaim-diagnostic46-copy.jsonl`） |
| 最新 generation | `diagnostic-61-terminal-main-split61-v1`，**15:14**；此后 26 分钟无新代 |
| 规模 | `du -sb` **21,330,111,740 B**、**36,497** 条、一级 generation **47** 个 |
| 活跃扫描写在哪 | `diagnostic-63` 的 `hot/`、`results/` 全在 **`/var/lib/omni-hot-runtime/…/diagnostic-63`**（15:30–15:38 持续写入）；`/opt` 停在 15:29 |

关键一点：**omni 的扫描确实还在跑，但写目标已经迁到 `/var/lib`**——`lease_controller`
（pid 1798791）与 watchdog（pid 1799326）都在操作 `/var/lib` 那棵树，`/opt` 这棵是残留的
历史代。这与用户「后续不会往这个目录下写东西」一致，也是判定反转的实质依据：原「不能删」
的三条理由（仍在推进、有活进程、它自己在回收）现在都不再作用于 `/opt`。

> 一级 generation 数：删除前实测 **47**；15:30 的旧版本文档写「71 个 generation 目录」
> 用的是更早、口径不同的一次统计，未再复核，二者不要交叉引用。

### 收据归档（先存后删）

每代都带 root 属主的 `COMPLETE.json` / `COMMIT.json` / `FINAL-AUDIT.json` /
`*-inspect.json` / `external/*-STAGE-PINS.json`，属运行证据（AGENTS.md 禁止直接删除），
因此先把收据打包留档：

```bash
S=/var/backups/omni-hot-runtime-opt-receipts-20261116T153915Z.tar.gz
find /opt/omni-hot-runtime -xdev -type f \
  \( -name COMPLETE.json -o -name COMMIT.json -o -name FINAL-AUDIT.json \
     -o -name '*-inspect.json' -o -name '*-STAGE-PINS.json' \) -print0 |
  tar --null -T - -czf "$S"
```

| 项 | 值 |
| --- | --- |
| 归档路径 | `/var/backups/omni-hot-runtime-opt-receipts-20261116T153915Z.tar.gz` |
| 成员数 | **2,194** 个收据文件 |
| 大小 / SHA256 | 18,771,088 B / `cb70ae6bb350d53d3db143f378a2bac169febe8d89b3966cde00dafc33756cae` |
| 具名清单 | 同目录 `….diag63-named.txt` |

`/opt` 里唯一被**活跃计划**点名的是 `diagnostic-63/dispatch-plan.json` 的
`known_terminal_receipts` 中的 6 份 `diagnostic-61-terminal-main-split61-v1/{COMMIT,
COMPLETE,controller-inspect,FINAL-AUDIT,npu-inspect,watchdog-inspect}.json`；已逐份确认
全在归档内（各命中 1 次）。这些是**来源存证**，不是逐次读取的运行依赖。

### 执行与磁盘效果

```bash
ssh npu2 'sudo -n rm -rf /opt/omni-hot-runtime'
```

| 指标 | 删除前 | 删除后 |
| --- | --- | --- |
| `/` 已用 | 238 G / 298 G（84%） | **218 G / 298 G（77%）** |
| `/` 可用 | 48 G | **68 G（+20 G）** |
| `/opt` 一级条目 | 5 | 4（`cni`、`containerd`、`data`、`os_start_run.sh`） |

`du -sb` 的 21,330,111,740 B（19.87 GiB）与 `df` 的 +20 GiB 相符（块分配与四舍五入）。

> **任何单次读数都只是快照**：15:18 一次分目录统计读到 32 G，15:29 精确 `du -sb` 为
> 21.33 GB，差值来自 omni 自身按代回收——不要拿不同时刻的数字相互校验。

### 残留与风险（如实记录）

- 运行中容器 `configured-sync-MAIN-instance-down-actual63-v1`（启动 15:30:17，命令是
  diag-63 的 `lease_controller.py`）带一条**只读** bind，源即被删的
  `/opt/omni-hot-runtime/…/diagnostic-61-terminal-main-split61-v1`。删除后容器内该路径
  `ls` 为空、`find` 无结果；容器仍 running（`oom=false`）、`docker logs --since 15:39`
  无新增输出。只读挂载无法写回已删 inode，因此**没有**「写进已删除目录」的数据丢失。
- **删除后 4 分钟的复核（15:43:47Z）**：`/opt/omni-hot-runtime` 未被重建（`absent`），
  `/opt` 仍是 4 项；diag-63 在 `/var/lib` 正常换段——15:38:21 写 `NPU-EXIT.json`，
  **15:43:21 写 `HF-SERVICE-RESTORED.json`**，即删除后仍在推进下一阶段。
  这是「该目录已不是写入目标」最直接的验证：活任务没有失败，只是继续用 `/var/lib`。
- **若该容器被重建**，这条 bind 的源已不存在，Docker 会按挂载方式重建空目录或报错；
  它引用的 diag-61 终态收据在上面的归档里，可恢复。
- **未停止**任何进程或容器：`lease_controller`(1798791)、watchdog(1799326) 均存活，
  操作的是 `/var/lib`。也未触碰 `/opt/cni`、`/opt/containerd`、`/opt/data`、
  `/opt/os_start_run.sh`（后者含 ceph key，**绝不入库**）。

### 181 真正紧张的位置（不在 `/opt`）

删除后的 `df`：`/` **218G/298G（77%，68 G 空闲）**、`/var` 783G/984G（84%，151 G 空闲）、
`/data` **5.5T/5.9T（99%，仅 110 G 空闲）**、ceph 63T/65T（98%）。

三处的占用主体各自不同，不要混为一谈（早先版本把后两者写成同一段，是错的）：

| 挂载点 | 占用主体 | 归属 |
| --- | --- | --- |
| `/var` | `/var/lib` 约 774 G，其中绝大部分是**容器镜像与层**；`/var/lib/omni-hot-runtime` 约 244 GB | `omni-hot-runtime` 属其他项目，按决策 3 未触碰 |
| `/data` | 其他账号的模型权重（见上节明细），**与 omni 无关**；我方 482 MB 已清空 | 非本项目 |
| `/home` | `/home/xingchao/omni-hot-runtime` 约 156 GB | 其他项目；**已于 16:04–16:15Z 回收 155.2 GiB**，见下节 |

**181 真正的瓶颈仍是 `/data`（仅 118 GB 空闲）**，而它不由我方占用，也不由 omni 占用。
`/opt` 已经不是因素。

## 全项目运行态保护（未受影响）

清理全程未触碰正在运行的简单任务效率实验：
`context-macro-eff-v1-micro8-attempt2-{client,service-0,service-1}`（镜像 `52ba94e1b6ba`）
及其依赖 `/data/xingchao/jev-eff-v1/micro8`、`simple-final-v1/context-learning-cache-v1`、
`context-learning-audio-mirror-v1`、`models/Qwen__Qwen3-ASR-0.6B-hf`。未停任何容器、
未修改冻结运行包或 `pins`。

## ceph `/mnt/cephfs/xingchao` 分析与建议（**本轮不删**）

> **更正（2026-11-16 晚）**：本节最初用 `getfattr -n ceph.dir.rbytes` 取体积，但该
> 属性在**本挂载上返回空**，据此得出的 1.71 TB 及下面各分项数字**全部作废**。
> 已改写为一次 `du -sb` 实测，见
> `ceph-datasets-models-normalization-v1.md`。

实测（**一次** `du -sb`，必须一次，因为 `models/`、`datasets/` 有 13 个一级目录
与 `omni/` 是**文件级**硬链接镜像，`nlink=2`；分次跑会重复计数）：

| 目录 | 实测 | 说明 |
| --- | --- | --- |
| `models/` | 459,093,454,708 B = **427.6 GiB** | 已对 `omni/models`（0 B）去重 |
| `datasets/` | 245,025,940,414 B = **228.2 GiB** | 已对 `omni/datasets`（125,681,519 B）去重 |
| `omni/models` | 0 B | 全部与 `models/` 同名目录共 inode |
| `omni/datasets` | 125,681,519 B | 仅未被 `datasets/` 镜像的余量 |

`omni-retained/`、`experiments/`、`docker/`、`repos/`、`asset-registry/`
本轮**未测量**——按用户决策 2「其他数据先放一放，交给其他任务去修复」，只登记不动，
也不再引用那些无效数字。

`models/` 与 `datasets/` 的逐目录明细、散落族清单（SLURP 10 处、FSC 4、SFC 4、
mac-slu 5、MMAU 3、audio2tool 2）、错位训练产物（`models/local-training` 112 GiB、
`models/multimodal-jev` 11.7 GiB）以及唯一真派生物（`models/serving-views` 34.2 GiB，
可由源权重按 `audio_tower.* → thinker.audio_tower.*` 前缀重写再生）见
`ceph-datasets-models-normalization-v1.md`；声明式清单为 `configs/ceph-assets-v1.json`，
只读校验工具为 `tools/ceph_asset_catalog.py`。

按决策 5「延后清理远端文件」，以上仅登记，本轮**未删除任何 ceph 文件**。

## 复现命令

```bash
# 判龄 + 存活检查 + 删除（181 /var/tmp 使用的模式）
now=$(date +%s)
for e in *; do
  m=$(stat -c %Y "$e"); [ $((now-m)) -lt 86400 ] && continue
  rm -rf "$e" 2>/dev/null || sudo rm -rf "$e"
done

# ceph 体积：一次 du -sb 覆盖所有一级目录（硬链接只在单次运行内去重）
du -sb /mnt/cephfs/xingchao/{models,datasets}/* \
       /mnt/cephfs/xingchao/omni/{models,datasets}/*

# 或者直接用归一工具（会自动做单次 du 与硬链接镜像检测）
python -m tools.ceph_asset_catalog --check --sizes --report -
```

不要用 `getfattr -n ceph.dir.rbytes`：本挂载返回空值。

## 边界

- 本文只记录已执行的主机侧清理；仓库索引策略见 `large-artifact-index-policy.md`。
- 未删除任何训练产物、数据集、模型权重或 ceph 文件。
- 未停止任何容器或后台驱动，未修改冻结运行包 / `pins` / 配置。
