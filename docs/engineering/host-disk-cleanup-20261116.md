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
- `/var/lib/omni-hot-runtime` 244 GB、`/home/xingchao/omni-hot-runtime` 156 GB、
  `/data/xingchao/s2-dev1850-hot-preparation-v1` —— 其他项目存活挂载（已由
  `docker inspect` 与运行中 `row_operator_probe.py` pidfd 等待确证），**未触碰**。

结论：181 的 `/data` 仍 99% 满（110 GB 空闲），但占用主体是 omni 的 244 GB + 156 GB，
不属于本项目。本项目侧可回收空间已在 `/var/tmp` 回收完毕。

## `/opt` 专项（175 + 181，**本轮不删**）

用户要求「继续清理 175 和 181 下面 `/opt` 的临时文件」。实测结论：**两台主机的 `/opt`
都没有可按「>1 天」规则删除的本项目文件**——175 只有 12 KB 属主为 root 的 NGA 探针，
181 的全部非系统内容属**另一个正在推进的任务**。本轮 `/opt` 未删除任何文件。

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
| `omni-hot-runtime/` | root | **2026-11-16 15:29** | **21,330,111,740 B** | **其他项目的活跃任务，不动** |

`/opt` 下**恰好 8 个条目超过 1 天**，全部是上表前四行的系统项；其余 44 条都属于
`omni-hot-runtime` 且为当天写入。也就是说：**181 `/opt` 没有一条「>1 天且属本次可清理」
的条目**。

### 为什么 `/opt/omni-hot-runtime` 不能删

它是 omni 音频塔跨后端（HF vs vLLM）分层对齐诊断扫描的产物树，`diagnostic-38`
（05:39）到 `diagnostic-61-terminal-main-split61-v1`（15:14）共 **71 个 generation 目录**，
最大三项为 `diagnostic-46` 2.59 GB、`diagnostic-48-…-capacity-failed` 1.95 GB、
`diagnostic-42-…-rope42-v1` 1.14 GB。判定依据（全部为实测，非推测）：

1. **仍在推进**：整棵树最后一次写入为 **15:29**（观察时刻 15:32），`stage-54…stage-61`
   每 20–30 分钟一个新目录，且最新 `reclaim-diagnostic61-copy.jsonl` 同样写于 15:29。
2. **有活进程**：`pgrep` 命中
   `lease_controller.py --dispatch-plan /var/lib/omni-hot-runtime/…/diagnostic-63/dispatch-plan.json`
   （pid 1798791）与同目录 watchdog（pid 1799326）——扫描已到第 63 代，`/opt` 只是它的
   克隆/回收目标。
3. **它自己在回收**：树内 20 余份 `reclaim-diagnostic*-{original,copy}.jsonl` 每份都以
   `{"event":"complete","files":…,"bytes":…}` 收尾，单代回收约 3.2 GB
   （如 `reclaim-diagnostic61`：1017 文件 / 3,247,579,648 B）。**我们再删就是和它的回收
   循环抢同一批文件**，会破坏它的证据链。
4. **不是 `/home/xingchao/omni-hot-runtime` 的副本**：两者无同 inode 文件
   （`nlink` 全为 1），且 `/home` 那棵树里没有任何 `*diagnostic*` 条目；
   在 omni 工程中也搜不到对 `/opt/omni-hot-runtime` 的引用——它是按代重建的临时区。
5. 结构上每代都带 root 属主的 `COMPLETE.json` / `FINAL-AUDIT.json` /
   `external/*-STAGE-PINS.json` / `watchdog-inspect.json`，属**冻结证据**（本账号读不了，
   `Permission denied`），AGENTS.md 禁止删运行证据。

结论：按用户决策 3「暂时不动，有一些其他项目还在驱动」，本轮**登记不删**。
可回收条件写成可执行判据，不靠人记：

```bash
# 满足全部三条才可清理 /opt/omni-hot-runtime：
#   (a) 整树 24h 内无写入  (b) 无 lease_controller/reclaim 进程  (c) 最新 reclaim jsonl 的
#   "complete" 事件之后再无新 generation
ssh npu2 'sudo find /opt/omni-hot-runtime -xdev -newermt "-1 day" | wc -l'   # 须为 0
ssh npu2 'pgrep -af "omni-hot-runtime"'                                      # 须为空
# 三条都满足后才执行：
# ssh npu2 'sudo rm -rf /opt/omni-hot-runtime'
```

预计届时释放约 **21 GB**（181 `/` 现 238G/298G used，占 84%）。注意**任何单次读数都只是
快照**：15:18 一次分目录统计读到 32 G，15:32 精确 `du -sb` 为 21.33 GB，差值来自它自身
按代回收——不要拿不同时刻的数字相互校验。

### 181 真正紧张的位置（不在 `/opt`）

同一时刻 `df`：`/` 238G/298G（84%）、`/var` 780G/984G（84%，其中 `/var/lib` **774 G**
为容器镜像/层）、`/data` **5.5T/5.9T（99%，仅 110 G 空闲）**、ceph 63T/65T（98%）。
`/opt` 的 21 GB 排在最后。`/var/lib` 与 `/data` 的占用主体是 omni 的
`/var/lib/omni-hot-runtime`（244 GB）与 `/home/xingchao/omni-hot-runtime`（156 GB），
属其他项目，按决策 3 未触碰——已在上一节登记，不作为本项目清理目标。

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
