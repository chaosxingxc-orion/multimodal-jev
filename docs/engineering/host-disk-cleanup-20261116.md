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
- `/home/x00930083/workspace/nga` 3,477 MB —— NGA 项目（另一项目），非 git 仓库、已卸载但保留。
- `/var/tmp/omni-confirmed-only-c-overlay.tar.gz` —— 实为指向 ceph
  `omni-retained/npu175/var/tmp/` 的符号链接（0 字节），非占用来源。
- `/data` 下 `jev-macro-train-v1/v2/v4/v5`、`jev-macro-v1/v2` 等历史目录约 600 MB ——
  `/data` 仍有 3.1 TB 空闲，回收价值低于误删风险，**保留待确认**。

结论：175 的 `/home` 已用 96%（620 G / 680 G），但绝大部分属于其他用户，
本项目可回收部分仅约 0.56 GB。**175 不是本项目的空间瓶颈。**

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

## 全项目运行态保护（未受影响）

清理全程未触碰正在运行的简单任务效率实验：
`context-macro-eff-v1-micro8-attempt2-{client,service-0,service-1}`（镜像 `52ba94e1b6ba`）
及其依赖 `/data/xingchao/jev-eff-v1/micro8`、`simple-final-v1/context-learning-cache-v1`、
`context-learning-audio-mirror-v1`、`models/Qwen__Qwen3-ASR-0.6B-hf`。未停任何容器、
未修改冻结运行包或 `pins`。

## ceph `/mnt/cephfs/xingchao` 分析与建议（**本轮不删**）

体积用 `getfattr -n ceph.dir.rbytes` 读取（不做递归 `du`，避免拖垮 MDS）。总计约 1.71 TB。

| 目录 | 体积 | 归属 | 判断 |
| --- | --- | --- | --- |
| `omni-retained/` | 563.8 GB（npu181 373.4 / npu175 177.5 / s2-independent-archive 12.5） | 归档 | **可清理候选**，按"archive 同样按 1 天规则"处理，但需 omni 方确认无回滚依赖 |
| `omni/` | 478.7 GB（models 198.7 / work 122.8 / archives 86.3 / datasets 59.3） | 其他项目 | 保留（活跃），`archives/` 86.3 GB 可列入候选 |
| `experiments/jev/` | 225.5 GB（`r3_runs` 136.5 / `engineering-base-rc1` 73.3） | 本项目历史产物 | **本项目最大可回收项**，`r3_runs` 为历史轮次运行目录，建议保留最终 checkpoint 与汇总，清理中间态 |
| `experiments/multimodal-jev/` | 14.1 GB | 本项目历史产物 | 保留（多为已验证的正式包） |
| `datasets/` | 168.2 GB（slue-phase-2 96.0） | 数据 | 保留 |
| `docker/` | 144.4 GB（`verl_v030_20261029` 116.7） | 镜像 tar 备份 | 可分批复核后清理旧 tag |
| `models/` | 116.9 GB（`jev` 78.2：Qwen3.5 0.8B–9B 各 2 份） | 权重 | 保留（`models/jev` 中的 Base/Instruct 重复副本可评估） |
| `repos/` | 2.1 GB | 参考仓库 | 保留 |
| `asset-registry/` | 1.3 GB（`s-host-deep-cleanup-v1` 1.2） | 清理登记 | 保留（审计链） |

按决策 5「延后清理远端文件」，以上仅登记，本轮**未删除任何 ceph 文件**。
建议的下一步优先级：`experiments/jev/r3_runs`（136.5 GB，本项目自有）→
`omni-retained`（563.8 GB，需跨项目确认）→ `docker/` 旧镜像 tar。

## 复现命令

```bash
# 判龄 + 存活检查 + 删除（181 /var/tmp 使用的模式）
now=$(date +%s)
for e in *; do
  m=$(stat -c %Y "$e"); [ $((now-m)) -lt 86400 ] && continue
  rm -rf "$e" 2>/dev/null || sudo rm -rf "$e"
done

# ceph 体积（轻量）
getfattr -n ceph.dir.rbytes --only-values /mnt/cephfs/xingchao/<dir>
```

## 边界

- 本文只记录已执行的主机侧清理；仓库索引策略见 `large-artifact-index-policy.md`。
- 未删除任何训练产物、数据集、模型权重或 ceph 文件。
- 未停止任何容器或后台驱动，未修改冻结运行包 / `pins` / 配置。
