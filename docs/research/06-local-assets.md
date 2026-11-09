# 本地共享资源与清理

共享根目录为 `/mnt/cephfs/xingchao`。新语料进入 `datasets/`，模型进入 `models/`，评测/参考源码进入 `repos/`。本页区分已验证资源与访问限制；逐文件来源清单位于共享根目录 `asset-registry/multimodal-jev-20261109/`，本地执行凭证位于 `.local/assets/`。

## 目录兼容与复用

Omni 的旧容器有的只挂载 `omni/`，直接迁移后回链到父目录会使容器内路径失效。因此既有资源采用同文件系统硬链接共享视图，保留旧路径，不重复占用 payload 数据块。硬链接不是独立备份，两处修改会影响同一 inode；共享数据与权重应只读使用，训练输出写入新目录。

已整理原 Omni 的 13 组模型/数据目录，新增 153,864 个硬链接，另有 624 个文件原本已共享 inode。`jaredpalmer__kev-suites` 存在 126 个不同文件冲突，未覆盖任一版本；Omni 版本单列于 `datasets/versions/jaredpalmer__kev-suites/omni-snapshot-20261109/`。27 组旧性能测试的模型 checkpoint 另建到 `models/local-training/verl_v030_20261029/`。

## 下载与校验

- Qwen3-ASR-0.6B、ForcedAligner-0.6B、AudioJev 使用固定模型 revision；已有 Qwen3-ASR-1.7B 和 Omni 模型复用本地资源。
- 首批下载 SLURP（复用）、SFC-Bench、MAC-SLU、MINDS-14、Audio2Tool、SLUE Phase 1（受限）/2、MMSU、MMAU test/mini、MMAR、VoiceBench、FDB-v3、EVA 发布资产及 SpokenWOZ。原始归档可供后续按任务解压，归档下载完成不等于全部训练适配器已实现。
- SFC 的 Git 中音频是 LFS 指针；已解析指针、下载真实音频并校验 SHA256。8,895 个 WAV 均通过头部读取验证。
- FDB-v3 已取得发布 ZIP 并解压 100 个 `input.wav`；保存本地 SHA256，发布方没有提供独立校验和，不能称发布者签名验证。
- Audio2Tool 普通文件通过 pinned Git blob 校验；音频使用 LFS SHA256。批量 LFS 避免 3,000 次/5 分钟 resolver 限流；失败项可按原 manifest 重试。
- Fluent Speech Commands 已从用户指定的 OpenDataLab 镜像下载并解压，归档 SHA256 与发布清单匹配；30,043 条 WAV 逐条可读，官方 train/valid/test 和说话人划分均已核验。
- SLUE Phase 1 返回 **401 gated dataset**，公开 README 不代表音频允许匿名下载；当前环境无 HF 登录凭据。需账号接受官方条款并在机器登录后补齐，不用第三方镜像绕过门槛。
- τ² voice 数据/场景已从固定代码版本共享到 `datasets/tau2-voice/data/`；运行时 TTS/对话生成/judge 仍依赖单独配置的 provider，不声称全部离线可跑。EVA 的 HF 资产仅覆盖公开 airline 入口。

`tools/shared_assets.py` 支持不覆盖既有文件、哈希校验、断点分片和终态凭证。运行方式：

```bash
.venv/bin/python -m pip install -e '.[dev,assets]'
.venv/bin/python tools/shared_assets.py download \
  --root /mnt/cephfs/xingchao \
  --state .local/assets/download \
  --manifest .local/assets/download-manifest.json --workers 12
```

manifest、下载事件、逐文件校验凭证、分片状态和短时签名 URL 只保留在本机 `.local/assets/`。签名 URL 不入 Git。完成后共享根目录保存可复用的 catalog 和来源清单。小文件重算哈希，大文件凭证仅在 inode/size/mtime/ctime 和静置时间满足时复用；它是续传优化，不是对并发恶意写入的防护。

## 清理语义

用户已明确授权删除临时文件与训练中间产物。清理按文件清单执行，删除前复查 inode、size、mtime、link count 与目录边界。记录保留在 `.local/cleanup/`，它是删除审计，不能恢复已删除字节。

已删除 2026-10-29 完成的 27 组性能测试的 58 份 optimizer 状态（153,762,372,638 字节），保留模型分片、配置、DONE、loss 与结果。另删除 27 份成功训练任务的旧 step 3000 checkpoint，保留 step 3116 和最终 adapter；27 组 NGA 最终 adapter 已共享到 `models/local-training/nga/`。清理后这些性能测试无法从旧 optimizer 状态精确续训。

缓存仅删除有源文件对应的旧 `.pyc`、pytest 缓存、已结束任务的旧运行时诊断日志。活跃 NGA/Omni 任务、当前恢复路径、下载残片、模型/数据和终态凭证保留；不能把名称含 scratch/lock/checkpoint 的所有文件一概视为垃圾。部分历史目录需通过 npu2 已有管理权限读取/清理，未修改其 ACL。

释放量按实际删除且为最后硬链接的文件块统计；Ceph 实时空闲量受其他用户和下载影响，不能直接作为本次删除量。

## 清理统计（2026-11-09）

扫描可访问的 4,122,479 个文件，实际删除 10,069 个文件；逻辑字节 163,056,421,416（约 163.06 GB / 151.86 GiB），最后硬链接分配块约 163.06 GB。初始扫描记录 1,694 个权限错误，其中部分已通过管理通道处理；未把无法判定的目录或活跃路径当作已清空。审计摘要见 [cleanup-summary.json](cleanup-summary.json)。

## 已完成的独立验证

- SLURP：real train/dev/test 共 72,396 条、synthetic 69,253 条录音引用无缺失；两个原始 tar.gz 的发布方 MD5 均匹配。
- SFC-Bench：8,895 条 LFS 音频 SHA256 匹配，8,895 个 WAV 头均可读。
- SpokenWOZ：20 个文件约 12.82 GB，通过发布对象哈希校验。
- Qwen3-ASR-1.7B：已有 12 个文件约 4.70 GB 与固定官方 revision 全量哈希匹配。
- Qwen3-ASR-0.6B、ForcedAligner-0.6B、AudioJev：计划中的全部权重及配置已校验，AudioJev LICENSE/Notice 已随资源保存。

数据集下载完成与完成解压、训练 loader、模型推理是独立状态；本任务没有运行训练或正式评测。

已另外删除 SLUE Phase 1 匿名访问失败留下的 10 个预分配残片（约 3.58 GB）；登录后从正式清单重新下载。

## 最终文件校验汇总

以下选定发布文件合计 **85,544 个 / 189.36 GB**，包含复用的 ASR-1.7B，未计既有 SLURP、其他历史模型、FDB 解压副本和源码；不能将其全部算成新增网络下载量。另有 FDB-v3 0.736 GB 原始归档与 100 条已解压输入、τ² 约 0.920 GB 场景/资产。

| 资源 | 校验文件 | 字节规模（GB） | 状态 |
| --- | ---: | ---: | --- |
| `datasets/mmau` | 4/4 | 10.569 | 已校验 |
| `datasets/slue-phase-2` | 9/9 | 103.033 | 已校验 |
| `datasets/slue` | 1/26 | 0.000 | 账号访问受限；仅 README |
| `datasets/minds14` | 16/16 | 0.567 | 已校验 |
| `datasets/mmsu` | 5007/5007 | 3.153 | 已校验 |
| `datasets/voicebench` | 76/76 | 11.200 | 已校验 |
| `datasets/mmar` | 11/11 | 2.994 | 已校验 |
| `datasets/eva-bench` | 3/3 | 0.000 | 已校验 |
| `datasets/mac-slu` | 8/8 | 1.905 | 已校验 |
| `datasets/audio2tool` | 71441/71441 | 10.411 | 已校验 |
| `models/Qwen__Qwen3-ASR-0.6B` | 9/9 | 1.881 | 已校验 |
| `models/Qwen__Qwen3-ForcedAligner-0.6B` | 9/9 | 1.840 | 已校验 |
| `models/shlv__AudioJev` | 17/17 | 18.819 | 已校验 |
| `datasets/sfc-bench` | 8895/8895 | 2.703 | 已校验 |
| `datasets/spokenwoz` | 20/20 | 12.823 | 已校验 |
| `models/qwen3-asr-1.7b` | 12/12 | 4.703 | 已校验 |
| `datasets/fsc` | 3/3 | 1.544 | 已校验并解压 |
| `datasets/mmau-mini` | 3/3 | 1.213 | 已校验 |

完整路径、来源和缺失列表见 [local-assets.json](local-assets.json)。批量任务 exit_code=1 的 14 个条目均为 SLUE Phase 1 受限 LFS 文件；主清单该资源另有 11 个受限小文件，共缺 25/26 个，已如实保留失败记录。

## FSC 补充下载

共享目录：`/mnt/cephfs/xingchao/datasets/fsc/`，可用数据位于 `extracted/fluent_speech_commands_dataset/`。保留原始归档、镜像说明、原发布方许可与 `VERIFICATION.json`。官方划分 train 23,132 / valid 3,118 / test 3,793，共 30,043 条；97 名说话人、31 个组合意图，三个划分无音频路径或说话人重叠。全部 WAV 头可读且采样率为 16 kHz。

归档 SHA256：`4376699f7daf134a9fa57a1d880ffcaaf94a3e2551ba0b40ad894d7abb71aacb`。未下载相同哈希的 `.tar.1.gz` 副本及 `.aria2` 状态文件。镜像 CC BY-NC 4.0 标签不能替代原发布方专用许可。

## NGA 运行时兼容权重与 Jev 派生数据

另下载并逐文件核验官方 `Qwen/Qwen3-ASR-0.6B-hf` 和 `1.7B-hf`，合计
18 文件 / 5,664,027,655 字节；原始权重保持独立。版本及源哈希见
[HF 模型清单](../engineering/hf-models-manifest.json)。

FSC 首版派生发布位于 `datasets/jev/fsc/62625c057b1b45dcf9c94443fd68915ac0f05f9986619be6ac7b3a12419b9b1c/`：
train 138,792 / valid 18,708 / test 22,758 个决策，共 180,258 个，复用原有 30,043 条录音。
六个视图不构成六倍独立样本；音频不复制。协议与校验见
[数据契约](../engineering/data-contract.md)。

SLURP Jev 发布已生成于
`datasets/jev/slurp/ba31cd72120ab28309fe40089274b6b70e9a9605c2fcd56e5de4a0323805f3f8/`。
数据复用真实录音；只写派生 JSONL，不复制音频。59,014 录音 / 413,098 决策，
保留官方划分并增加三个业务域留出。原始 SLURP 数据未变。
[发布记录](../engineering/slurp-transfer-v1-receipt.json) 与
[全量复核](../engineering/slurp-transfer-v1-verification.json) 均已保存。
