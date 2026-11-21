# 69 → 181 实验交接（2026-11-19 本地时间）

用户已指定后续实验全部在 **181** 执行。69 的旧配置只用于追溯训练来源；新派发由本地宏解析为 181 的主机、运行目录、设备和资源锁。下表中的远端路径是本次核验记录，便携配置不写死这些路径。

## 69 收口

1.7B Stage1 v0.0.146 完成 318/318 步，四个 rank 的末步和第 318 步检查点已收集。训练容器 `f211905fbec8` 于 11 月 20 日 01:17:20 UTC 正常退出，未 OOM。独立审核通过，绑定 11 个预登记检查点；评测尚未完成。第 318 步策略 `2326e233c821b034014752e6a5bcd2bd81893b84dd6ae2424c072eaac50957c7` 固定为 Stage2 初始化与 reference。

[审核终态](/srv/multimodal-jev-xingchao/operations/context-1p7b-host69-v0.0.146/formal-finalized/terminal.json) · [训练审核](/srv/multimodal-jev-xingchao/operations/context-1p7b-host69-v0.0.146/formal-finalized/training-audit.json) · [检查点计划](/srv/multimodal-jev-xingchao/operations/context-1p7b-host69-v0.0.146/formal-finalized/source-evaluation-plan.json)（SHA-256 `4ec1941ec654751f7f5531f81f960d6ad3ef65c2a3f7eda6dfe3424cfb51d94d`）。

69 的 v151 主接续、v155 备用接续及 v157 打分监听均已按进程身份停止。原队列停在等待 Stage1 审核，1.7B 的 TRAIN256、全量打分及 Stage2 尚未派发，未产生可续用的评分块。01:25:44 UTC 现场核验八卡均无 NPU 进程；其他容器未动。历史 0.6B 的 69 打分、块和视图不用于 1.7B。

## 路径与身份

| 资产 | 181 的来源 / 用法 | 核验 |
| --- | --- | --- |
| 1.7B 基座 | Ceph `models/Qwen__Qwen3-ASR-1.7B-hf`，容器只读 | 按真实 Stage1 配置逐文件核对，包括模型权重；base ID `feecbef15963f282a2eeb36165f4b8ca127911b30330b62e23e87bc2c7e8c436`。同盘另有 `qwen3-asr-1.7b`，不能按名称互换。 |
| 1.7B LoRA/head | 来自控制端已收集的 Stage1 `policy-0318`，连同冻结 processor 和末步提交封包；部署时随包传到 181 | checkpoint manifest `0506acea994188ebf80715487c821d158a910ca298197abbb2ae55bcf9ef3936`。评测 step 0 来自同计划的初始化策略。跨阶段按原协议重置 optimizer。 |
| SLURP clean | Ceph `datasets/slurp-jev-benchmark/official-clean-intent-v1` | TRAIN 40,513；selection 9,226；clean TEST 12,991。发布清单 `38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`，181 已复核。旧发布证据中的控制端绝对路径由配置指定只读映射，并逐文件核验。 |
| 原生评测封存 | Ceph `experiments/multimodal-jev/native-evaluation-seal-v0.0.129` | 清单 `31f5b64b70f48227932828fa0c775f264973479cbeb66eacbfbe10313970adea`，181 已复核。FSC 3,793、MINDS 8,168；MINDS 是上游 TRAIN 的外域评测。 |
| 训练音频和特征 | 共享资产 `context-learning-audio-mirror-v1/run`、`context-prefix-cache-v0.0.115/cache`（均在 `experiments/multimodal-jev/shared-runtime-v0.0.125`） | 181 两份清单一致，SHA-256 分别以 `e7069e60`、`a76581a1` 开头。提交前还核对 TRAIN 人口、音频哈希、特征回执、processor、marker 与 `jev-stable-prefix-audio-readout-v1`。 |
| 181 评测特征 | 现有 0.6B 运行只读挂载各自 `jev-native-{slurp,fsc,minds}-136-prep/{panel-cache,full-cache}`；这些是 181 本地目录 | 后续任务须重核封存哈希、processor、布局、候选顺序及全量输入回执；其他主机路径不能直接代入。身份不符则从封存音频重建新版本缓存。 |
| 镜像与源码 | 镜像已在 181 本地 Docker 存在；共享 `shared-runtime-v0.0.125/verl-v030-20261109.tar.gz` 为补装来源 | 镜像 ID `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。源码从冻结工程包按 manifest 发到 181，运行时不引用控制端工作树。 |
| 热状态 | 181 本地 ext4 工作目录；运行、日志、IPC、临时缓存、checkpoint 各对应明确的本地路径 | 确认实际挂载、inode 和空间；共享 Ceph 模型、数据和已冻结缓存只读。遵守[存储规约](runtime-storage-policy.md)。 |

转移时不传整个本机工程目录。对每个输入路径先判定其角色：共享资产在 181 直接核验，控制端目录由打包器封入版本化包，目标端输出重新绑定并检查目录为空。包内逐文件哈希、真实挂载或张量回执出现差异时停止；目录名相同不能作为复用依据。验收和正式派发都使用解析后的配置，新产物用独立目录，旧收据和配置原样保留。

## 181 接续顺序

1. 0.6B Stage2 step 102 的 SLURP clean TEST、FSC native31 和 MINDS native14 均已审核，见 [0.6B 评测交接](stage2-preflight-host181-20261119.md)。MINDS v160 的独立 HF 路径覆盖 8,168 条，平均 Top‑1 概率 35.51%、Top‑1 macro‑F1 62.01%、准确率 60.97%；v158 服务路径仍未过精确一致性验收。MINDS 不参与训练准入。
2. 1.7B v161 的 TRAIN256 于 01:57 UTC 退出（exit 1、非 OOM）；[HF 日志](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.161/parity-operations/jev-1p7b-181-161-parity-collected/hf-process/process.log)显示旧版批次校验在推理前拒绝了本轮实际 Stage1 的 `microbatch=2`。没有提交可恢复的评分块。v162 仅在控制端验证；v163 另行封包，按冻结模型与检查点派生 2048 维 head 和 LoRA 配置，并绑定有界预取及明确的服务退出时限。[真实检查点隔离收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.163/endpoint-isolated-audit.json)只代表 CPU 检查通过。
3. v163 [接续驱动](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.163/reference-chain-driver/started.json)先并行准备 TRAIN256 和全量包。TRAIN256 的独立 HF 对照先用 1 卡，随后 8 个 TP1 实例做同一面板比对；数值审核通过才放行 8 卡全量打分。全量逐块提交，共 40,513 条。预计用时待 1.7B 首批及稳态数据确认，历史 0.6B 吞吐不作为本轮 ETA。MINDS 已退出；验收的 HF 对照、CPU 准备和独立审核阶段不再占用额外卡。复跑用 [接续配置](../../configs/portable/operations/stage1-train-reference-chain.json)与 `tools.reports.run_stage1_train_reference_chain --detach`，新版本绑定新输出、锁和准确容器句柄。
4. [视图后继](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.163/dataset-driver-started.json)等待全量独立审核后在 CPU 生成并复核 1.7B 专属难例与简单回放视图。正式 Stage2 在 181 使用 **6 卡 DP6 训练 + 2 个 TP1 推理实例**，global batch 384，两轮共 `2×ceil(N/384)` 步，`N` 取新冻结视图。用该视图和同一 reference 完成前缀、恢复、真实尾批和容量验收后再开正式训练。1.7B 端点评测单独绑定 step 0/318 和封存输入；外域数据不参与采样、选模或阈值调整。

2026-11-20 03:21 UTC：[v163 TRAIN256](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.163/parity-operations/jev-1p7b-181-163-parity-collected/evaluation/comparison.json)跑完同一批 HF 与 8 个 TP1 前向后，容器以 exit 1 退出，未 OOM。241/256 条的 logit 有差异，最大 logit 差 0.5625，最大概率差 0.06631，4 个候选跨 0.5，1 条 Top‑1 翻转。全量打分没有开始，无评分块可恢复；视图后继随之退出。

[v164 有限诊断](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.164-diagnostic/run/terminal.json)用冻结的 step 318 和同一输入缓存完成五组前向。关闭声学批处理后，64 条批量输出与 6 条单条输出分别逐值不变；3 条跟踪样本的声学投影也与 HF 一致。6 条单条推理仍有差异，3 条跟踪样本最早捕获到的数值差在语言模型第 2 层的注意力后归一化（层索引 1）。批量诊断改变了服务设备和批次上下文，未逐值复现 v163 的批量输出。该证据用于缩小范围；未通过 TRAIN256 精确验收，正式训练 loss 仍为 0/3。

[逐层对照](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.164-diagnostic/decoder-registered/trace-comparison.json)进一步定位到第 2 层的 Q 投影：两条输入均有 62 个元素不同，最大差 0.001953125，该层的输入和 K/V 投影仍逐值相等。[LoRA 运算顺序对照](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.164-diagnostic/actor-order/run/terminal.json)将服务 Q/V LoRA 按训练端的 BF16 中间舍入及缩放顺序执行，2 条单条与 64 条批量的 logits/prob 与 HF 逐值一致。v165 在受控服务配置中启用该实现，并在切换策略时重新绑定原始 A/B 张量。

v165 [TRAIN256 与全量接续](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-chain-driver/started.json)并行封包，保留 TRAIN256 精确审核门槛。固定镜像的 [扩展测试 47 项](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/extension-cpu/report.json)和 [DP6 打包测试 9 项](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/window-cpu/report.json)通过。[视图接续 v3](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/dataset-local-v3.json)绑定 v165 全量接续进程及其启动时钟，并使用包含 LoRA 修复的冻结审核源码；[训练接续 v3](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/formal-controller/config-v3.json)仅在完整视图独立审核后运行前缀、恢复和尾批验收，正式派发后按实际拓扑收集前 3 个已提交的训练损失。

2026-11-20 04:35 UTC：v165 [TRAIN256 独立审核通过](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/parity-operations/finalization/audit.json)。256 条音频的 logits 和概率逐值一致（最大差均为 0），无 Top‑1 或阈值翻转。驱动已进入 [40,513 条 TRAIN 打分的派发流程](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-operations/finalization/config.json)，本次快照尚无正式评分块或训练 loss。以全量容器的实际状态和首个提交块确认开始；不能把 TRAIN256 审核当作正式训练准入。

04:44 UTC：[训练接续 v4](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/formal-controller/config-v4.json)替换尚未派发的 v3 驱动，直接绑定固定镜像测试使用的扩展包。已对照收集的 `FILES.json` 和审核报告核对全部运行时源码的清单与四个扩展文件的实际字节。仍由 [视图接续 v3](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/dataset-local-v3.json)生成输入；本次变更不触碰正在准入的全量打分或其冻结包。

04:46:50 UTC：181 启动 [v165 全量评分容器](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-operations/jev-1p7b-181-165-reference-launch.json) `d046592a1b37`，实际挂载 0–7 八张卡，配置为每块最多 256 条、有界预取 4 个 I/O worker。首次现场检查容器运行、非 OOM，尚无评分块提交。正式训练与前 3 步 loss 仍待验收后确认。

04:51:53 UTC：[第一个全量评分块已提交](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-first-commit.json)，256 / 40,513 条。首块含容器及输入启动开销，不用于推算稳态吞吐；视图及正式训练尚未开始。

04:57 UTC，当前打包器含新增的目标源码哈希校验。固定镜像重新运行 [DP6 打包测试](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/window-cpu-v2/report.json)，9/9 通过，未挂载 NPU。[训练接续 v5](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/formal-controller/config-v5.json)绑定新收据与对应源码，接续同一视图生产任务。

05:01 UTC：[全量评分现场收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-observations/1795150886468601260-result.json)确认原容器运行且未 OOM，连续提交 51 块、13,056 / 40,513 条。按首块提交到本次观察的速度约 22 条/秒，评分预计在 05:22 UTC 左右完成；该估算不含退出后收集、独立审核、视图发布与训练准入。普通观察下次不早于 05:16 UTC，容器退出仍由原 `docker wait` 实时收集。

05:16 UTC：[第二份现场收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-observations/1795151788399060580-result.json)确认同一容器运行且未 OOM，连续提交 123 块、31,488 / 40,513 条。近 15 分钟约 20.4 条/秒，评分预计 05:24 UTC 左右完成。终态到达即由原驱动收集，普通远程观察下次不早于 05:31:28 UTC。

全量评分容器以 [exit 0](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-operations/jev-1p7b-181-165-reference-wait.json) 退出，NPU 租约已释放。[收集的生产结果](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-operations/jev-1p7b-181-165-reference-collected/reference-build/result.json)为 159 块、40,513 条不同输入，评分构建用时 33 分 20 秒。[传输收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/reference-operations/jev-1p7b-181-165-reference-collected.transport.json)确认已收集，尚不代表全量独立审核或训练准入。

05:43 UTC：原全量审核读取了错误层级的主机与资源锁配置（实际在 `deployment`）。[v166 重审](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.166-reaudit/started.json)仅处理已经收集的 v165 评分块，不重复派发 NPU 打分。[视图接续 v4](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/dataset-driver-v4.json)绑定该重审的进程身份与终态；[训练接续 v6](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/formal-controller/driver-v6.json)等待新视图审核，再按原门槛跑有限验收。两条接续均已启动等待；这不是正式训练的启动收据。

05:56 UTC：[v166 全量独立审核](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.166-reaudit/audit.json)通过，确认原容器收集的 159 块覆盖 40,513 条不同输入。Reference 清单 SHA-256 为 `5468306c406b81252e73c0752e08cfedd2e9aa2fbc29c603453ae977aeb9fa5a`。视图接续已开始 CPU 侧准备，训练接续等待视图审核；正式 loss 尚无。

11 月 20 日，用户为本轮 **1.7B Stage2** 临时改用 **69**。181 的 Stage2 自动派发监听已停止，保留原评分和数据发布收据。新[训练视图](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/dataset-successor-v4/dataset-audit.json)从 SLURP clean TRAIN 保留 9,967 条难例和 2,492 条简单回放，共 **12,459 条**。两轮 **66 步**，global batch 384，评测点 0、64、66。每轮尾批 171 条真实样本，213 个补位不进入分母。

[69 的 v177 训练准备收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.177-preflight/terminal.json)绑定前缀、尾批和未准入的正式包。0–5 卡训练，6、7 卡各运行一个 TP1 推理实例。[完整 reference](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.177-preflight/reference-transfer.json)已复制到 69 的本地盘并逐文件核对；共享基座、音频和特征继续只读使用。v177 使用与固定镜像收据一致的冻结源码，避开共享工作树中的后续改动。

06:50:16 UTC：v177 [两步前缀](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.177-preflight/operations/prefix/jev-1p7b-69-177-prefix-collected/supervision/actor.log:171)以 exit 1 退出并已收集。首次更新在 Qwen MLP 前向触发 NPU 显存不足，未提交更新。Docker 的 `OOMKilled=false` 只说明容器未被系统按内存超限终止。v178 的 microbatch 4 被现有批次契约拒绝，未派发。

07:02:52 UTC：[v179 两步前缀](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.179-preflight/operations/prefix/jev-1p7b-69-179-prefix-launch.json)在 69 启动。v179 保持 microbatch 8，将 actor 反向需要且达到配置大小的激活暂存到主机内存；[固定镜像 CPU 测试](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.179-preflight/cpu/report.json)通过 10 项，[封包审核](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.179-preflight/terminal.json)通过。只有独立审核前缀后才使用第一步检查点做新进程恢复，再验证真实尾批与下一轮首批。正式运行按同轮实测步时和临时卡时的截止时间核算。

07:09:14 UTC：v179 [首次更新前退出](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.179-preflight/operations/prefix/jev-1p7b-69-179-prefix-collected/supervision/actor.log:68)。初始策略一致性检查把运行配置变量复用于权重字典，随后删除了该变量，激活暂存路径未执行。v180 把两者分开，在[固定镜像 CPU 测试](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.180-preflight/cpu/report.json)再次通过 10 项，于 07:14:48 UTC 启动[新两步验收](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.180-preflight/operations/prefix/jev-1p7b-69-180-prefix-launch.json)。恢复及尾批后继以本次独立审核为依赖；本轮正式步骤仍未提交。

07:29:06 UTC：v180 [以 exit 1 终止](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.180-preflight/operations/prefix/jev-1p7b-69-180-prefix-inspect.json)。69 内核日志确认该容器触及 896 GiB 主机内存上限，rank 0 被终止，首次更新未提交。v182 放弃向主机暂存激活，仅在训练侧反向时重计算语言解码层，沿用冻结推理运行时、reference 与数据日程。[固定镜像 CPU 测试](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.182-preflight/cpu/report.json)通过 10 项，包含重计算与普通前向的输出及梯度对照；NPU 准入以新两步、恢复和尾批审核为准。

07:55:07 UTC：v182 两步容器以 [exit 0](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.182-preflight/operations/prefix/jev-1p7b-69-182-prefix-wait.json) 退出，两步均已提交。rank 0 实测每步总耗时 174.4、164.0 秒。独立审核发现新版本号被误用于旧 reference 收集路径；这不构成准入。v183 从原发布绑定引用读取审核路径，并在派发前核对 reference manifest，沿用 40,513 条已审核分数。

07:59:11 UTC：[v183 两步验收](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.183-preflight/operations/prefix/jev-1p7b-69-183-prefix-launch.json)在 69 启动。[执行配置](../../configs/experiments/stage2-view-execution-v0.0.183.json)通过 `activation_checkpointing.enabled` 和 `module_lists` 声明是否重计算及其目标，目标路径相对 actor 模型根；关闭时显式设为 `false` 并提供空列表。运行器校验目标存在、是非空模块列表、且同一层只出现一次。执行配置的哈希已纳入封包和恢复身份。主机、路径、物理设备与锁仍由本地宏绑定，批次与训练步数从冻结视图派生。[固定镜像 CPU 验证](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.183-preflight/cpu/report.json)通过 10 项。

08:10:01 UTC：v183 前缀容器正常退出，两次更新已收集。[前缀独立审核](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.183-preflight/prefix-driver/window/audit.json)通过，含同后端 objective replay。两步最慢 rank 用时分别为 175.0、165.8 秒。

08:21:08 UTC：用户要求释放 69。已停掉派发后继，恢复验收容器按准确句柄停止（exit 137，非 OOM）。[释放收据](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host69-v0.0.183-preflight/host-release-user-stop.json)记录资源锁可获取、0–7 号 NPU 无运行进程。恢复和尾批均未通过验收，正式训练未派发。保留前缀及其检查点、[1.7B Stage2 训练视图](/srv/multimodal-jev-xingchao/operations/context-1p7b-stage2-host181-v0.0.165/stage2-difficulty-view/MANIFEST.json)和全量 reference；已复核视图的全部文件与 69 正式包绑定的清单一致。后续接续须重新绑定目标主机及资源，使用已审核数据，无需重打分或重新采样。
