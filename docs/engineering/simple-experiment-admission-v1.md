# 简单任务实验基座：历史复盘、启动与最小验收

> 难例视图训练的当前工程候选入口及证据见 [Stage2 v0.0.125 启动交接](stage2-view-launch-v0.0.125.md)。旧全量后端保持原准入边界。

2026-11-19，开发候选，宿主 CPU 验证。目标是让合作团队用一个入口选择阶段、复用已经证明的结果，并能解释每一次拒绝。适用于 clean SLURP Stage1、Stage2 TRAIN 数据发布和原全量 Stage2 运行协议。本变更未切换现有运行包，也未为难例子集授予 NPU 训练资格。

## 故障证据与处置

| 现象与证据 | 真正失效的边界 | 今后在哪里拦 |
|---|---|---|
| v77 控制器只报 isolated terminal 不一致；actor 首错是 sealed source 外导入 tools.reports；服务随后误读 hello 为 close ack | 真实启动入口与测试入口不一致；收尾异常掩盖首错 | 保留既有固定镜像真实文件入口、RPC 退出协议测试；事故先对齐同一 container ID 下 actor、服务、supervision 三份证据，不依据最外层异常改镜像 |
| v87 固定镜像中两个测试失败；离线容器 standalone 会解析 hostname，而正式入口使用静态 loopback。历史尾批两个 rank 创建同一目录也曾失败 | 启动命令、网络、目录所有权都是运行协议 | 镜像/入口变更时覆盖实际 torchrun、network=none、rank 并发、异常退出；由唯一 owner 建立运行根，重复启动同一根直接拒绝 |
| v80/v81 审核期间工作树变动；v123 prefix/replay 后，tail 报 candidate differs from CPU snapshot | 测试、打包、尾批补拷有多个源码来源。旧 CPU 支持清单还缺 supervisor | 本轮让 finite 与 tail 从同一 CPU 快照按清单取文件，完整比对成员、符号链接和内容，再核对复制的字节；缺件先于输入缓存等待发现。旧收据仍只代表旧包 |
| v121 DP4 micro16：rank 1 申请 356 MiB 时 NPU OOM | 真实设备容量，不是训练目标或 CPU 脚本问题 | 新统一入口锁定本轮已准入的 global256 / micro8；改几何、微批或精度，在新包做有限设备验收 |
| v92 曾在租约前耗时约 757 秒；另有历史采样表明 JSON 重建、嵌套来源审核耗时大 | 同一依赖被多层审核，耗时不等于物理盘吞吐 | 沿用既有 CompletedStageReceipts 和锁内内容校验；本轮去掉 Stage2 detach 父进程的完整证据核验。数据 release 一次调用完成生产和绑定 |
| Stage1 两个入口曾把“已收集，待独立审核”作为 exit0 终态；旧 Stage2 外层直接写 634 | 子进程退出、工件正确、模型有效是三种不同结论 | 正式 Stage1 仅在原审核器核对完整更新、样本曝光、同一容器后写成功；Stage2 从审核结果读更新数；模型指标由独立评测提供 |
| 175/181 历史 Ceph fsync 崩溃 | 主机存储边界，不能靠修改 reward 或全局关闭 fsync 解决 | 沿用实际挂载准入；本地持久盘写热状态，Ceph 共享输入只读 |

证据来源：[启动故障、v87、v92 与已有修复](stage2-startup-root-causes-20261118.md)、[采样口径与时延](stage2-admission-latency-20261118.md)、[v123 事件与后续独立准入](stage1-margin-stage2-difficulty-v1.md)、[主机存储](runtime-storage-policy.md)、[历史按时间归档](agents-history-v1.md)。本机历史原始证据包括 operations 下的 `context-stage1-start-v0.0.123/terminal.json`、`context-stage2-parallel-v0.0.77/terminal.json` 和 `context-stage1-local64-micro16-v0.0.121/*collected/supervision/actor.log`。这些是历史故障证据，不描述当前 NPU 状态。

## 启动契约

唯一的**新实验前端**为 [run_simple_experiment](../../tools/reports/run_simple_experiment.py)，共享模板为 [simple-experiment.json](../../configs/portable/operations/simple-experiment.json)。使用本地宏，不能在模板写机器路径。前端只选择已注册的后端，不另写训练循环，不按配置执行任意 Python 函数，不在失败后自动重启。

| EXPERIMENT_STAGE | 复用的后端模板 | 该动作完成的含义 |
|---|---|---|
| stage1 | context-stage1-start-pipeline.json | DP4，global256 / micro8。只在完整正式训练独立审核后报告训练工件通过；尚不表示效果通过 |
| stage2_dataset | stage2-dataset.json | 本轮终点全 TRAIN 固定 reference → 难例+简单回放 → 日程与新配置；始终 training_dispatch_allowed=false |
| stage2_full | composed-stage2.json | 仅供已准入的原全量配置（DP2 + 两个 TP1）；拒绝难例视图。本轮新实验不要以此替代子集训练 |

1. 发布方用既有 `source_snapshot.freeze` 生成独立快照和 FILES.json（应包含本阶段用到的 src、tools、配置模板、插件和测试支持文件）。在本地宏使用其返回值作为 `EXPERIMENT_SOURCE_SNAPSHOT`；源码验证依赖实际文件全集，不用工作区名或 Git 分支代替。
2. 同一本地宏文件填阶段的既有绑定；Stage1 还在 `STAGE1_START_PIPELINE` 对象里填 `design_sha256`（设计 JSON 文件的 SHA256）。首次读取与正式使用均核验，运行中封存该设计。数据阶段给 `STAGE2_DATASET_RECEIPTS.release` 新路径；`STAGE2_DATASET_SHA256` 在首次 release 可为空字符串。文件、目录、凭据均不写入共享模板。
3. 用既有 `load_config(backend_template, local)[1]['resolved_semantic_sha256']` 取得解析后后端身份，填 `EXPERIMENT_BACKEND_SHA256`；再填 `EXPERIMENT_STAGE`、`EXPERIMENT_IO_WORKERS`、`EXPERIMENT_DRIVER_ROOT`（本地持久盘、全新且与阶段输出不重叠）。每个阶段是一个显式请求，不自动衔接未准入阶段。
4. 从冻结源码根启动，解释器必须有阶段所需依赖；禁写 pyc。以下 shell 变量由本地发布环境提供，不能指向正在运行的工作树：

```bash
cd "$EXPERIMENT_SOURCE_ROOT"
PYTHONPATH=src:.:tools/jev "$EXPERIMENT_PYTHON" -B -m tools.reports.run_simple_experiment \
  --config configs/portable/operations/simple-experiment.json --local "$LOCAL_MACROS" --action plan
# 正式长任务只提交一次
PYTHONPATH=src:.:tools/jev "$EXPERIMENT_PYTHON" -B -m tools.reports.run_simple_experiment \
  --config configs/portable/operations/simple-experiment.json --local "$LOCAL_MACROS" --action detach
```

plan 不创建运行目录、不扫描完整训练集、不占卡，只报告请求和阶段配置身份。detach 由子进程履行完整核验；读取 driver 的 started.json、terminal.json 及阶段输出。未出现成功终态即未完成；监测沿用 900 秒。前台 run 适于短数据动作。驱动目录和阶段结果用于不同目的，不能复用已有的驱动目录发起第二次实验。

## 哪些验收是必要的

| 发生变化 | 需要的证据 | 不再重复做 |
|---|---|---|
| 配置、设计、来源或执行代码 | 解析后的配置身份，精确快照、输入来源及隔离；占卡时检查实际租约、资源与封包身份 | 同一 detached Stage2 请求，父进程不再先跑完整 CPU/来源哈希 |
| TRAIN 终点、抽样种子、阈值或预算 | 来源审核 + 全样本确定性采样 + 对发布内容做一次独立复算，再绑定该内容的 manifest 与完整共享 reference | 标准路径不手工搬 hash，不串行跑 prepare→audit→bind。已有内容仍校验；外部独立 audit/recovery 保留 |
| 镜像、执行入口、LoRA 装载、部署拓扑、精度、优化器恢复或尾批语义 | 对变化包做固定镜像真实入口、短程数值、恢复、真实尾批。Stage2 按适用范围验证 actor/rollout/reference 的 BF16 策略及缓存身份；已通过且身份完全一致的阶段按现有收据复用 | 不新增独立效率跑分；不把宿主 CPU 通过当作 NPU 准入 |
| 正式训练结束 | 单次收集后独立核对实际容器、工件、更新数/真实曝光、登记的可重评分策略点 | 不再人工补写训练完成状态；正式审核通过不代替 selection/test/FSC/MINDS 的评测 |
| 日常观测 | 固定脚本读计时、首错及主/次进程日志 | 不触发模型审核或按分钟重新分析 |

**失败纪律**：请求失败即停止并保留原始证据，不改原包、不自动降 microbatch、不重新拉起正式训练。先定位首个失效边界，添加能重现该类故障的负例；修复后用新版本、新输出根，仅对已失效的证据重新验收。GPU/NPU OOM、主机与存储故障仍可能发生；本规约禁止以正式任务反复试错，不承诺硬件不会失败。

**尚未发布的边界**：新入口和数据消费路径是 CPU 开发候选；继承的 Stage1 trainer/actor 内还有源码生成，Stage2 旧包仍有组合适配债务。修改这些实现需要新包及训推数值回归，不能在已准入包热改。难例视图需在正式 trainer **及其独立 auditor** 共同接入动态更新数、真实曝光、恢复与尾批后，才可做设备有限验收并发布；不能把外层统计改为动态数字就放行。

## 本轮可复核的收益

此前文档建议顺序执行 prepare / audit / bind：在一次新发布上，producer 审核调用 3 次；采样/日程构造 **5 次**（prepare 的创建和复算各一次，audit 一次，旧 bind 的外层 audit 和 TrainingDataset.audit 各一次）。推荐的 release 路径为 **1 次来源审核、2 次构造**（生产和独立复算）；已有视图复用时仍复算一次。这是组件调用数，不是宣称真实全量时延或设备吞吐提高几倍。测试覆盖错误 producer、视图篡改后重签、源配置改变、快照缺件/额外文件、加载前后内容变化和假成功终态。

宿主测试和 JUnit 留在本机 `/srv/multimodal-jev-xingchao/operations/experiment-base-review-vzVpBN/`。三份 JUnit 分别为 76、5、40 项，通过用例按 classname/name 去重为 106 项；后两份针对设计身份和监测接口的后续改动。未做全 TRAIN 的本轮时延对照，未重新派发固定镜像或 NPU 任务。

## r64 八卡实验的配置与接续

本轮 Stage1 在八卡上做区分性学习，Stage2 使用六卡训练和两个单卡推理实例。团队调整 rank、alpha、是否复用 Stage1 adapter、总卡数、推理实例数、每卡 batch、microbatch 和 warmup 时，编辑[两阶段配方](../../configs/experiments/simple-two-stage-recipe.json)并生成新实验目录。每阶段的 global batch 由该阶段的训练 DP 数推导。入口读取[已支持的 Stage1 档位](../../configs/experiments/stage1-simple-entry-profiles-v1.json)，改档位后仍执行该配置的短程数值、恢复与尾批验收。冻结 Stage1 控制器时把这份档位 JSON 一起纳入源清单。

```bash
PYTHONPATH=src:. python3 -Bm tools.reports.compile_simple_experiment \
  --recipe "$RECIPE" --output "$NEW_EXPERIMENT_DIR"
```

新实验从生成的 experiment.json、two-stage.json 和 dataset.json 制作 Stage1 布局计划与训练包。Stage1 的模型、语料、设备编号和机器绑定沿用该轮 pipeline 的本地宏。Stage2 的接续模板描述并发、预取、等待预算和输出目录结构，直接从本轮输入读取训练设计、真实设备、缓存身份、镜像及审核证据。封包时校验的是本轮 Stage1 的实际设计，旧 rank 或不同 batch 的凭据无法用于新配方。[接续模板](../../configs/portable/operations/simple-stage2-handoff-v1.json)

将[本地宏示例](../../configs/portable/operations/simple-stage2-handoff-local.example.json)复制到本地宏目录，在 macros 中填入 ATTEMPT_ROOT（本轮本地持久盘目录）与 ATTEMPT_TAG（本轮远端唯一标识）。如各工件不在示例约定的位置，可在同一文件改路径。SIMPLE_RECIPE 指向启动 Stage1 使用的配方。准备好本轮 finalizer 句柄、runtime、全 TRAIN 日程、无标签对齐面板及镜像审核收据后即可生成接续请求。以下命令从对应源快照根运行，BOUND_LOCAL 指向有足够空间的本地持久盘新文件。

`STAGE2_OFFLINE_SCORING_DEVICES` 指定 TRAIN 数值对齐和全量 reference 打分共同使用的卡。独占八卡时可填满八张，每张一个 TP1 实例；四卡实验填本轮实际可用的四张。离线打分的宏批和预取按实例数 × 单实例 batch 派生，对齐面板应至少覆盖一个完整宏批。正式 GRPO 的设备划分和 global batch 仍由两阶段配方决定。缩放只在生成新接续请求时生效；已提交的 reference 宏批绑定了当时的实例数和宏批大小。

```bash
PYTHONPATH=src:.:tools/jev python3 -Bm tools.reports.bind_simple_handoff \
  --config configs/portable/operations/simple-stage2-handoff-v1.json \
  --local "$HANDOFF_LOCAL" --output "$BOUND_LOCAL"
PYTHONPATH=src:.:tools/jev python3 -Bm tools.reports.context_stage2_endpoint_successor \
  --config configs/portable/operations/context-stage2-endpoint-successor.json \
  --local "$BOUND_LOCAL" --action validate
```

bind 和 validate 都只生成或检查配置。新实验派发一次时，上述第二条命令使用 `--action detach`。接续器等待独立终态后，在设备上比对同一策略的 actor 与推理服务，再用 Stage1 全 TRAIN 固定输出生成难例和简单回放视图。正式 GRPO 预算取该视图实际人口。重复生成相同请求可复用原文件；新内容需要新文件。本轮已有接续句柄，工程复现可单独生成配置与冻结请求对照。

本轮准备 Stage2 原生 selection/heldout 评测包时，在 `STAGE2_VIEW_SELECTION_PREPARE` 中增加 `training_request: {"path": "...", "sha256": "..."}`，指向本轮实际派发的 Stage2 本地宏及其内容哈希。历史运行配置保持原样。这个字段让封包器验证同一训练镜像和基座，将封存的 forward 源码按清单装入新的评测快照，并让各评测实例使用与 actor 相同的 BF16 Q/V LoRA 计算方式。每个新策略点在派发前从联合快照认证决策 head、marker 和 LoRA，并重新推导 PEFT 导出与各个矩阵精确比对。正式效果仍需按数据集验证同一策略的 HF/服务概率一致。[有限 r64 策略点的 CPU 实测](/srv/multimodal-jev-xingchao/operations/next-simple-r64-dp8-v1/evaluation-baseline/forward-binding.json)核验了 56 个 rank64 目标和 459 个 forward 源文件；不是正式 Stage2 末点或 NPU 评测。
