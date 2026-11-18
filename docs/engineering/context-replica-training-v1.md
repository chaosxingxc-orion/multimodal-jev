# 配置化副本训练（任意 DP/TP 拓扑）

状态（2026-11-17）：**DP6 双奖励两更新 NPU 验收已通过并完成独立审核（v6，见第 6、7 节）；
全量 212 更新正式训练尚未派发，新进程恢复（放行条件 5）尚未验证**。本文描述
“拓扑是数据、代码是同一套”的副本训练链：DP2/DP3/DP4/DP6 等任意宽度只改解析后的配置，不新增
per-topology 脚本。放行前不得派发正式训练，也不得据此声称任何效率或效果结论。

## 1. 原则

- 拓扑、batch、推理实例数量与服务卡号、rollout 宽度全部来自解析后的配置（宏 + 可移植模板），
  不在源码里写死设备号、DP 宽度或服务数量。
- 复用既有冻结模块：`context_topology_batch_plan.derive`、
  `context_replica_schedule`、`context_replica_trainer_candidate`、`context_replica_macro_client`、
  `MacroRolloutPool`、`context_replica_runtime`、`context_process_supervisor`。
- DP4 冻结包、spec、调度与 pins 保持不变；新拓扑只生成独立版本化候选包。

## 2. 链与入口

1. 批量计划：`derive(batch_config)` → 9 键配置（`training_devices`、TP、PP、每副本 64、
   microbatch、推理实例数、推理 batch queries、source queries、epochs、efficiency steps），
   产出 DP 宽度、全局 batch、更新数、尾部与 `macro_queries`。
2. 调度：`tools/reports/prepare_context_replica_schedule.py` 从已认证的 DP4 父调度重排为声明宽度，
   保留真实顺序、去掉旧 padding、补零权重尾部，产出 `jev-replica-source-schedule-v1`。
3. 打包：`tools/reports/prepare_context_replica_training_candidate.py` 校验父包、调度、策略与拓扑，
   复制并变换 runtime 源码，生成 `config.json`、每个推理服务一份配置、`run.py`、
   `RUNTIME-SOURCES.json`、`PREPARED.json`（status `replica_integrated_prepared_only`，
   `training_dispatch_allowed=false`）。模板
   `configs/portable/operations/context-replica-training-prepare.json`。
4. 启动/校验：`tools/reports/context_replica_training_operations.py`，未加 `--launch` 时只做本地
   校验并打印 docker argv；`--launch` 需显式 `finite_training_dispatch_allowed=true`、持租约、
   仅挂载声明设备并核验这些 NPU 空闲、核验镜像/外部 pins/本地容量。模板
   `configs/portable/operations/context-replica-training-operations.json`。
5. 收集：`tools/jev/collect_context_replica_training.py`，传输集合来自启动记录里的包 layout
   （配置文件、全部服务配置、run/supervision/服务目录、reference-cache），并校验每个受绑定文件
   的哈希。传输成功仍不等于训练成功，需独立审核。

## 3. DP6 实例（当前候选）

| 项 | 值 |
| --- | --- |
| 训练卡 | 0–5，`torchrun --nproc-per-node=6` |
| 推理卡 | 6、7，两个 TP1 服务（`ASCEND_RT_VISIBLE_DEVICES=6`/`7`） |
| 全局 batch | 384（每副本 64） |
| 更新 | 每 epoch 106，总 212 |
| 真实曝光 | 81,026（2 epoch） |
| 尾部 | 每 epoch 193 真实 / 191 零权重，loss 分母 193 |
| rollout | 每命令 `rollout_queries=32`，pool `max_queries=384` |
| 调度 MANIFEST | `d8175b03…`（212 更新） |

`macro_queries == global_batch`；客户端按 `rollout_queries` 分片调用，因此 384 行更新不需要另建
192 行分块层，池上限放开到 4096 即可。

## 4. 同容器进程隔离

一个容器挂载声明的训练卡与推理卡。每个推理服务在自己的配置里声明
`physical_devices`、`process_device_scope`（`same_container_runtime_visibility_v1`，
`mounted_physical` 为容器挂载集合，`visible_ordinal` 为目标卡在集合内的序号）与
`device_scope_module`（`context_configurable_device_scope`，按文件哈希固定）。该模块在
导入 torch/torch_npu/vLLM 之前校验静态配置，之后校验 `torch.npu.device_count()==1` 与
可见序数到物理卡的映射。只挂一张卡时不需要 scope 键；DP4 冻结的
`context_shared_device_scope` 与哈希 pin 原样保留。

生命周期由 `supervise_services` 负责：启动 1–8 个声明服务，逐个等待 `ready.json` 与 Unix socket，
再启动 actor；任一服务退出而 actor 仍存活即失败；结束时按 close 契约回收并写 fsync 的
`supervision/terminal.json`。

## 5. 放行条件（未满足前不得派发正式训练）

1. 固定镜像内导入/检查通过。
2. 真实音频 4 次推理 pass 跨每个声明服务通过。
3. NPU 两次更新，全局归约与尾部零权重分母正确。
4. 每个服务都有非零 adapter/head 激活。
5. 从提交的 checkpoint 做新进程恢复。
6. 训练卡与推理卡隔离（CANN 可见性、设备占用、无交叉使用）。

短验收形态：4 推理 pass / 2 训练更新；通过后才讨论正式训练预算。

## 6. NPU 验收尝试记录（失败保留，不原位修改）

| 尝试 | 远端根/镜像标签 | 失败现象 | 根因 | 修复与证据 |
| --- | --- | --- | --- | --- |
| v1 | `context-replica-training-dp6` | vLLM 初始化 `ValueError: complete expanded integer prompt IDs required` | `_dummy_pooler_run_task` 的矩形 int32 dummy 行宽与最后一个调度的 token 余数不一致，profiling 分支按行宽取长度 | `vllm_context_pooler.py` 按调度长度重建定长合成 prompt；回归测试与固定镜像 CPU 校验 `docs/engineering/context-pooler-profiling-fixed-image-cpu-v1.json`（3/3） |
| v2 | `context-replica-training-dp6-v2` | actor 各 rank `FileExistsError: …/run` | torchrun 每个 rank 执行同一入口，共享输出根用 `exist_ok=False` | `context_replica_trainer_candidate.prepare_output` 幂等创建，per-rank 目录仍 `exist_ok=False` |
| v3 | `context-replica-training-dp6-v3` | `<replica-trainer>` 第 40 行 `ModuleNotFoundError: multimodal_jev.slurp_portable_release` | 训练器变换把 `slurp_official_training` 改成 `slurp_portable_release`，但打包 `SOURCES` 未包含该模块，父包也没有 | `SOURCES` 增加 `slurp_portable_release.py`；打包时新增“生成训练器引用的 `multimodal_jev` 模块必须在装配后的 source 内”门禁；固定镜像 CPU 校验 `docs/engineering/context-replica-packaging-fixed-image-cpu-v6.json`（23/23） |
| v4 | `context-replica-training-dp6-v4` | actor 各 rank 在 `prepare_full_schedule` 抛 `ValueError: FileNotFoundError(2, …)` | 冻结 release 的 `publication-config.json` 在**发布主机的绝对路径**（`/home/xingchao/workspace/multimodal-jev/...`）上校验 9 个证据文件；副本容器只挂了候选包与共享根，证据路径不存在 | 操作配置新增数据化的 `release_evidence: {root, destination}`：按 release 自身 pins 校验 ceph 证据镜像逐字节一致后只读挂载到记录的绝对路径，并对运行根做不相交校验；固定镜像内诊断确认 `load` 40513 行、调度、处理器身份与音频镜像 pin 全通过 |
| v5 | `context-replica-training-dp6-v5` | 容器 exit 1；launcher 随后以 `actual isolated terminal runtime differs` 拒绝（退出码非 0 的运行不能算成功） | actor 各 rank `ModuleNotFoundError: No module named 'multimodal_jev.local_config'`（打包的 `context_replica_macro_client` 首次引用 `local_config`，父包不提供） | 打包闭包改为对装配后的 source 做完整 `multimodal_jev` / `tools.*` 导入闭包扫描并自动补拷（`required_modules` / `required_tools`），缺件即拒绝封包；新增负向测试；固定镜像 CPU 校验 `docs/engineering/context-replica-packaging-fixed-image-cpu-v6.json`（23/23） |
| v6 | `context-replica-training-dp6-v6` | —（通过） | — | 两更新 NPU 验收 + 独立审核通过，证据见第 7 节；版本语义下记为 **`v0.0.rc0`**（工程基座锁定点） |
| v7 | `context-replica-training-dp6-v7` | 正式训练启动即 exit 1：rank0 `KeyError: 'checkpoint_interval'`（`<formal-replica-trainer>` 第 78 行） | 正式窗口已删掉 checkpoint 间隔字段，但生成的训练器仍读取旧字段 | `context_replica_formal_candidate.py` 生成期把该守卫替换为 `'checkpoint_interval' in c`；回归断言加在 `tests/test_context_replica_formal_candidate.py`；0 次更新，无训练预算消耗 |
| v8 | `context-replica-training-dp6-v8` | 正式训练启动即 exit 1：`ValueError: continuous candidate is engineering-only, not released` | 生成训练器保留“仅工程前缀”守卫，正式契约被它拒绝 | 同一模块把该守卫替换为正式契约守卫（`engineering_prefix is not False or training_dispatch_allowed is not True` 才拒绝）；固定镜像 CPU 校验 53/53（`docs/engineering/context-replica-packaging-fixed-image-cpu-v8.json`）；0 次更新 |
| v9 | `context-replica-training-dp6-v9` | —（2026-11-17 06:27Z 派发，运行中） | — | 前两条根因同时修复后的正式候选（开发态 `v0.0.3`）；固定镜像 CPU 校验 53/53（`docs/engineering/context-replica-packaging-fixed-image-cpu-v9.json`）；正式结果是 212 更新完整曲线 + 独立审核后才有的结论 |

每次修复都走新的版本化包（新远端根、新镜像标签、新 attempt id），不修改已运行的包。

### 6.1 版本语义与历史映射（2026-11-17 用户指定）

运行包分**开发态**与**锁定基线**：

| 语义 | 命名 | 含义 |
| --- | --- | --- |
| 开发态候选 | `v0.0.N` | 每次根因修复一个包，过即用、不过即弃；不承诺长期保留 |
| 锁定基线 | `v0.0.rcN` | 通过验收（工程验收或正式实验独立审核）才发布；发布后删除被它取代的全部开发态小版本，只在本文档保留结论与证据指针 |

映射（历史包不重命名，避免破坏已运行的重哈希与证据链）：v1–v5 = 开发态迭代；**v6 = `v0.0.rc0`**；v7 = `v0.0.1`；v8 = `v0.0.2`；v9 = `v0.0.3`（当前）。v9 若通过正式 212 更新独立审核，则这一整套（工程基座 + 正式结果）记为 **`v0.0.rc1`**，随之按清单删除 v7、v8 的开发态包。

v1–v5 的开发态残留已于 2026-11-17 在 rc0 发布后按清单删除（本地 113.6 MB：5 个包、5 组 acceptance 采集与 tar、10 个宏、5 个 CPU 校验目录；远端 ~111 MB：5 个运行根 + 1 个 `-v5-package` 残桩；10 个已退出容器）。清单与执行记录在 `/srv/multimodal-jev-xingchao/operations/cleanup-20261117-pre-rc0-{inventory,executed}.json`；删除的是原始采集，结论保留在本节表格。

## 7. v6 DP6 双奖励两更新验收证据（2026-11-17）

- 包：`context-replica-training-dp6-v6-package`（414 文件，`PREPARED.json` `83e02f71…`）。
- 运行：容器 `a2fbe710…` exit 0；`supervision/terminal.json` `passed=true`、actor 0、两个服务 0、
  无强制清理；总耗时约 9 分钟（13:20 启动，13:29 收集完成）。
- 收集：`.local/decision-objective-research/context-replica-dp6-acceptance-v6-collected`（含传输
  manifest 与 372 MB tar），绑定文件哈希全部一致。
- 独立审核：`tools/reports/audit_context_replica_training.py`（配置驱动，无 per-topology 常量），
  结果 `context-replica-dp6-acceptance-v6-audit.json`，`status=finite_replica_training_audited`：
  - 两更新各 384 真实查询（无 padding），与认证 schedule 的 `real_queries` 一致；尾部 193 真实 /
    191 零权重与总曝光 81026 由 schedule 重建校验。
  - 4 个完整 macro（2 参考 + 2 行为），每个 384 行、按 32 行 rollout 分派到服务 a/b，journal 成员
    与计时逐条绑定。
  - 每步 LoRA 112 个张量、head 12–13 个张量发生变化；冻结 marker 不变。
  - 每步 6 个 rank 的 FP32 master/Adam 分片逐一还原 BF16 导出（`numel=6878209`），trainable scope
    与提交的 checkpoint 一致。
  - 两个服务各有 112 个 adapter/head 张量激活回执；`ready.json` 物理卡为 6、7，训练卡 0–5，容器
    设备挂载与 argv 全部复现一致。
  - 单步最慢 rank 耗时 152.1 s / 158.1 s（工程参考，不是效率结论）。
- 仍未覆盖：新进程恢复（放行条件 5）、全量 212 更新曲线、任何效果/迁移结论。
