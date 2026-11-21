# 组件化工程基座 v1

2026-11-18 用户要求：一套高内聚、低耦合、组件化的工程基座，完整支撑训练与评测；
Qwen3-ASR 0.6B 与 1.7B 走同一套配置和算法；数据集、算法、评测方式是迭代面，其余保持稳定；
前处理与审核必须并行化。本文记录这轮 P0–P5 的组件边界、验收证据和未验证边界。

**结论边界：全部六个阶段都是 CPU 确定性验收，`model_calls=0`、`npu_executed=false`。**
没有任何一条结论证明 NPU 训练、正式效果或迁移收益；被 `not_exercised` 记录的步骤就是
本轮没有执行的部分。冻结运行包（v0.0.49 及其依赖）未做原位修改，改进全部是新文件、
新本地宏、新运行根。

## 1. 组件边界

五个组件平铺在 `src/multimodal_jev/`，任务侧实现通过注册表接入，训练器与控制器只做组合：

| 组件 | 文件 | 公开面 | 职责 |
| --- | --- | --- | --- |
| 模型几何 | `context_model_profile.py` | `ModelProfile`、`from_hf`、`lora_targets`、`decision_head`、`build_head`、`identity`、`validate_against_config`、`record` | 从 HF `config.json` 派生 hidden/层数/qkv/音频维度、56 条 q/v LoRA 目标、决策头尺寸 |
| 并行执行 | `context_execution.py` | `width`、`widths`、`Stats`、`map_io`、`map_cpu`、`singleflight`、`content_key` | 唯一并行层：线程池做 I/O 与哈希，spawn 进程池做 Python 密集循环 |
| 组件注册表 | `context_component_registry.py` | `COMPONENTS`、`KINDS`、`names`、`entry`、`resolve`、`catalog`、`validate` | 字面量注册 release / objective / evaluator / topology / controller，按名解析 |
| 目标函数注册表 | `context_objective_registry.py` | `OBJECTIVES`、`names`、`entry`、`resolve`、`available`、`catalog` | 字面量注册 `candidate_infonce`、`dual_credit_grpo_floor_clipped_v1`、`dual_credit_rloo_clipped_v1`、`rloo_v1` |
| 发布句柄 | `context_release_handle.py` | `ReleaseHandle`、`reader_digests`、`READERS` | release 只全量认证一次，按内容键复用；读取器源码摘要参与键 |
| 评测契约 | `context_evaluation_contract.py` | `METRIC_FIELDS`、`DERIVED_FIELDS`、`SELECTION_FIELDS`、`IDENTITY_FIELDS`、`validate_row`、`from_result`、`ranked_key`、`delta` | 唯一指标字段元组与评分行契约 |

服务与产物层（`vllm_context_model`、`context_policy_artifacts`、`context_policy_snapshot`、
`context_stage_lora`）本轮零改动：它们本来就从 contract/branch 派生维度，因此 hidden 2048
不需要新代码，只需要新的几何输入。

**训练器与控制器组合而不是改写源码。** 新的 `tools/reports/context_composed_pipeline.py`
不含任何 `inspect.getsource` / `str.replace` 文本拼接，唯一的注入点是把 body 绑定到一个
`FunctionType`（第 74 行），用于构造阶段性的依赖注入。旧控制器路径上的 98 处改写
（104 个文件、102 处注入）没有删除，作为测量债务原样记在 P4 报告里。

## 2. 注册表语义

- 注册条目是**字面量元组**，不做插件式动态发现。每个条目固定
  `kind / name / module / attribute / status / requires / summary`。
- `resolve(kind, name)` 只接受已 `admitted` 的条目；未知名字、重复名字、`design_only`
  状态一律报错。配置里的算法名、评测名在 `validate` 期就必须解析成功，不允许运行时拼 import。
- `catalog()` 返回解析后的真实对象与模块文件 `sha256`，因此“按名解析”与“直接 import”
  可以得到同一对象；测试断言两者相等，保证零行为漂移。
- `context_evaluation_contract` 把散落的指标字段收敛为一个元组，writer 输出键集合必须
  **恰好等于**该元组（多一个、少一个都失败），不是子集检查。

## 3. 并行宽度语义

两类工作、两种池，不混用；宽度从配置读，不写死：

| 层 | 适用 | 机制 | 关键约束 |
| --- | --- | --- | --- |
| `map_io` | 读文件、算哈希、解析 JSON | 有界线程池 | 按输入顺序返回；首个异常取消未开始任务 |
| `map_cpu` | target 编译、batch 重算等 Python 密集循环 | spawn 进程池 | 每子进程 `OMP_NUM_THREADS=1` 且 `torch.set_num_threads(1)`；按块派发 |
| `singleflight` | 内容寻址复用 | 复用 `AuditSingleFlight` | 不复制实现，所有调用方共享同一淘汰策略 |

- `widths(config)` 要求配置**恰好**含 `io_workers` 与 `cpu_workers`，范围 1–32。
- `map_cpu` 的 `min_parallel` 是配置值，不是隐藏启发式：小于阈值的面板走串行。
  已实测 8-batch 面板并行更慢（3.759 s 串行 / 4.630 s 并行），所以默认阈值 16 高于它。
- `chunk` 控制每次 IPC 派发的连续条目数：1.75 ms/条的负载上，逐条派发只有串行的 0.36×，
  分块后才超过串行。
- `Stats` 记录墙钟、逐派发单元秒数、峰值在飞与加速比；`unit` 明确区分 item 与 chunk，
  分块运行不会被误读成逐条耗时。
- 并行只换执行方式，不换输入：调用方仍然传**完整**条目列表，不抽样。

**实测（配对单次测量，不是重复试验基准）：** `cpu_workers=8, window=16, chunk=16,
min_parallel=16` 在 512 个单元上墙钟 0.2587 s、任务秒合计 0.9136 s、峰值在飞 16、
加速 3.4756×；release 全量重认证 5.982 s 降到复用 0.5239 s / 0.5216 s，省掉 91.2% 的重认证，
三次输出 `sha256=bdb93b78…` 逐字节相同。

## 4. 0.6B 与 1.7B 的同路径

两个尺寸的差别只有 `hidden_size 1024→2048`、`intermediate_size 3072→6144`、
音频 `d_model 896→1024`、音频编码层 `18→24`、音频输出 `1024→2048`。
**文本层数都是 28，q/v 目标集合结构完全相同。**

- 大模型配置 `configs/experiments/context-model-profile-1.7b-v1.json` 与 0.6B 设计
  `context-two-stage-v0.0.11.json` 的差异**恰好**是 `model` 与 `model_profile` 两个路径；
  其余字段（含两阶段各 56 条字面量 target、`reuse_stage1_adapter: true`）逐项相同。
  该配置 `training_dispatch_allowed: false`，本轮不派发训练。
- 共享 LoRA 目标 56 条，两个模型 `sha256=745050d297469bcf…` 相同。
- 决策头由几何派生：0.6B `(1024, 8, 1024, 0.1)`，1.7B `(2048, 8, 2048, 0.1)`，
  1.7B 头 25,192,449 参数。
- 容量：0.6B 611 张量 / 782,426,112 参数；1.7B 707 张量 / 2,038,052,480 参数，
  比例 2.6048。两者都是 BF16，56 条目标全部存在、0 条宽度错误。注意力几何不对称：
  q 宽 2048、v 宽 1024，导出 payload 必须是 q `B[2048,4]`、v `B[1024,4]`。
- 可训练量：LoRA 917,504 + 头 25,192,449 = 26,109,953 参数（BF16 52,219,906 字节）。
- 策略快照往返：128 张量导出 → 129 还原（含 marker 行），`byte_identical: true`，
  头宽 2048，56 目标，`policy_id=4b738af3d008…`。
- 设计一致性检查：小模型对大手设计被**拒绝**（声明 hidden 与模型 config 不符），
  大模型对小设计通过——说明几何校验真的在比较，不是两边都放行。

## 5. 验收证据

六阶段严格串行，每阶段一个独立 `v0.0.N` 开发态候选，通过后才是下一个。

| 阶段 | 运行根 | 状态 | 关键结论 |
| --- | --- | --- | --- |
| P0 基线测量 | `operations/context-engineering-baseline-v0.0.52` | `measured` | release 全量加载 5.231 s / 40,513 行 / 60 候选；8 文件 49,465,358 字节在 io 1/2/4/8 下 0.0544/0.0278/0.0260/0.0275 s |
| P1 模型几何 | `operations/context-model-geometry-v0.0.52` | `audited` | 0.6B 派生目标与冻结 56 条逐项一致；1.7B 同样 56 条、头 `(2048,8,2048)`；18 处硬编码尺寸站点中 17 处已派生、1 处仍写死（测量债务） |
| P2 并行执行 | `operations/context-parallel-execution-v0.0.53-r3` | `accepted` | 7 项检查全过；宽度网格全部与串行等价；采纳 8/16/16/16；release 复用省 91.2% 重认证 |
| P3 组件契约 | `operations/context-component-smoke-v0.0.57` | `accepted` | 五个组件 admitted、按名解析可执行；合成 release → objective → evaluator 单进程完成；`deterministic: true`；`source_rewriting_sites: []`；selection 越权调用被拒绝 |
| P4 组合式控制器 | `operations/context-composed-differential-v0.0.67` | `accepted` | 六个阶段准入结果一致且委托；四个 seam 与重建同代码；selection 三检查点 [0,8,318] 一致；两个 producer 包生命周期间数据一致；两种派发形式都被拒绝（`dispatch_authorised: false`） |
| P5 1.7B 同路径 | `operations/context-model-profile-1-7b-v0.0.68` | `accepted` | 设计差异恰好 `['model','model_profile']`；56 目标、头宽 2048、快照往返逐字节一致；参数比 2.6048 |

P4 的范围必须如实理解：v0.0.49 控制器已退出且**没有发布终态产物**，所以组合链不可能与它
做“逐字段同产物”比较。实际比较的是 **CPU 可计算的冻结输入**——每个真实 prepared 包的
docker argv、launch wrapper、selection 曲线发布文件与调用顺序、准入链结果与委托、
seam 绑定与代码对象；9 个 release pin 按新路径重定位。NPU 执行、ssh/docker/scp 传输、
收集传输、下游数值审核明确列为 `not_compared`。

失败的尝试与拒绝运行根（`-attempt1..4`、`-r2`、`-rejected-2/3/…`）与通过的一同保留，
在结论归档前不得删除。

## 6. 未验证边界

| 未验证项 | 原因 | 影响 |
| --- | --- | --- |
| 真实 PEFT adapter 构造 | 本机无 `peft` | LoRA 契约只由替身与快照 payload 证明 |
| 真实权重加载与前向 | 本机无 `transformers` | 头尺寸、tensor 数量来自 safetensors 头解析，不是前向 |
| 导出写出 | 本机无 `safetensors` | 往返用的是同格式写入器 |
| vLLM 引擎加载与对齐 | 本机无 `vllm` | 1.7B 的 serving 路径未跑 |
| 1.7B 四卡容量与 microbatch | 本轮范围只要 CPU 同路径 | 是否需降到 microbatch 4、packed-row kernel 是否逐位对齐，待队列排空后独立决策 |
| 1.7B 头 NPU 数值对齐 | 同上 | 只为设计预留，不构成准入 |
| 98 处源码改写债务 | 冻结包依赖未清空 | 新组合链不依赖它，但旧控制器仍在用 |
| 生产加速 | 全部为配对单次测量 | 不能当重复试验基准或生产性能承诺 |

## 7. 运行入口

```bash
PYTHONPATH=src:.:tools/jev /usr/bin/python -B -m <模块> \
  --config configs/portable/operations/<模板>.json \
  --local .local/config/<模板>-v0.0.N.json
```

| 阶段 | 模块 | 模板 | 本地宏 |
| --- | --- | --- | --- |
| P0 | `tools.reports.measure_context_engineering_baseline` | `context-engineering-baseline.json` | `context-engineering-baseline-v0.0.52.json` |
| P1 | `tools.reports.audit_context_model_geometry` | `context-model-geometry-audit.json` | `context-model-geometry-audit-v0.0.52.json` |
| P2 | `tools.reports.measure_context_parallel_execution` | `context-parallel-execution-measurement.json` | `context-parallel-execution-measurement-v0.0.53.json` |
| P3 | `tools.reports.context_component_smoke` | `context-component-smoke.json` | `context-component-smoke-v0.0.57.json` |
| P4 | `tools.reports.audit_context_composed_pipeline` | `context-composed-differential.json`（单值 `${COMPOSED_DIFFERENTIAL_CONFIG}`） | `context-composed-differential-v0.0.67.json` |
| P5 | `tools.reports.audit_context_model_profile_1_7b` | `context-model-profile-1-7b.json` | `context-model-profile-1-7b-v0.0.68.json` |

模板只写 `${MACRO}` 占位，个人路径全部在未入库的 `.local/config/` 中绑定。
所有入口的 `commit_json` 拒绝覆盖，`mkdir(exist_ok=False)` 保证运行根唯一。

组件测试 7 个文件一起执行：88 passed。

```bash
PYTHONPATH=src:. /usr/bin/python -B -m pytest -q \
  tests/test_context_model_profile.py tests/test_context_execution.py \
  tests/test_context_release_handle.py tests/test_context_evaluation_contract.py \
  tests/test_context_component_registry.py tests/test_context_composed_pipeline.py \
  tests/test_context_model_profile_1_7b_audit.py
```

`.venv` 缺 torch，CPU 检查统一用 `/usr/bin/python`（3.11.6 / torch 2.10.0+cpu / aarch64 /
24 核）。全库 collect 会因大量可选项缺依赖报错，按文件定向执行，不拿全库 collect 当门禁。

## 8. 维护规则

1. 新增组件先改注册表字面量，再改调用方；不允许运行时拼 import 或动态发现。
2. 并行宽度只从配置读；新增热点接入 `map_io` / `map_cpu` 时必须提供该负载自己的
   串行/并行等价证据，并测量该负载的交叉点，不用旧数字外推。
3. 任何“加速”结论必须来自同一负载的改前/改后配对测量，报告中保留两侧摘要。
4. 迁移站点一律新文件呈现，不原位改写被 pins 引用的文件；冻结运行包依赖清空前，
   旧改写债务保留并计入测量报告。
5. 组件契约（注册表字段、指标字段元组、宽度键名）变化时，先改本文件与测试，
   再考虑进入 `v0.0.rcN`。


## Stage2 难例训练数据组件补充

注册 release/stage2_difficulty_release_v2 → TrainingDataset；新增数据组件不改变KINDS。
固定入口 tools.reports.prepare_stage2_dataset，prepare/audit/bind动作，
配置 configs/portable/operations/stage2-dataset.json。使用context_execution的线程/进程池，
实现与训练consumer均为普通组合接口，不做源码改写。准入仅为CPU数据功能，非NPU训练。
详见[数据发布协议](stage2-dataset-release-v2.md)。
