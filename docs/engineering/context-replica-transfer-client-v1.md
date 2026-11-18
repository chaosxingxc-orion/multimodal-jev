# 简单任务冻结策略迁移客户端

本轮仅覆盖 FSC native31、MINDS native14，以及源域 SLURP clean TEST60。
前两项是跨数据集意图迁移，TEST 是源域留出评测，均不能冒称新任务泛化。
复杂数据集、多音频历史、多步判决保持 deferred。

`tools.reports.context_replica_transfer_client` 使用当前两个 replica 推理实例，
macro128、rollout32；每个 rollout 完整交给一个实例，完整 macro 才提交结果。
模型直接消费音频预处理张量和候选定义；预处理缓存是音频输入，不能替换为参考转写。

## 输入、来源和评测门禁

1. 在打开目标输入前，从完整 SLURP selection 曲线重算预登记的三个角色：
   `common_Stage1`、`source_selected_new_GRPO`、`fixed_endpoint_new_GRPO`。
   Stage2 的 step0 始终是本轮 Stage1 实际318终点；目标标签不能选策略。
2. 核验完整无标签 portable 输入包及独立人口证据，再将输入长度/公开分层决定的
   HF 面板绑定到完整人口。候选宽度固定31、14、60，题目不按当前样本答案生成。
3. 每个策略先在独立 HF 进程运行对应宽度面板，再通过 replica 服务运行全量人口。
   HF 与服务的 logit、prob 均要求 exact；概率舍入相同不能掩盖 logit 差异。
4. 缓存只读，缺失、hash不符或tensor身份不符立即失败，不在线重建、不修改共享缓存。
5. 全量预测必须完整且有序；独立 collector 重放每个 macro/rollout 日志、来源绑定、
   输入证据及HF比较，拒绝部分人口和篡改结果。客户端不读取目标gold、不评分。

客户端输出 `source-binding.json`、`input-evidence.json`、`panel.json`、
HF `reference.json`，以及全量推理journal、`parity.json`、`result.json`、`terminal.json`。
成功结果明确 `target_scored=false`。容器来源、镜像、设备、实际adapter张量、
完整源曲线独立审核仍由部署/收集层负责；客户端连接成功不代表这些审核已完成。

## 配置与冻结模型源码

入口模板 `configs/portable/operations/context-replica-transfer-client.json` 读取
`${REPLICA_TRANSFER_CLIENT_CONFIG}`；本地路径、缓存位置、socket和身份由本地宏绑定。
配置包含冻结来源 `source_evaluation`、`policy_role`、策略/adapter/runtime绑定，
`portable_full_inputs`、`portable_parity_inputs`、`macro_client`、`output`。
macro阶段额外绑定实际HF结果的路径/hash。与selection使用同一BF16模型和数学实现。

v0.0.37模型源码清单未包含完整portable读入和迁移审核模块，不能直接从工作区补import。
`context_replica_transfer_auxiliary.prepare` 将四个评测专用模块封在包外的
`transfer-auxiliary/`，生成逐文件hash绑定的 `transfer-client.py`。
严格白名单禁止覆盖封存源码或已加载模块；独立进程使用封存source目录导入。
模型源码清单保持不变，辅助模块和入口仍须被整个运行包清单绑定。

## 2026-11-17 验证及交付边界

90项相关CPU测试通过，包含两实例macro128/rollout32真实journal、129条尾批、三个候选宽度、
来源先行、HF策略/面板不匹配、logit差异、缺缓存、证据篡改和不支持任务拒绝。
模型/NPU前向使用测试替身，不能作为NPU数值或迁移成绩。
实际v0.0.37封存源码目录下的独立导入通过，无需更改模型runtime。

现有完整portable输入与其面板已重新核验：FSC 3793/64、MINDS 8168/259、
SLURP TEST 12991/269（全量/面板）。这是输入身份检查，不是远端缓存或模型输出审核。
同时发现旧TEST包绑定clean-intent-v2，而本轮训练使用v1；不能直接接本轮策略。
新的v1来源绑定已通过现有 `context_transfer_macro_inputs` 与
`context_transfer_portable_inputs` 固定脚本生成并重新读取核验，保留旧包，不原位改manifest或gold。
新人口、候选、tensor收据与旧包对应payload hash均相同。
新包为 `/srv/multimodal-jev-xingchao/operations/slurp-test-portable-full-inputs-v0.0.43`，
manifest SHA256 `7707ed311d6653cd6ea0c35e6588bd131281b6cdf8447bcbb6fc3e2e90e3ec01`，
独立人口证据 SHA256 `d9ebed7facb19ed6754dc2d766a5d47086b9f02a061f63e093c093f5d6c9ea7f`。
本地配置为 `.local/config/slurp-test-portable-full-inputs-v0.0.43.json`。
新包与原269条HF面板的独立覆盖审核通过，收据为
`/srv/multimodal-jev-xingchao/operations/slurp-test-full-parity-input-coverage-v0.0.43.json`；
只证明输入绑定，远端缓存实际字节仍须消费时核验。
隔离导入收据为 `/srv/multimodal-jev-xingchao/operations/context-replica-transfer-import-v0.0.43.json`。

封包、生命周期、评分接口及固定后继现已实现（见下节），尚需实际逐策略NPU对齐及
全量预测收集，然后统一评分并报告相对Stage1的变化及退化。在此之前没有本轮TEST或迁移成绩。
冻结策略角色尚待本轮完整源曲线，不提前派发目标评测，不改变已排队训练依赖。

## 运行封包、独立收集和原生评分

`context_cached_grpo_transfer_source.admit` 复用既有来源审核逻辑，将模型来源绑定到
本轮cached-GRPO实际训练布局。重新审核完整11点曲线、真实policy/head/adapter、
发布分数和选择结果，之后才打开目标输入。缺少完整源曲线时不制作可派发迁移包。

`prepare_context_replica_transfer` 从同一个Stage2父runtime封包，替换为已审核来源策略，
复制无标签完整输入和面板、封入来源分数与只读辅助模块。
模型source清单逐文件保持不变；唯一目标相关服务参数为候选宽度31/14/60。
独立 `audit` 在临时本地目录重建完整包并比较hash，不接受仅重新计算manifest后的篡改。
这不是再次保存主模型；策略包只复制可训练LoRA/head及处理器身份。

`context_replica_transfer_operations` 复用四卡设备租约、固定镜像、只读挂载检查、
准确容器wait、归档和传输审核。HF进程必须退出成功才启动两个服务；macro客户端没有
训练进程。审核核对实际设备、服务batch、候选宽度、supervisor正常终态、输入/输出及
每个rollout的完整分配，并在独立collector重算HF比较。任一失败保存终态、保留句柄，
不自动重试。只有容器/传输等外层审核通过才设置 `runtime_provenance_verified=true`。

`collect_context_replica_transfer_scores` 接回原生评分器。必须提供全部三个冻结角色，
各唯一policy先通过上述完整审核，最后才读取目标gold。相同policy的角色必须引用同一
执行与collection，保留零差值。报告set-F1@0.5、top1、概率质量及相对Stage1变化；
MINDS保留14地域变体/12语言统计。目标分数不参与训练、阈值调整或选模。

| 功能 | 模板（`configs/portable/operations/`） | 本地宏 |
| --- | --- | --- |
| 封包/重建审核 | `prepare-context-replica-transfer.json` | `PREPARE_REPLICA_TRANSFER_CONFIG` |
| 部署/收集/审核 | `context-replica-transfer-operations.json` | `REPLICA_TRANSFER_OPERATIONS_CONFIG` |
| 三角色评分 | `collect-context-replica-transfer-scores.json` | `REPLICA_TRANSFER_SCORING_CONFIG` |

封包配置含 `parent`、`source_admission`（原始source_evaluation/完整source_curve_config/
policy_role/policy_id）、`dataset`、两种portable输入spec、全新output/remote_root；
TEST还需实际source_release。运行配置含 `transfer_package`、package/manifest身份、
host/lease/共享只读路径/必要release_evidence、operations_root/tag及显式派发开关。
评分entries使用实际finalization/audit-config.json，不能填尚未执行的收据。

新增封包、来源、生命周期以及共享客户端的76项CPU回归通过；包括实际生成runner的
HF子进程退出顺序、真实macro journal重放，以及错误挂载/容器/传输/封包拒绝。
原生评分接入另有12项CPU检查通过，覆盖三数据集完整评分夹具、同策略角色复用、
缺失准入拒绝和新输出不覆盖。首次评分写出测试触发tmpfs保护，修正测试夹具的存储
替身后通过；生产本地持久盘检查保留。两组共88项，不代表实际模型分数。
完整模型、远端执行和源曲线在测试中使用明确替身，不能作为NPU准入或效果证据。
当前没有派发迁移任务，正式训练及源评测队列没有改变。

## 完整源曲线到迁移的固定后继（2026-11-17）

`tools.reports.context_replica_transfer_sequence` 已登记为本轮完整GRPO源评测的后继。
入口模板为 `configs/portable/operations/context-replica-transfer-sequence.json`，宏为
`${REPLICA_TRANSFER_SEQUENCE_CONFIG}`；本地配置
`.local/config/context-replica-transfer-sequence-v0.0.44.json`。
配置绑定前驱config/driver hash、201个代码/模板依赖、三数据集原生输入/评分契约及隔离输出路径。
等待使用准确pidfd，期间没有新增远程轮询或模型分析。

固定执行顺序：

1. 等待v0.0.42源评测准确驱动终态，重新核对全部11点的实际包与collection来源。
2. 重审训练、源评测及权重，冻结三个源角色，继承原host/四卡lease/父runtime/只读挂载。
3. 依次执行FSC、MINDS、SLURP TEST的全部唯一policy；重复角色复用同一执行。
4. 三数据集全部预测及审核完成后，才逐数据集读取gold并调用原生评分器。
5. 保存完整结果矩阵、配置和证据hash；任一失败停止下游、保留句柄、不自动重试。

后继及相关封包/来源/收集/上游队列共86项CPU回归通过。
测试覆盖前驱身份变化、失败终态、缺点、路径重叠、复杂数据集拒绝、别名复用、
所有推理完成后才评分以及重复启动拒绝；实际NPU为明确替身。
真实本地配置validate通过，登记不表示NPU或迁移已经完成。

本次启动驱动PID2330248，started_unix=1794955523.4252915，start_ticks=353217925。
现场 `/proc` 核对PID启动时间和argv一致，状态S、等待于poll；waiting.json已写，尚无终态。
该记录是启动时快照，不代替后续现场状态检查。
结果根 `/srv/multimodal-jev-xingchao/operations/context-replica-transfer-v0.0.44/`，
驱动记录在同级 `context-replica-transfer-v0.0.44-driver/`；终态为结果根下terminal.json。
等待中的配置和201个依赖进入冻结范围，不原位修改或清理。
