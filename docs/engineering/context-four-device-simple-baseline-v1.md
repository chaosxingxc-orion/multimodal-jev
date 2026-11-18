# 简单任务四卡实验基线

2026-11-17 用户最新约束：正式实验总预算四卡；只研究简单任务的训练效率、源任务收益和跨数据集迁移。当前配置为 `configs/experiments/context-two-stage-v0.0.11.json`，替代八卡 Stage1、六卡训练加两卡推理方案；旧验收不冒充新拓扑准入。

当前交付边界再次确认：复杂数据集和复杂判决不作为本轮依赖。Stage1 已完成318次更新并独立审核；11点selection尚未评分，Stage2尚未准入，源任务与迁移收益均待验证。本轮终点TRAIN256面板已通过exact审核。v0.0.37曾因部署漏传历史审核文件的只读挂载而失败；修复后的v0.0.39全量reference已于2026-11-18（北京时间）完成收集和独立审核，40,513条输入、317个macro全部从日志及独立输入身份重建，固定使用Stage1终点318。该结论是缓存完整性，不是全人口数值对齐或正式训练准入。继续完成GRPO有限验收和100步效率，再锁定正式训练；证据见[发布依赖校验](context-release-deployment-v1.md)。下方历次启动状态均为当时快照。

2026-11-17 21:06 UTC：固定100步效率后继已排队，等待v0.0.37有限验收链路；
此时没有效率包运行结果，不代表Stage2准入或效率收益。后继入口
`tools.reports.context_cached_efficiency_successor`，本地宏
`.local/config/context-cached-efficiency-successor-v0.0.37.json`；驱动PID2315346，
终态路径`/srv/multimodal-jev-xingchao/operations/context-cached-efficiency-successor-v0.0.37/terminal.json`。
前驱/配置哈希绑定与pidfd等待、三项有限证据重审、100步派发/收集/审核/成本报告均为固定程序；
失败停止、不自动重试，不派发634步正式训练。后继、效率封包及成本统计71项CPU测试通过，
NPU执行为测试替身，不替代实际验收。等待链路与引用文件均须保护，不原位修改或清理。
统计及可比性约束见[效率协议](../../experiments/context-cached-grpo-efficiency-v1.md)。

2026-11-17 21:13 UTC证据更新：v0.0.37 TRAIN256/60候选的本轮318终点数值验收
已独立收集审核，logit/prob最大绝对差均0，threshold flips/top1 changes均0。
容器`857c118969e724115e89d1fa3b14fcd42d27075697dc7e3ac43e912e57d74e80`；
证据`/srv/multimodal-jev-xingchao/operations/context-train-parity-v0.0.37/finalization/audit.json`，
准确传输清单SHA256 `627171987e7efb6762ae6cee48e4049fe7940004e68fb0e16aeaa959a2162ebe`。
外层审核确认runtime来源，内层比较仅负责数值；该结论为有限TRAIN面板，非全量或任务分数。
全TRAIN reference后继驱动已进入执行流程，尚无完整缓存与有限GRPO审核结果。

正式缓存GRPO封包/生命周期/独立审核入口`tools.reports.context_cached_formal`已实现，
相关53项CPU测试通过；要求真实100步效率审核通过、同起点/缓存/host/lease，
全量634步从Stage1重新开始，固定11点评测及稀疏optimizer checkpoint。
当前未生成或派发真实正式包，未将CPU验证或TRAIN256一致性作为正式训练成功。

2026-11-17 21:29 UTC现场快照：全TRAIN reference容器
`ad505c893c1619d7ab4ae2d9eb1cdea20f0dd83a556523afdfc58bbe9ee24434`
Running=true、OOMKilled=false，StartedAt=21:25:17Z；准确驱动负责wait/collect/独立审核。
下一次普通远端检查不早于21:44 UTC，除非收到终态或用户主动询问。

selection评测新增客户端、封包与独立审核已完成77项CPU检查，输入已绑定本轮release及
9226条原缓存证据，256输入面板已固定；终点318封包正重验真实源训练，尚无评测分数。
服务及模型源码保持v0.0.37不变，旧dual协议客户端不再直接接当前replica服务。
每策略HF面板验收、完整输入/日志重放和重计分边界见
[完整selection衔接](context-replica-selection-v1.md)。

后续更新：11点后台串接与replica曲线汇总已完成69项CPU回归及实际配置校验，
v0.0.40驱动PID2321924等待v0.0.39效率前驱。固定脚本逐点独立审核，再汇总完整曲线，
不以prefix选模；318终点评测包复用已独立重建通过的v0.0.38产物。
尚无本轮selection成绩，不自动派发正式GRPO，不推进复杂任务。

| 项目 | Stage1 | Stage2 |
| --- | --- | --- |
| 学习目标 | 单一 InfoNCE 区分性任务学习 | 排序/存在性双信用 GRPO |
| 资源 | 4 卡训练 | 2 卡训练 + 2 个单卡推理实例 |
| 每 DP 有效 batch | 64 | 64 |
| global batch | 256 | 128 |
| micro / 梯度累积 | 8 / 8 | 8 / 8 |
| TRAIN | 全量 clean SLURP 40,513 条 | 相同 TRAIN |
| 训练预算 | 两轮、318 更新 | 两轮、634 更新 |
| 每阶段真实样本暴露 | 81,026 | 81,026 |
| 每轮尾批 | 65 真实 + 191 零权重 padding | 65 真实 + 63 零权重 padding |
| warmup | 前16更新到1e-4，再恒定 | 按冻结GRPO配置 |
| 推理 | 无独立服务 | service batch32，rollout32，macro128 |
| checkpoint / selection | 0,32,64,96,128,160,192,224,256,288,318 | 0,64,128,192,256,320,384,448,512,576,634 |

设备映射由本地宏绑定，拟复用175的4/5/6/7卡：Stage1四卡，Stage2训练4/5、推理6/7。仅是配置映射，启动前仍检查实际资源。每次macro完成后才交训练消费，每个rollout由一个推理实例完整处理。全量正式训练不受效率对比100步上限约束。

Stage1终点318初始化Stage2，同时生成全部TRAIN的固定reference输出缓存。Stage2只读，缓存缺失拒绝；不以selection或迁移目标补充缓存、挖掘难例或调参。当前两阶段LoRA都是q/v rank4 alpha8，显式 `stage2.reuse_stage1_adapter=true`；false表示即使配置相同也新建参数，详见[分阶段LoRA规约](stage-lora-policy-v1.md)。

验收关注三项结果：

1. **效率**：固定四卡预算，对比真实queries/s、macro端到端时间，拆分服务准备/推理、权重切换、传输、actor更新、checkpoint和reference建库成本。效率对照最多100步，不能把减少真实数据或漏算离线成本作为收益。
2. **源任务收益**：完整9226条selection预登记曲线，固定阈值0.5，报告top1、macro-F1、set-F1、NLL、Brier、空集/多选情况及退化。正式策略由set-F1、NLL、最早步依次选择；完整clean TEST12991条只在冻结后评测。InfoNCE相对区分不等于独立sigmoid概率校准。
3. **迁移**：冻结策略和模板协议，评测FSC native31、MINDS native14，目标不参与更新、选模或阈值调整。它们是跨数据集意图迁移，不能称为跨任务收益；跨任务需另有可信新判决监督。

SFC、Audio2Tool、多音频历史、多意图组合、多步判决均延期，不进入当前研发、训练或评测依赖。

当前工程证据：新DP4/DP2完整日程已生成，分别在 `/srv/multimodal-jev-xingchao/operations/context-stage1-dp4-schedule-v0.0.11` 和 `context-stage2-dp2-schedule-v0.0.11`。固定镜像CPU原生verl检查61项通过；新四卡NPU普通prefix、真实尾批和新进程恢复均已独立collect及审核通过。它们是有限工程验收，不代表完整曲线、稳态效率或效果收益。Stage2 batch32数值一致性问题仍待解决，不放宽阈值。

固定镜像分阶段LoRA检查73项已通过，包括同配置显式复用/新建、异配置新建的真实PEFT CPU输出保持和冻结参数验证；报告 `/srv/multimodal-jev-xingchao/operations/context-stage-lora-cpu-v0.0.10/report.json`。这是CPU证据，不是双adapter的vLLM/NPU验收。普通训练监控900秒，事件只由确定性脚本处理。

独立日程复核通过：DP4 manifest `841979dac6891f6e9f4c52f26617bbd9dd3f7d1466e9732795e9a1ccf7353377`，DP2 manifest `39264e5146dccbd733dda60bf8283d5442a51d30785fd7fdf40e9a56787c0baa`。每轮完整人口、相同真实顺序、尾批权重均已重构；报告 `/srv/multimodal-jev-xingchao/operations/context-instruction-dp4-v0.0.11/schedule-audit.json`。

2026-11-17 Stage1发布证据（下列路径相对 `/srv/multimodal-jev-xingchao/operations/`）：

| 验收 | 证据 | 边界 |
| --- | --- | --- |
| 固定镜像CPU | `context-four-device-cpu-v0.0.11/report.json` | 61项通过 |
| DP4连续两更新 | `context-instruction-dp4-v0.0.11/infonce-audit.json` | 每更新256真实样本 |
| 新进程恢复 | `context-infonce-replay-dp4-v0.0.11/replay-audit.json` | step1恢复到2，master/Adam/执行权重/策略与连续运行相同 |
| 真实尾批 | `context-infonce-tail-dp4-v0.0.11/tail-audit.json` | 65真实+191零权重padding，再256真实；是诊断游标，不冒充正式159/160步 |

正式Stage1运行包 `context-infonce-formal-dp4-v0.0.rc5` 已启动，manifest为
`80b199f8a450ebc52d6338efbc6d480d2c8eced6ee0cb1d766d64d2c822c9f45`。
容器 `921b51a969630113e5fe7aabe3f418a54e11aafa3be01988c16b47f506d55341`
于 `2026-11-17T15:49:00.22813318Z` 启动，175的4/5/6/7卡，目标318更新。
本次现场inspect为running/nonOOM；这是启动后的观察快照，后续状态须查准确句柄。
设计JSON中的pending/禁止派发字段保留设计时快照；正式Stage1准入以本运行包及上述审核为准，
不原位改写被运行包引用的设计文件，也不因此放行Stage2。

收尾由固定 `context_infonce_formal` driver准确wait后collect，
`finalize_context_infonce`等待原driver的pidfd，再独立审核并绑定11个真实checkpoint的selection计划。
计划生成不代表评测执行；完整11点selection评分、TRAIN reference结果建库、Stage2准入及正式GRPO、
冻结TEST和迁移结果仍待完成。不会因Stage1退出码0直接报告收益或自动放行复杂任务。

正式运行已接入共享 `watch_context_replica_training`，配置模板为
`configs/portable/operations/context-replica-training-watcher.json`，本地绑定为
`.local/config/context-infonce-watcher-dp4-v0.0.rc5.json`。900秒观察间隔、准确容器/镜像身份校验、
单实例锁及终态停止均由固定脚本执行，不唤起模型，不自动重启或改参。
新增启动入口与间隔/终态检查在系统CPU Python下通过2项测试；项目`.venv`缺少torch，
现有observer的传递依赖需要torch，因此此监听使用与formal driver相同的`/usr/bin/python`。

## InfoNCE选择集衔接候选（2026-11-17）

共享评测入口原来只接受GRPO的634步来源，不能将InfoNCE的318步结果改名后接入。
新增 `tools.reports.context_infonce_selection`，重新独立审核完整训练来源，再绑定各点的
真实LoRA/head快照和导出的原生adapter；推理代码、macro协议及原生评分继续复用共享实现。
服务配置声明batch32、token上限131072、60个原生候选；模板历史batch4不作为当前配置。
新增 `tools.reports.collect_context_infonce_curve` 复用全曲线一致性核验，单独发布InfoNCE曲线。
选择集最优模型与Stage2初始化/reference所用固定318终点分别记录；不得按最优点替换reference。

本地配置模板为 `configs/portable/operations/context-infonce-selection.json` 和
`configs/portable/operations/context-infonce-source-curve.json`，分别读取
`${INFONCE_SELECTION_CONFIG}`、`${INFONCE_SOURCE_CURVE_CONFIG}`。前者的prepare配置需提供
`source_plan`及hash、`source_finalization`（原正式收尾配置）、`source_step`、
`runtime_template`及manifest、`binding_root`及hash、`cache_collection`、`cache_root`、
`release`及hash、新`output`及`remote_root`；后者提供相同来源绑定、按预登记顺序列出的
`evaluations[{step,config,local}]`与新`output`。部署配置还需精确旧版release证据的只读挂载，
不更改历史发布内容。

已实际核对本轮v1数据版本的9226条选择集请求、60候选顺序、全部独立缓存收据与既有
验证产物一致，允许复用该输入缓存；远端实际文件仍在消费时逐项验证。
CPU测试覆盖真实小模型BF16 adapter导出/矩阵重建、父包不可变、错误来源/人口/阈值拒绝、
完整曲线选择和固定终点保留；缓存与NPU执行在打包单测中是明确替身。
测试命令使用系统CPU torch及既有本地`safetensors-0.8.0`依赖目录，未修改训练镜像。

该入口当前仅开放prepare/validate/audit；**未派发评测**。
仍需将本轮策略的真实HF/vLLM batch32数值验收绑定到发布包，并完成自动串接，之后再跑11个
完整选择集点。旧runtime或CPU打包通过不等于新策略数值准入。当前Stage1冻结包及收尾进程
未修改；它们仅负责训练collect、独立审核及计划绑定，不会自动调用这个新候选。

## batch32语言层定位队列（2026-11-17）

原MINDS batch32诊断的 `stage-summary.json` 及实际捕获记录表明：baseline/acoustic两路径
声学各层均exact，单条最终输出exact，但整批32条中31条不同，最大logit误差0.34375、
prob误差0.0440187454。证据根为
`/srv/multimodal-jev-xingchao/operations/minds-audio-stage-batch32-v1/`。
这定位到声学之后，尚不能证明具体根因。源码中actor采用固定形状LoRA投影，而vLLM
保留原生LoRA内核，是待验证假设；没有据此改权重、扩大容差或宣称已修复。

独立诊断包 `minds-language-stage-v0.0.12-prepared` 已封存，manifest为
`f22ad2ed4a694347ef7f2b6e0ab5ec7d3fce7aad7f694fca2b6c855bbc5c9d19`，
根位于上述operations父目录。复用原始rollout32的完整邻居、顺序、输入、旧策略和镜像，
不读取gold或计算目标任务成绩。依次捕获HF、baseline、acoustic；保存每层input norm、
最终norm、首层Q/K/V、attention输出、post-attention norm、MLP输出与真实scheduler positions。
vLLM残差分离时不拿未相加的layer输出与HF完整残差直接比较。CPU clone避免后续原位操作
污染证据；独立比较拒绝缺阶段、长度/顺序/shape错误和非有限值。捕获耗时不是吞吐证据。

固定入口 `tools.reports.context_language_stage_operations` 复用既有准确句柄、租约、
设备空闲检查、收集和独立审核；宏模板 `configs/portable/operations/context-language-stage-probe.json`，
本地绑定 `.local/config/minds-language-stage-v0.0.12.json`。新增hook/比较/队列及原声学回归共14项
CPU测试通过；这不是新诊断的NPU结果。

`tools.reports.run_context_diagnostic_successor` 已启动一次性队列，启动PID2264698，
通过pidfd等待Stage1正式收尾PID2259036，并要求其完成完整训练审核及11点计划绑定。
之后才在同一租约下使用释放的175卡7做诊断；不打断当前四卡训练，不增加并发卡预算。
本地队列绑定 `.local/config/minds-language-stage-successor-v0.0.12.json`；
输出 `operations/minds-language-stage-successor-v0.0.12/`，此处operations指上述本地父目录。
队列校验源码/pins及前驱身份，无远程周期轮询，无重试、模型调用或自动改参。
当前为**已排队、尚未执行NPU语言诊断**；结果须真实collect及独立重算后再定修复方案。

## TRAIN reference独立输入证据（2026-11-17）

全量40,513条TRAIN已由固定 `tools.reports.context_reference_input_evidence` 实际复核：
本轮数据release、60候选及任务定义、每条audio/request身份、原始独立CPU审核流的
processor与张量哈希全部匹配。产物
`/srv/multimodal-jev-xingchao/operations/context-reference-input-evidence-v0.0.12.json`，
状态 `full_train_independent_input_stream_bound`，人口证据digest为
`d9104b8601ea55aaf3f5d065e1c1e9564da9b4db86ec3d54733abcfb4c9aa899`。
此审核不构造模型输出、不使用目标集，也不是本轮NPU或reference数值验收。

建库/审核候选直接消费原始JSONL收据与其独立proof，无须额外生成重复JSON收据。
build配置新增`input_proof`、`input_proof_sha256`、`input_receipts`、
`input_receipts_sha256`及`audio_mirror_root`，路径仍由本地宏绑定；audit配置同样绑定proof及stream。
实际缓存输入在交给推理macro前逐条对比独立张量哈希；事后审核将具体marker位置编码成
int64张量，再与原始位置哈希核对，并重建完整macro/rollout输出及样本映射。
相同输入可复用输出，但保留全部sample ID；缺样本、错位置或模型身份均拒绝。

终点校验新增四卡DP、instruction阶段、完整318步、joint快照manifest及source contract一致性。
音频只读本地mirror，保留原logical request路径和缓存键；不再逐条访问Ceph音频。
CPU检查覆盖输入篡改、重复输入、缺失、错误终点和真实GRPO session只发behavior请求；
尚需在Stage1完成及推理数值验收后实际建库、收集审核并接入Stage2有限验收。

## Stage2缓存消费与目标校验候选（2026-11-17）

`tools.reports.context_cached_reference_trainer` 的有限及正式生成入口均改为只读加载
音频前处理缓存，缺失即拒绝，在线processor工厂显式禁用；只检查运行输出的可写存储。
reference输出缓存继续使用完整TRAIN的只读结果，两类缓存分别绑定身份，不混淆输入缓存
与模型输出。当前Stage1及已排队诊断的冻结代码、配置不变。

新消费配置必须显式提供 `stage2_objective`，字段取当前设计JSON的Stage2：
`algorithm`、`arm`、`group_size`、`clip_ratio`、`advantage_std_floor`、
`ranking_rl_weight`、`existence_rl_weight`、`ranking_auxiliary_coefficient`、
`binary_nll_coefficient`、`reference_bernoulli_kl_coefficient`、`learning_rate`。
现有共享loss实现只支持已冻结的GRPO双信用常数，入口在NPU加载之前校验这些字段，
拒绝缺项、不支持的数值及冲突的顶层配置；不会把未实现的调参声明当作有效参数。
校验后的目标写入checkpoint contract。后续调整loss须新候选及独立验收。

系统CPU Python检查48项通过，覆盖真实缓存命中/缺失不写入、不前处理，实际GRPO session
只请求behavior，以及reference人口、输入证据和建库审核回归。命令：

```bash
PYTHONPATH=src:tools/jev /usr/bin/python -m pytest -q \
  tests/test_context_reference_dataset.py tests/test_context_reference_builder.py \
  tests/test_context_reference_audit.py tests/test_context_reference_input_evidence.py
```

此候选尚未发布为NPU训练包。部署入口仍需接入新配置、缓存只读挂载及完整依赖，
随后完成batch32数值准入、实际reference建库审核、DP2+两个单卡实例有限验收和恢复检查；
通过后才能启动正式Stage2。CPU通过不代表吞吐提升或源任务/迁移收益。

## reference与训练共享源码封包（2026-11-17）

reference身份包含全部Python源码manifest，因此必须在建库前封存建库、审核、缓存消费及
训练依赖，建库后不能补源码再假称使用相同runtime。新增固定入口
`tools.reports.prepare_context_cached_runtime`，通过
`configs/portable/operations/context-cached-runtime.json`读取本地整对象宏
`CACHED_REFERENCE_RUNTIME_CONFIG`。prepare参数为`parent`、`parent_sha256`、`project`、
`output`、`remote_root`；audit参数为`package`、`manifest_sha256`、`audit_output`。
入口复用原有运行包，递归补齐静态及生成trainer的依赖，并封存新源码身份；仅为代码准备，
旧`run.py`替换为明确拒绝派发的入口，不准许继承旧策略、旧六卡配置自动启动。
正式四卡配置、当前318终点来源、数值准入及已审核reference须在后续绑定，且不改变源码。

真实候选`/srv/multimodal-jev-xingchao/operations/context-cached-runtime-v0.0.14-prepared`
已完成426文件审核及仓库目录之外的新Python进程隔离导入、有限/正式trainer生成检查。
manifest为`3cc95889d94b78bf7ef5596c04fdb109ae0f8d81b7c545edb2558cccb1aa6eb7`；
runtime源码manifest为`8e4affbe1e399cae97a6d6e283fde157cd638c4cd3896827c757ec8519aba767`。
审核报告为同一operations目录下`context-cached-runtime-v0.0.14-audit.json`。
新增封包回归与reference测试合计54项通过；没有NPU部署、吞吐或效果结论。

开发包v0.0.13失败结论：继承旧macro trainer缺少`integrated`接口，隔离导入失败；
v0.0.14显式覆盖对应trainer模块后通过。失败证据指针为operations目录下
`context-cached-runtime-v0.0.13-prepared`。两者均为开发态代码包，不是锁定基线；
按规约待后继完整验收及保护清单确认后清理，不影响运行中/排队的Stage1与诊断包。

## InfoNCE终点到Stage2来源绑定（2026-11-17）

旧`prepare_context_stage2_source`要求历史634步及旧选择集来源，不能用于本轮318步
InfoNCE。新入口`tools.reports.context_infonce_stage2_source`复用完整训练独立审核与
source plan重构，核对真实318步commit、joint manifest及完整LoRA/head张量，再导出原生
LoRA。Stage2初始策略和reference策略固定同一终点，与选择集最优点分开。
新包只含policy、派生adapter及最小来源证据，不复制base权重或Stage1 optimizer状态。

配置模板`configs/portable/operations/context-infonce-stage2-source.json`读取
`${INFONCE_STAGE2_SOURCE_CONFIG}`：prepare需要`source_plan`及`source_plan_sha256`、
`source_finalization`、`design`及`design_sha256`、`output`；audit还需`manifest_sha256`和
`audit_output`。两种action均重新审核真实训练来源，不能凭一个passed报告跳过。
四卡预算、DP2+两个推理实例、batch32、macro128、全量40513 TRAIN和固定reference
均在绑定时校验；分阶段LoRA必须与Stage1预登记transition一致。当前仅复用adapter路径
获准进入后续验收；显式不复用仍需独立stacked actor/export/vLLM准入，不能偷换成复用。

新增来源绑定及已有InfoNCE选择集检查共14项CPU通过，使用真实小模型BF16张量和
原生adapter导出/矩阵重建，NPU训练来源在测试中为明确替身。
实际本轮318终点尚未产出，尚未创建本轮Stage2来源包或派发训练；需完整collect及独立
训练审核后运行该入口，之后仍需batch32数值准入、reference建库与DP2有限验收。

## 四卡运行配置衔接（2026-11-17）

`tools.reports.prepare_context_stage2_runtime`在共享源码包和真实终点来源包上分别重新审核，
再发布独立配置包：DP2、global128、micro8、两个TP1推理实例、service batch32/token131072、
rollout32/macro128。设备来自本地宏，必须四卡互异；进程可见ordinal按实际挂载集合计算，
不继承历史八卡scope。它替换旧策略路径和外部pins，清除旧resume/212步曲线/在线reference
缓存配置，保持Python源码manifest逐字一致。仅允许两更新工程窗口，启动入口仍拒绝派发。

封装与独立audit都会使用共享schedule加载器，从原始日程重建本轮DP2两轮全量TRAIN，
核对40513人口、634更新及真实/零权重尾批，不能只相信manifest中的总数。
audit还重建服务/训练配置、检查复制策略资产及依赖pins；篡改batch后重新封哈希仍会拒绝。

模板`configs/portable/operations/context-stage2-runtime.json`消费整对象宏
`${STAGE2_RUNTIME_BINDING_CONFIG}`。配置包含：

- `runtime`：共享源码审核参数`package`、`manifest_sha256`。
- `source`：上一节来源包审核所需全部参数（含`output`、`manifest_sha256`和训练重构来源）。
- `output`、`remote_root`、`training_devices`、`inference_devices`、`attempt_id`。
- `schedule`、`schedule_sha256`、`parent_schedule`、`parent_schedule_sha256`、`parent_release_sha256`。
- audit追加新配置包的`manifest_sha256`与`audit_output`。

执行`--action prepare`或`--action audit`，均不远程派发。相关41项CPU回归通过；封包测试
中实际使用小模型BF16终点、真实LoRA导出，完整NPU来源和日程以显式替身隔离。
实际本轮318终点尚未可用，因此未制作该真实绑定包；batch32数值准入、reference建库、
只读挂载及DP2有限训练/尾批/恢复验收仍待完成，不能以CPU通过宣称Stage2发布或收益。

## 缓存消费的容器存储边界候选（2026-11-17）

固定工具`tools.reports.context_cached_storage_operations`复用原Docker参数构造器，
只替换存储决策，不修改冻结启动器：前处理缓存及其manifest父目录、音频mirror、
外置reference publication、包内initial策略/adapter、源码/配置及其他已pin证据均只读。
运行根、exchange和运行缓存可写。reference必须与运行根不相交；四张物理卡必须互异。
审查按最长路径匹配判断有效挂载，拒绝嵌套可写挂载覆盖只读数据和重复挂载目的地。

`storage_preflight`生成的远程预检仅对新run根执行本地热存储检查；共享输入只检查目录、
regular manifest和依赖哈希，不创建文件、不同步缓存。`audit_container`复用准确终态
容器/镜像/设备/环境核验，再逐项核对Docker inspect实际`Mounts`的Source、Destination、
Type与RW，不能用HostConfig中请求的`:ro`参数替代实际权限证据。

宏模板`configs/portable/operations/context-cached-storage.json`消费
`${CACHED_REFERENCE_STORAGE_CONFIG}`，沿用package、manifest、共享目录/driver/tag配置。
此CLI只生成已封包的存储计划，没有launch入口；完整reference、来源和数值准入须由后续
部署生命周期绑定，不因生成Docker命令自动获得训练准入。
挂载/预检/实际inspect差异及旧共享命令回归共40项CPU测试通过；inspect为明确测试替身，
尚未部署容器验证真实只读挂载。旧启动器仍保留其历史可写cache逻辑，新候选部署时必须
接入本工具的command、storage_preflight与audit_container，不能直接调用旧launch冒充完成。

## 全量reference建库任务封装（2026-11-17）

`tools.reports.prepare_context_reference_job`先重审上一节四卡绑定包，并核对原始全量
TRAIN输入proof/JSONL，再生成独立建库包。复用同一Python源码manifest、endpoint策略、
两个batch32推理服务及macro128/rollout32协议；建库客户端单独使用reference journal和
attempt ID。训练器音频mirror配置指向`run`，建库器配置明确转为`run/audio`。
生成的`run.py`调用共享`supervise_services`，只启动两个推理服务和建库客户端，不执行
torchrun/actor更新；仍保留匹配准入的四卡设备scope，避免改变服务visibility语义。

模板`configs/portable/operations/context-reference-job.json`读取`${REFERENCE_JOB_CONFIG}`：
`parent`包含四卡绑定包的完整独立审核配置；`input_evidence`提供`input_proof`及hash、
`input_receipts`及hash；另需新的`output`与`remote_root`。proof和原始stream复制入包并pin，
builder通过现有宏加载器读取生成的`reference-template.json`和`reference-local.json`。
Stage1 commit、模型身份、输入缓存、processor、完整TRAIN覆盖和输出绑定继续由共享
builder执行校验；建库后必须重算全部journal及独立输入证据，不能仅凭exit0放行训练。

新增任务封装、生成runner执行检查及reference建库/审核回归共24项CPU通过。测试中的
服务supervisor与全量人口来源是替身，验证了实际生成的命令只启动建库客户端及两个服务；
不代表真实推理或建库已经执行。新包标记reference/training dispatch均未准入；实际318
终点、batch32数值验收和远程准确句柄生命周期仍是运行前置条件，本轮未启动远程任务。

## 固定镜像缓存Stage2集成CPU验证（2026-11-17）

通过共享`launch_fixed_cpu_tests`实际派发、准确docker wait、独立
`collect_fixed_cpu_tests`收集审核：97 tests、0 failures、0 errors、0 skipped。
容器`faff222f17838b95dc292be9585c855bd99d7184a1533cc26074b39a3b51b180`，
固定image `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
2026-11-17T16:57:20Z至16:58:25Z，ExitCode0/nonOOM。4 CPU、8GiB内存、network none，
未挂载任何NPU设备，不增加正式四卡预算。

源码/支持配置共1485文件，manifest为
`006963b28d444e5430dbc535bfd691b8355da2365b4279066e118b78c17ea19c`。
模板`configs/portable/operations/context-cached-stage2-cpu-tests.json`，本地绑定
`.local/config/context-cached-stage2-cpu-v0.0.15.json`；报告
`/srv/multimodal-jev-xingchao/operations/context-cached-stage2-cpu-v0.0.15/report.json`。
覆盖共享源码封包、InfoNCE来源/原生LoRA导出、DP2+2配置重构、只读挂载计划及inspect审核、
reference任务runner、不可变结果缓存、建库/journal及独立输入证据。
其中训练来源、NPU服务和实际Docker权限仍使用各测试声明的替身；报告实际解释器为
Python3.12.13、torch2.10.0+cpu。本结果证明固定镜像的CPU依赖兼容，不证明Ascend后端、
batch32精度、完整reference输出、吞吐或模型收益。

## reference任务独立审核及收集衔接（2026-11-17）

`prepare_context_reference_job`新增`--action audit`：重新审核父包、原始全TRAIN输入证据，
再重构builder/service配置、pins、runner和collection layout。修改配置后自封哈希不构成
来源证明；额外资产、修改endpoint/源码、缩小人口、改变设备或batch均拒绝。
包内新增`reference-build-config.json`，与宏解析结果严格相同，供共享结果审核器直接读取。
不修改已封存的v0.0.15 CPU测试快照；此变更为后续候选。

`--action audit-results`先完成上述审核，再核对外部声明的`expected_argv`、准确容器ID、
固定image、设备、环境、终态，transport完整文件清单和未变的执行配置，要求supervisor
确认两个服务及builder正常退出、没有强制清理。之后调用共享
`audit_context_reference_dataset`，用本地保留的已认证endpoint和原始输入证据审核收集的
全部dataset与journal；不会改写生产时remote路径或把producer自身哈希当独立输入证据。

audit追加`manifest_sha256`和`audit_output`；audit-results还需`collection`、`transport`、
`launch`、`inspect`、`expected_argv`及独立的`results_audit_output`。
结果报告明确`numerical_parity_verified=false`及`training_dispatch_allowed=false`；后续
部署必须将expected argv与实际只读挂载/数值准入绑定，完整结果审核不代替这些门禁。
新增任务及结果审核、原建库/journal回归共42项系统CPU测试通过；服务、NPU训练来源及
集成测试的最内层output replay有明确替身，真实journal重构另由现有测试覆盖。
实际reference尚未推理生成，因此没有真实audit-results报告或Stage2放行。

## 缓存结果接入训练与独立封包审核（2026-11-17）

`tools.reports.prepare_context_cached_training`复用共享runner，绑定DP2训练与两个TP1服务，
actor入口使用`context_cached_reference_trainer.py`。prepare先重新执行四卡父包审核及
reference任务完整结果审核，再核对同一Stage1终点、base、模型接口、runtime、数据release、
候选顺序和40,513条TRAIN人口；实际通过`ReferenceDataset`验证收集的缓存文件。
reference保留生产端独立路径，只固定manifest、records、population文件hash；不复制模型
主体或reference记录，不回退在线reference推理。

本入口只生成两次更新、逐次checkpoint的有限验收候选。父包、已收集reference与冻结
源码均不修改；不生成634步正式训练放行。四卡分工、global128、service32继续继承经过
独立审核的当前父包。只读挂载部署须结合`context_cached_storage_operations`，仍需实际
容器inspect证明权限、准确句柄收集及训练结果审核。

模板`configs/portable/operations/context-cached-training.json`读取`${CACHED_TRAINING_CONFIG}`。
prepare配置含`parent`（四卡绑定包完整审核配置）、`reference_job`（完整audit-results配置）、
新`output`和`remote_root`。独立`--action audit`另需`manifest_sha256`和包外`audit_output`。
审核重新核验生产来源与缓存，并从父包重构配置、pins、runner及收集目录；即使修改者重新
计算包内哈希，也拒绝改变更新预算、服务、源码、reference人口或将状态改为已准入。
审计报告与本地宏文件放在包外，避免修改冻结包。

本轮系统CPU回归79项通过，覆盖缓存训练封包、reference读取、只读挂载及reference任务审核。
测试使用真实缓存序列化/加载和实际生成runner，但NPU生产来源、进程supervisor及部分
输出审核有明确替身；不代表真实reference已经生成、Stage2已经启动或获得吞吐/任务收益。
固定镜像v0.0.15的97项通过不覆盖本节后续变更，需在后续候选中重新验收。

## reference建库与训练共用只读存储（2026-11-17）

`context_cached_storage_operations`现支持`kind=reference_build`和默认的
`kind=cached_training`，复用同一Docker命令、存储预检及实际挂载审核逻辑。
建库模式必须是全量40,513条、actor更新数0的reference包；仅新运行目录可写，输入cache、
音频mirror、endpoint、源码和pins只读，不要求尚未生成的reference目录存在或可读。
消费模式继续要求外置、已发布reference只读且manifest已pin，不允许隐式在线缓存。

reference `audit-results`现在额外需要`deployment`，包含tag、shared_root、driver_root及
可选release_evidence映射。它从独立审核过的包及deployment重构完整预期Docker命令，
与`expected_argv`及准确句柄比较，再核对inspect中的每项实际Mounts的source、destination、
type、RW。自声明argv或仅请求`:ro`不足以证明实际挂载只读；不接受可写输入、缺失或额外挂载。
结果报告新增`storage_audit`。这一验证不代替batch32数值准入或训练结果审核。

相关系统CPU回归89项通过。新增固定镜像集成模板
`configs/portable/operations/context-cached-stage2-integration-cpu-tests.json`独立于旧快照，
包括缓存训练封包及共享启动命令回归；本轮测试仍不挂载NPU设备，不增加四卡正式预算。

固定镜像v0.0.16已实际执行、准确wait并独立collect：**163项通过，0失败、0错误、0跳过**。
容器`0a14ad4b401abe02658ba1dfdf9c87c44e66e67928cfb27a613d608499a84c11`，
2026-11-17T17:15:37Z至17:16:53Z，ExitCode0、nonOOM；image沿用上述固定image，
4 CPU/8GiB、无NPU设备、network none。1487个源码/支持文件，manifest为
`5d517cb9d2f9848e3454438352a06c587e04d33913133f8cfa2e7442c4672273`。
本地绑定`.local/config/context-cached-stage2-cpu-v0.0.16.json`，报告
`/srv/multimodal-jev-xingchao/operations/context-cached-stage2-cpu-v0.0.16/report.json`。
实际torch为2.10.0+cpu；本结果验证镜像内CPU集成，NPU策略来源和服务仍由测试替身覆盖，
不证明实际建库容器挂载、数值一致性、reference输出或训练收益。

## 缓存训练结果审核接入（2026-11-17）

`tools.reports.audit_context_cached_training`适配原共享GRPO结果审核器。旧审核器假定首次
遇到请求会在线补算reference，因此不能直接审核本轮提前建库的训练结果。新入口先重做
训练包来源及reference输出审核，使用已收集、已认证的完整reference表逐条核对每个训练
slot的样本ID、音频/输入张量身份、marker及概率。重复样本和零权重padding也必须命中
原样本的同一reference；不以“本轮第一次出现”作为允许补算的理由。

每次更新必须记录reference_misses=0、bulk_reference_calls=0、bulk_behavior_calls=1、
switches为空。共享审核器仍重建每个behavior macro及其rollout分派，拒绝额外journal，
并检查真实日程/监督/双信用GRPO信号、actor行为概率对齐、FP32 master到BF16导出、
checkpoint状态和所有推理实例的权重激活。新增checkpoint契约精确绑定分阶段LoRA复用
决策、冻结objective和reference manifest；旧契约不静默视为新契约。

模板`configs/portable/operations/context-cached-training-audit.json`读取
`${CACHED_TRAINING_AUDIT_CONFIG}`，字段为`training_package`（完整prepare/audit配置）、
`deployment`（tag/shared_root/driver_root及可选release_evidence）、`launch`、`inspect`、
`wait`、`transport`、`archive`、`collection`及包外`output`。先核对准确容器、只读挂载、
exit0、transport完整清单、归档hash及原配置字节，再执行训练结果重构。
当前仅支持0→2的连续更新有限验收；真实尾批、新进程恢复、正式634更新仍需后续准入，
本入口不会派发或放行正式训练。

本轮系统CPU回归70项通过，包含实际GRPO session生成收据与只读reference核对、
概率/输入/样本替换和在线reference调用拒绝、契约及精确transport检查。
公共结果审核测试的NPU生产来源与最内层完整训练重放有显式替身；另有真实共享审核器
拒绝缺失训练产物的测试。这尚不是本轮两步NPU训练审核通过，v0.0.16固定镜像163项
证据也不覆盖本节后续新增代码。

## 本轮策略的TRAIN数值验收输入（2026-11-17）

`tools.reports.context_train_parity_inputs`从完整40,513条TRAIN及原始独立cache proof/JSONL
重构256条native60输入面板。先保留token长度和声学特征长度各16条最短/最长样本，
再按样本ID固定hash补齐，按ID排序；选取函数只接收输入证据，不读取gold、模型输出或错误。
面板与迁移诊断面板分别命名，不把TRAIN冒称TEST，也不把有限面板exact称全TRAIN精度一致。
此256条只服务数值工程验收，正式训练仍覆盖完整TRAIN。

模板`configs/portable/operations/context-train-parity-inputs.json`读取`${TRAIN_PARITY_INPUTS_CONFIG}`，
沿用reference独立输入配置的release/processor、input_proof/input_receipts及hash，新增
独立`output`。prepare生成无gold的请求、缓存key/hash、tensor证据、规则及选择清单；
`--action audit`追加manifest_sha256和包外audit_output，并从完整人口重新生成比对。
部署时只读encoder通过cache_root逐项验证文件hash、请求身份、实际张量和全部marker；
复用已有encoder但替换为禁止写入/缺失即失败的cache，不打开原始音频或调用processor。

该入口不加载策略、不执行HF/vLLM、不派发任务；后续须绑定实际Stage1来源和同一sealed
runtime，在batch32服务中执行并独立比较真实logits/prob。原排队语言层诊断不变。

实际输入包已生成且从完整TRAIN重新独立审核：
`/srv/multimodal-jev-xingchao/operations/context-train-parity-inputs-v0.0.17`，
manifest `37bd5c289d7bf396f8fa26ecaf8856e6fc94fb2ef273a4ffa81bfaef6f57cd72`；
审核报告为同父目录`context-train-parity-inputs-v0.0.17-audit.json`，
panel digest `fd8f088c72bd9920b6ce32b8215f836754854be20aa767e494710e946e3a8241`。
状态`train_parity_inputs_independently_audited`，256条、完整源人口40,513；远端cache实际
字节仍待消费时验证，未声称NPU执行或数值准入。本地prepare/audit分别使用
`.local/config/context-train-parity-inputs-v0.0.17.json`和`context-train-parity-inputs-audit-v0.0.17.json`。

`tools.reports.context_train_macro_parity`提供HF/macro两个客户端phase，模板
`configs/portable/operations/context-train-macro-parity.json`读取`${TRAIN_MACRO_PARITY_CONFIG}`。
绑定实际base、endpoint318 commit/joint policy/native adapter、同一runtime与TRAIN release。
HF复用原独立actor前向，仅将单挂载卡检查替换为共享四卡容器中的单卡CANN映射核验；
macro复用训练用的replica pool、macro128/rollout32和原输出/journal审核，实际logits与prob
均要求exact。差异先保存比较结果再以失败退出，不放宽阈值或自动回退batch1。

客户端只连接已启动服务，不负责资源租约、容器/封包或数值准入。必须将新代码纳入
独立版本源码包后再绑定本轮策略；旧v0.0.14源码包不原位覆盖，新reference只能在最终
源码锁定后构建。HF与服务supervisor串接候选见下节；准确容器收集和独立数值结果审核仍待完成。
本轮客户端/输入/独立证据系统CPU测试使用真实只读cache及replica journal路径；NPU
模型来源及服务计算是明确替身，HF共享设备映射分支尚待实际NPU验收。

## TRAIN数值验收封包候选（2026-11-17）

本轮继续只推进简单任务的效率、源任务收益和冻结策略跨数据集评测。复杂数据集不进入
本轮验收依赖。有限256条TRAIN面板用于查验数值路径，不替代40,513条正式训练。

共享源码包`context-cached-runtime-v0.0.18-prepared`已生成并通过隔离子进程导入审核，
共447个文件；manifest为
`02a6414bd986f963b02768586b441e6e5257154f80c2ba3e988eac7421cd4266`，
源码manifest为`309f728e98cf102d4c7dd76145bdb05bad8a20695559f8f6a5e426d978430c0c`。
包及`context-cached-runtime-v0.0.18-audit.json`位于
`/srv/multimodal-jev-xingchao/operations/`。这仅证明源码闭包及有限/正式入口生成可用，
没有绑定本轮实际endpoint318，不是NPU准入或reference建库完成。

新增`tools.reports.prepare_context_train_parity_job`复用共享服务supervisor，绑定已独立
审核的四卡Stage2父包及TRAIN面板。HF独立子进程先运行，成功退出并回收后才启动两个
TP1服务；macro客户端按128条完整macro、32条rollout消费256条面板，进行logits/prob
exact比较。该任务不启动训练rank，actor更新数为0；不会放宽阈值或回退成batch1。
HF真实设备映射、服务数值和进程清理行为仍须实际NPU运行验证。

模板`configs/portable/operations/context-train-parity-job.json`读取`${TRAIN_PARITY_JOB_CONFIG}`：
prepare提供`parent`（完整Stage2 prepare/audit配置及manifest_sha256）、`inputs`（完整
TRAIN面板prepare/audit配置及manifest_sha256、cache_root）、独立`output`和`remote_root`；
`--action audit`额外提供新包`manifest_sha256`和包外`audit_output`。
审核重新验证父包与全TRAIN面板来源，重建runner、HF/macro/服务配置、pins、设备范围及
采集layout，核对继承的模型端点与源码字节。仅重新计算被改文件的hash不足以通过审核。
输出明确标记`numerical_admission=false`、`training_dispatch_allowed=false`；审核不会派发任务。

本次系统CPU相关39项测试通过：包括真实HF替身子进程成功/失败退出后的服务启动门控，
真实只读cache与replica journal路径，以及重新计算hash后的配置/源码/端点/设备/manifest
篡改拒绝。父包NPU来源和模型计算为明确替身，不能视为实际HF或vLLM数值通过。
这些新增代码尚未包含在v0.0.16固定镜像163项证据中。新封包尚未绑定实际Stage1端点或
部署；准确容器生命周期、只读挂载、独立结果审核及batch32差异修复仍是后续工作。

## TRAIN数值验收生命周期与结果审核候选（2026-11-17）

新增`tools.reports.context_train_parity_operations`复用共享容器启动函数的租约、单实例锁、
固定镜像/设备检查、单次派发、准确wait及按layout采集；只替换封包审核、只读存储
preflight和状态名称。模板`configs/portable/operations/context-train-parity-operations.json`
读取`${TRAIN_PARITY_OPERATIONS_CONFIG}`，提供`parity_package`（完整封包audit配置）、
`package`、`manifest_sha256`、`host`、`tag`、`shared_root`、`driver_root`、`operations_root`、
`lease`及可选`release_evidence`。`--action launch`另要求
`finite_inference_dispatch_allowed=true`，不接受训练派发开关替代。长任务需由既有脱离
会话的工程驱动启动；本入口本身不增加队列、监控或自动重试。

共享数据、音频缓存、模型、端点和源码只读；preflight只检查新run目录可写，其他输入
仅验证存在与hash。启动响应超时后继续等待原容器，不创建替代实例；失败终态先采集，
再拒绝准入并保留现场。成功采集的状态仍为`collected_independent_train_parity_audit_required`。

新增`tools.reports.audit_context_train_parity`独立重建结果：原TRAIN证据的张量hash校验
实际marker值、全部输入身份和面板顺序，核对HF profile、两个服务的设备/策略/batch32/
token预算/入口身份与全部进程终态，再重放macro日志并重新比较logits及prob。报告中的
“exact”不能代替这些检查；不接受仅概率相同但logits不同。控制端审核无需访问远端cache。

模板`configs/portable/operations/context-train-parity-audit.json`读取`${TRAIN_PARITY_AUDIT_CONFIG}`，
字段为`parity_package`、`deployment`（tag/shared_root/driver_root及可选release_evidence）、
`launch`、`inspect`、`wait`、`transport`、`archive`、`collection`、包外`output`。
实际Docker Mounts、准确容器身份、执行配置、完整采集清单和归档hash必须先通过，
才执行数值重放。通过的范围仅是本轮endpoint318、TRAIN256/native60、macro128/rollout32
两实例；`full_population_parity=false`、`training_dispatch_allowed=false`，不替代后续
两更新/尾批/恢复验收及正式训练效果。

本地结果重放18项、transport审核7项、生命周期14项测试通过；模型计算、NPU来源、SSH
和Docker调用有明确替身，真实只读cache、宏批次日志、文件归档、锁及共享启动逻辑已执行。
固定镜像新快照为`context-train-parity-cpu-v0.0.19`；此前v0.0.16证据不覆盖本节代码。

该新快照已实际启动、准确wait并独立collect：**78项通过，0失败、0错误、0跳过**。
容器`fa5b36bcbb661b65f15d87d30184460bc6ba455578a0cc7981e778f99990bab2`，
2026-11-17T17:52:44Z至17:56:20Z，ExitCode0、nonOOM；沿用固定训练image，4 CPU/8GiB、
network none、无NPU设备。1499个源码/支持文件，manifest为
`695a763ff36b1f20d72760f840bb0cc5f64f97b70995dbc8ebb9811d799a2274`。
报告`/srv/multimodal-jev-xingchao/operations/context-train-parity-cpu-v0.0.19/report.json`；
共享模板`configs/portable/operations/context-train-parity-cpu-tests.json`，本地绑定
`.local/config/context-train-parity-cpu-v0.0.19.json`。测试包含真实cache和宏批次日志重放、
封包独立重构及异常拒绝、准确句柄与租约逻辑；NPU、Docker/SSH生命周期为测试替身，
不能称实际TRAIN数值验收通过或已解决batch32差异。没有修改正在运行的Stage1或排队诊断。

## 同端点数值验收到全量reference的衔接（2026-11-17）

新增`tools.reports.context_reference_job_operations`，以同一共享生命周期执行全量TRAIN
reference建库。启动前重新审核reference封包和实际TRAIN数值验收的完整采集证据，
要求两者来自同一个四卡Stage2父包，endpoint318、策略、base、数据release、输入证据、
源码manifest及服务配置一致。不同端点或runtime的成功报告不作为准入凭证；审核不只
读取状态JSON。256条数值面板通过仍只说明有限覆盖，不宣称40,513条全部HF/vLLM一致。

模板`configs/portable/operations/context-reference-job-operations.json`读取
`${REFERENCE_JOB_OPERATIONS_CONFIG}`。公共配置为`reference_job`（完整prepare/audit配置）、
`parity_audit`（完整TRAIN parity结果audit配置）、`package`、`manifest_sha256`、`host`、
`tag`、`shared_root`、`driver_root`、`operations_root`、`lease`及可选`release_evidence`。
`--action launch`必须显式`reference_dispatch_allowed=true`。建库仅启动两个TP1服务和CPU
客户端，不启动训练rank；按原四卡容器映射与租约执行，不增加设备预算，不自动重试。

`--action audit`需提供`deployment`、`launch`、`inspect`、`wait`、`transport`、`archive`、
`collection`、新的包外`results_audit_output`及`audit_output`。先重做数值准入，再核对准确
容器退出、执行配置与归档，最后调用原全TRAIN reference重建审核。只读表发布、数值
验收与训练准入分别记录；通过建库审核不自动放行GRPO或复杂任务。审核输出不可覆盖，
需要重做审核时使用新输出路径。

本地84项相关回归通过，覆盖不同端点/数据/源码/输入证据拒绝、数值失败传播、显式派发
开关、共享启动约束及结果审核衔接；来源NPU计算和远程调用为替身。固定镜像CPU候选为
`context-reference-lifecycle-cpu-v0.0.20`，实际reference建库仍等待本轮Stage1端点与数值准入。

固定镜像v0.0.20已准确wait并独立collect：**84项通过，0失败、0错误、0跳过**，
容器`2b909c49961f8e44d3bfc2d8ad54d0e11205933ad741e5c1ebe07aa8e598021b`，
2026-11-17T18:01:32Z至18:02:37Z，ExitCode0、nonOOM；4 CPU/8GiB、无NPU、network none，
1501个源码/支持文件，manifest为
`9148160b64d80b8e64473f59d279ea9277007b943b83d932278f6e9458baf993`。
报告`/srv/multimodal-jev-xingchao/operations/context-reference-lifecycle-cpu-v0.0.20/report.json`，
模板`configs/portable/operations/context-reference-lifecycle-cpu-tests.json`，本地绑定
`.local/config/context-reference-lifecycle-cpu-v0.0.20.json`。这是CPU工程集成证据，未执行
真实reference生成、没有发布Stage2训练准入或取得新的任务指标。

## reference重复审核修复（2026-11-17）

衔接检查发现：原`prepare_context_reference_job.audit_results`在被训练封包prepare/audit
重复调用时，每次都会调用dataset审核器发布同一个不可覆盖报告，导致第二次校验因
文件存在而失败。此问题会阻断缓存训练封包的正常审核，和模型数值差异无关。

现由`replay_reference_outputs`复用原审核器的完整逻辑，只在嵌入调用时将最终报告写入
替换为内存接收；每次都从当前输入和journal重新计算，不读取旧报告作为通过依据。
原dataset审核器及已封存runtime源码未变。独立CLI的`audit-results`和reference lifecycle
的`audit`仍显式发布`results_audit_output`与总报告，保持不可覆盖；嵌入prepare/audit
重算则不创建、覆盖或删除这些文件。此前发布的v0.0.20固定镜像快照不包含本修复。

本地74项相关回归通过，新增真实reference表/宏批次journal连续重放、原文件字节保持、
先通过后篡改拒绝及显式CLI审核函数仍禁止覆盖等检查。测试里的Stage1/NPU和完整native60
来源为明确替身；真实全TRAIN建库与训练准入仍待本轮端点和数值验收。

固定镜像`context-reference-reaudit-cpu-v0.0.21`已执行并独立collect：**74项通过，0失败、
0错误、0跳过**，容器`edd7da93414509637d2a017fc141d3c719885c81e352ba29874d125a78846d25`，
2026-11-17T18:07:36Z至18:08:49Z，ExitCode0、nonOOM，无NPU设备；1502个源码/支持文件，
manifest `0f60cd10e18fe56c4936e309e9e818e3cca4b59b9e6afef18bbbec971db59542`。
报告`/srv/multimodal-jev-xingchao/operations/context-reference-reaudit-cpu-v0.0.21/report.json`，
模板`configs/portable/operations/context-reference-reaudit-cpu-tests.json`，本地绑定
`.local/config/context-reference-reaudit-cpu-v0.0.21.json`。已核对原dataset审核器源码hash
仍与封存v0.0.18相同；修复只作用于控制端调用和显式报告发布，不改运行包。

## 缓存训练两更新执行入口（2026-11-17）

新增`tools.reports.context_cached_training_operations`，将已实现的只读缓存训练封包接到
共享租约、准确容器wait和collect。启动前重新执行训练封包审核及reference lifecycle完整
审核（含本轮端点TRAIN数值验收），要求同一父包、相同reference生产包与收集目录、
完整40,513人口、同一策略/runtime、源码与端点资产。旧成功报告不会替代实际证据重放。

此入口仅允许engineering prefix从0到2、每步checkpoint、无resume、无在线reference；
两卡训练加两个TP1服务，reference与输入cache只读。共享启动逻辑的可写cache检查由
consumer只读preflight替换，其余固定镜像、四卡资源检查、租约、单次尝试、准确句柄与
失败收集逻辑复用。冻结或待审核包不原位修改。

模板`configs/portable/operations/context-cached-training-operations.json`读取
`${CACHED_TRAINING_OPERATIONS_CONFIG}`。validate/launch提供`training_package`（完整
缓存训练prepare/audit配置）、`reference_audit`（完整reference lifecycle审核配置）、
`package`、`manifest_sha256`、`host`、`tag`、`shared_root`、`driver_root`、`operations_root`、
`lease`及可选`release_evidence`；launch额外要求`finite_training_dispatch_allowed=true`。
推理或建库开关不能代替该训练开关。长任务由现有脱离会话驱动运行，本入口不追加监控或
自动重试。成功collect仍只报告`collected_independent_cached_training_audit_required`。

audit使用原缓存训练结果审核字段（deployment/launch/inspect/wait/transport/archive/
collection/output），并提供相同`training_package`和`reference_audit`。先重做前置证据，
再审核两步真实训练、缓存逐slot消费、behavior rollout、权重切换及checkpoint；即使
通过仍标记`formal_training_admitted=false`，真实尾批、新进程恢复及正式发布另行验收。

本地86项联合回归通过，涵盖只读reference表、错误来源拒绝、有限更新边界、真实宏批次
和缓存消费核对、原有transport审核及两步启动衔接。NPU来源、模型计算、SSH/Docker为
明确替身，不能称真实两步NPU训练成功。固定镜像验证候选为
`context-cached-training-lifecycle-cpu-v0.0.22`；尚未派发实际Stage2。

固定镜像v0.0.22已准确wait并独立collect：**86项通过，0失败、0错误、0跳过**，
容器`37f11cfdb65a812b4fe3bbb026e00635f60434d2dd740fd5c03b95ac07ee59ce`，
2026-11-17T18:13:39Z至18:15:06Z，ExitCode0、nonOOM，4 CPU/8GiB、network none、无NPU。
1508个源码/支持文件，manifest为
`faf485b44d120e85c33da5bcd4154b2a950fdf82dcb38410f1c77e2208d10567`。
报告`/srv/multimodal-jev-xingchao/operations/context-cached-training-lifecycle-cpu-v0.0.22/report.json`，
模板`configs/portable/operations/context-cached-training-lifecycle-cpu-tests.json`，本地绑定
`.local/config/context-cached-training-lifecycle-cpu-v0.0.22.json`。该快照是CPU接口/生命周期
集成证据，未执行实际两卡GRPO更新，不覆盖真实尾批、新进程恢复或完整效果曲线。

## TRAIN数值面板实际rollout分派复核（2026-11-17）

原通用macro评测审核器验证完整人口、输入张量、返回结果和journal哈希，但不要求journal
中的分派与声明的rollout32算法一致。因此“服务配置batch32 + 输出exact”本身不足以排除
实际使用rollout16/64或只有一个实例的情形。

控制端`audit_context_train_parity`现复用训练侧`audit_journal`和相同`plan_rollouts`，从
已认证mask哈希恢复有效长度成本，逐macro重建rollout32的成员、顺序和实例归属，并核对
全部收据。TRAIN256必须恰为两个完整macro128，各四个rollout32；自行重算hash的错误
分派也拒绝。结果新增`dispatch_audit`。不修改已封存推理源码、数值阈值或原运行配置。

证据边界：此项证明客户端whole-rollout分派，服务启动证据证明配置上限32；vLLM内部
调度/前向大小仍不是这两者的同义词，报告显式`engine_internal_batching_verified=false`。
既有实际TRAIN面板256条缓存key均不重复，最长3104 token；这些静态事实不替代实际
NPU执行证据。模型batch32差异仍由已排队语言层诊断定位，不因审核增强而认为已修复。

本地与固定镜像均28项通过；固定镜像`context-train-dispatch-audit-cpu-v0.0.23`已准确wait
和独立collect，0失败/错误/跳过，容器
`ef0e87540393e4c9f28e167045c72ee29a61202fb5b330dafc1db2dc1a8dd1c9`，
2026-11-17T18:19:48Z至18:23:21Z，ExitCode0、nonOOM、无NPU。1499个源码/支持文件，
manifest `e79ad93b1061689cf6b993a33eef56af514b515541bbf89daebf987d8a9faf11`。
报告`/srv/multimodal-jev-xingchao/operations/context-train-dispatch-audit-cpu-v0.0.23/report.json`，
模板`configs/portable/operations/context-train-dispatch-audit-cpu-tests.json`，本地绑定
`.local/config/context-train-dispatch-audit-cpu-v0.0.23.json`。新增反例以真实macro pool生成
数值exact且哈希自洽的rollout16/64与单实例日志，独立审核均拒绝；模型计算和NPU来源
仍为替身。原Stage1及排队诊断未修改。


## 缓存GRPO真实尾批入口候选（2026-11-17）

现有缓存GRPO验收只覆盖普通前两批，不能据此认为65真实样本的尾批归约已通过。
新增 `multimodal_jev.context_cached_tail_window` 与 `tools.reports.context_cached_tail_trainer`，
复用原缓存GRPO生成器、InfoNCE终点初始化、只读reference、双信用目标和整个更新循环。
先验证完整40,513条两轮日程，再选零基索引316/317：65真实+63零权重填充，随后128真实。
保留原epoch/index及样本身份；优化器是新建的诊断0/1游标，不能冒称正式317/318步历史。
checkpoint额外绑定 `tail_admission`，禁止将诊断checkpoint当作普通正式历史恢复。

统一封包 `prepare_context_cached_training` 新增可选 `acceptance_kind`，默认
`engineering_prefix`；尾批显式配置为 `cached_reference_tail_admission`。同一生命周期和
审核入口消费该字段，无须另建一套调度脚本；仍只允许两次更新，不允许resume或正式派发。
尾批入口必须已包含在父包的源码manifest内，禁止在reference构建后注入新代码。
新源码闭包生成器包含尾批入口并在隔离进程编译验证；已有封存v0.0.18及运行/排队包不变。
下一runtime应在真实模型数值修复后统一封存，再做同版本parity和reference建库。

结果审核复用原宏批次journal、样本/权重、group reward、所有rank归约、LoRA/head导出、
Adam/master和服务切换核验。只替换经认证的日程窗口及尾批允许位置，完整日程仍先验证；
仍要求逐slot reference结果一致、零在线reference调用。成功状态单独命名为
`finite_cached_reference_tail_outputs_audited`，不自动准入正式训练或复杂任务。

本地87项测试通过，覆盖真实全量日程/桶划分、65/63权重、日程损坏拒绝、诊断历史边界、
更新循环AST一致、封包不可变、遗漏尾批源码拒绝，以及原只读生命周期与结果审核回归。
NPU来源和远程调用为测试替身；真实尾批梯度、服务推理和新进程恢复尚未验收。
固定镜像候选 `context-cached-tail-cpu-v0.0.24`，共享配置
`configs/portable/operations/context-cached-tail-cpu-tests.json`，本地宏
`.local/config/context-cached-tail-cpu-v0.0.24.json`，不占NPU。


v0.0.24固定镜像测试包漏带`context_reference_job_operations`和缓存训练审核模块，pytest
收集阶段2个import错误，未执行87项用例；失败报告
`/srv/multimodal-jev-xingchao/operations/context-cached-tail-cpu-v0.0.24/report.json`。
已将现有缓存训练生命周期测试依赖并入同一模板，独立快照v0.0.25接续验证；没有修改
旧容器/已封包文件，也没有将测试打包失败归因于NPU模型或训练镜像。


v0.0.25已准确wait并独立collect：**87项通过，0失败/错误/跳过**。固定镜像、4 CPU/8GiB、
network none、无NPU，2026-11-17T18:40:49Z至18:42:20Z，ExitCode0、nonOOM；
容器`26c94fac80c5f0d549fd53efdc9b47bea7d1f40a2bb3b324752ac87b59fa5604`，
1510文件，源码manifest `4d35f4c090c531f5772632b644a4c3483de9d57b045563aca3d42cce970aee6e`。
报告`/srv/multimodal-jev-xingchao/operations/context-cached-tail-cpu-v0.0.25/report.json`，
本地宏`.local/config/context-cached-tail-cpu-v0.0.25.json`。CPU证据不等于真实NPU尾批、
新进程恢复或GRPO效果通过；两步缓存训练与尾批运行仍等待本轮endpoint、batch32精度及
全TRAIN reference准入，当前不派发。


## 缓存GRPO新进程恢复入口候选（2026-11-17）

新增`prepare_context_cached_recovery`将已独立审核的普通0→2更新包绑定成1→2恢复包。
同一Stage1 reference、runtime源码、目标、日程、DP2结构保持一致；只复制第1步
adapter/head/optimizer及必要策略manifest，不复制主模型。两个推理实例以第1步behavior
启动，`source_policy_id`和reference仍固定为Stage1终点。源包及其结果不原位修改。
恢复输入置于新包`resume/`，Docker只读挂载；逐文件pins与实际Mounts审核继续执行。

模板`configs/portable/operations/context-cached-recovery.json`读取`${CACHED_RECOVERY_CONFIG}`，
prepare提供`parent_audit`（原缓存训练生命周期完整audit配置）、`output`和`remote_root`；
audit再提供`manifest_sha256`及新的`audit_output`。不接受尾批诊断作为普通恢复来源。
`context_cached_recovery_operations`复用缓存训练的validate/launch与共享租约、准确wait、
失败collect；模板`context-cached-recovery-operations.json`读取`${CACHED_RECOVERY_OPERATIONS_CONFIG}`。
配置以`training_package`承载恢复封包说明，其他package/host/tag/driver_root/shared_root/
operations_root/lease与原生命周期一致；launch仍需`finite_training_dispatch_allowed=true`。
结果audit沿用deployment/launch/inspect/wait/transport/archive/collection/output字段。

独立审核先重放原训练及parity/reference来源，再核验新容器完整产物；恢复前的behavior
权重由第1步真实joint policy提供，reference单独保持Stage1身份。最后比较连续与新进程
第2步的样本、groups、逐rank FP32 master/Adam、BF16 execution/joint policy，以及逐query
behavior/reference输出，排除耗时等非确定性字段。仅比较文件不等于NPU恢复：必须先有
两次独立执行审核且容器身份不同。正式GRPO准入仍不由本入口自动放行。

本地新增17项检查通过；本轮完整组合102项中两条测试断言曾将collect完整handle误当ID，
修正断言后新增集17项全部通过，其余85项此前通过。封包来源/NPU模型和远程调用是明确
替身；测试实际覆盖recovery封包重构、状态篡改拒绝、只读挂载及共享生命周期。
固定镜像候选`context-cached-recovery-cpu-v0.0.26`使用
`configs/portable/operations/context-cached-recovery-cpu-tests.json`与本地同版本宏，
尚待独立收集。未派发实际Stage2恢复训练；已有封存runtime与原排队诊断未修改。


v0.0.26固定镜像实际102项，96通过、6失败；失败均为漏带
`tests/test_context_replay_audit.py`夹具，未执行对应比较。报告
`/srv/multimodal-jev-xingchao/operations/context-cached-recovery-cpu-v0.0.26/report.json`。
补齐support清单后另行创建v0.0.27验证，不修改v0.0.26封包或容器。


v0.0.27已准确wait并独立collect：**102项通过，0失败/错误/跳过**。固定镜像、4 CPU/8GiB、
network none、无NPU，2026-11-17T19:01:56Z至19:03:24Z，ExitCode0、nonOOM。
容器`9ea0e7701f7bca29cdf85b09372d28e977dc7d271e4810533750aa3c360de7e1`，1515文件，
源码manifest `c1b6358b521245497aebd12f067647f5686b1b24890de1bd38f9d65a0dc258f8`。
报告`/srv/multimodal-jev-xingchao/operations/context-cached-recovery-cpu-v0.0.27/report.json`。
这些是CPU封包/生命周期/恢复比较证据，真实NPU恢复仍未执行；controller改动未覆盖旧封存runtime。


## 本轮Stage1完整训练收口（2026-11-17）

原Stage1驱动已exit0并collect，独立finalizer已完成全部318次更新审核，状态
`formal_infonce_artifacts_audited`，DP4、两轮完整81,026次真实样本暴露。
证据根`/srv/multimodal-jev-xingchao/operations/context-infonce-formal-dp4-v0.0.rc5/finalization/`：
`training-audit.json`、`source-evaluation-plan.json`、`terminal.json`。
11个预登记点0/32/64/96/128/160/192/224/256/288/318均绑定真实策略。
固定reference端点318为`926e4d1316cefea8e39272b5153cbc7fb96880eb67b0da42896cd07ddc620095`。
`evaluation_complete=false`，尚未实际评测11点曲线，不能称任务或迁移收益。

后继语言层诊断已由原队列于2026-11-17T19:02:08Z启动；现场inspect核对容器及镜像身份，
容器`2496e15c59fd1660099098742f47ada61dbe574f9dcf4a8551b1d2e2d0e29ae7`当时running/nonOOM。
这是启动快照，仍需原队列准确wait、collect和独立比较后解释batch32差异，不据此宣称修复。


## batch32语言注意力边界诊断（2026-11-17）

v0.0.12语言层诊断已完成且独立重算通过，状态
`language_tensors_independently_compared_not_admission`。报告：
`/srv/multimodal-jev-xingchao/operations/minds-language-stage-v0.0.12/jev-minds-language-v0-0-12-collected/evaluation/report.json`，
SHA256 `d8ef56dbb44f3b7f897d6b8b8e74999b3c488f2e2d23653c5e802634919f86bd`。
原32条单独请求均一致；batch32中31条logit/prob不一致。baseline与声学批处理候选均为：
首层input norm和Q/K/V投影输出完全一致，首层attention output有3438元素不同、最大绝对差0.0078125。
首个归一化隐藏差异位于layer1.input_norm；目前只能定位在QKV之后至o_proj之间，不能据此判定LoRA、RoPE或注意力核根因。

新有限诊断v0.0.28沿用同一32条有序输入和历史策略，无gold、无训练、无阈值调整；增加首层
q/k norm、RoPE及o_proj输入捕获。此MINDS输入仅用于复现已知工程差异，不消费目标标签或指标，
不作为迁移选模；最终仍需本轮Stage1终点在SLURP TRAIN256面板的独立验收。
共享capture/successor及原比较回归17项本地CPU测试通过，NPU结论仍待准确wait、collect及独立审核。
包manifest `646605e3437b1bed2ff2d66f6be287c04533efe5db1351f88028417af5e904f6`；
容器 `fa46d5c1fe06ec963c3485795ff43191ea42dc90f3c8d247aa6ae12ca3738d28`，仅175卡7。
本地宏`.local/config/minds-attention-stage-v0.0.28.json`与
`.local/config/minds-attention-stage-successor-v0.0.28.json`；固定驱动使用共享租约、单次派发、
准确容器wait、自动collect和独立张量比较，终态写入
`/srv/multimodal-jev-xingchao/operations/minds-attention-stage-successor-v0.0.28/terminal.json`。
此次启动不放行Stage2，不构成效率或任务效果收益。


## 实际Stage1终点导出与注意力诊断收集（2026-11-17）

Stage2初始化来源v0.0.30已通过`context_infonce_stage2_source`独立重建审核：
`/srv/multimodal-jev-xingchao/operations/context-infonce-stage2-source-v0.0.30-audit.json`，
包manifest `ba23e9a82721871af3f28fc06704fcf3d79a9fe3c2d1b43d9cba09831d669730`。
来源固定为本轮318终点`926e4d1316cefea8e39272b5153cbc7fb96880eb67b0da42896cd07ddc620095`，
导出adapter及joint policy，不复制base或optimizer，未执行NPU、未放行训练。
本地宏`context-infonce-stage2-source-v0.0.30.json`和同版本audit宏位于`.local/config/`。
v0.0.29因控制器未加载现有safetensors依赖导出失败，只留下未发布的policy副本；
v0.0.30复用`.local/verification-python`中的既有safetensors0.6.2，未安装新依赖或修改镜像。

注意力诊断v0.0.28于2026-11-17T19:31:38Z exit1、nonOOM，错误为比较阶段
`HF request length differs`。原始HF row00捕获显示：q_rope=[762,2048]，
kernel_output=[1,1560576]，是捕获器把HF o_proj输入[1,T,D]误展平为[1,T*D]。
原固定驱动继续收集，不能把容器退出或张量存在当作审核通过。
新增`tools/reports/replay_context_attention_capture.py`在传输完成后核验原容器/配置/源码/归档身份，
只将已知kernel捕获恢复为[T,D]，复用原比较器；原始失败终态及数据不改写。
2项CPU测试验证原值保持、输入不改写和未知形状拒绝。尚未获得重放比较结果，
不据此归因RoPE或注意力核，不派发新的NPU重跑。


离线维度恢复重放已完成（同日）：原116文件传输完成并逐文件校验，归档及来源身份由
`replay_context_attention_capture`再次验证，原失败exit1保留；输出
`/srv/multimodal-jev-xingchao/operations/minds-attention-stage-v0.0.28/replayed-comparison-v1.json`，SHA256 `17c3d498cbb2095600f1c4f4f664f75770d8d1880ded7d0a266642c7a0672f4e`。
CPU新增工具及比较器回归10项通过。

两种声学路径的batch32结果相同：layer0输入、Q/K/V、q_norm/k_norm、q_rope/k_rope全部精确一致；
**首次已捕获差异位于kernel_output（o_proj输入）**，86个元素不同，最大绝对差0.001953125。
o_proj输出3438个元素不同、最大绝对差0.0078125；32条单独请求所有捕获阶段均精确一致。
下一有限实验应直接重放同一Q/K/V，检查单条/批量的布局、stride/contiguous及Ascend TND核路径。
现有证据定位核输入至输出边界，尚不能区分输入布局、核tiling或内部累积差异；不能直接
声称已发现底层算子bug。无需重新训练、调整LoRA或修改RoPE，也不放宽batch32准入阈值。
本次重放是诊断证据，不把原失败容器改写为成功验收；Stage2仍未准入。


## 注意力核真实输入重放候选v0.0.31（2026-11-17）

复用统一有限生命周期与同一设备租约，固定卡7、固定镜像，启动`attention_kernel`诊断。
从v0.0.28已完整收集且哈希校验的原始HF捕获提取32条首层q_rope/k_rope/V和期望核输出；
逐值核验它们与原batch32捕获相同。输入不含gold、无需加载模型或再次执行完整网络。
在同一输入上比较batch1/8/32，contiguous及模拟融合QKV切片的strided V，
以及inner_precise=2/0/1。参数名不预先解释为精度等级，必须以实际差异判断。
每个模式同时比较原HF、原batch32及该模式单条结果，避免把改变所有结果当成一致性修复。
测量含传输且无warmup，只作核诊断，不能用这些耗时宣称吞吐提升。

新增`context_attention_kernel_probe`和对应operations入口，生命周期继续复用
`run_context_diagnostic_successor`、单次派发、准确wait/collect与CPU独立张量比较。
CPU布局/比较及successor回归9项通过。包manifest
`4a583375fc72e854ad1e7bdbbb37196aa30375346c23dade9ae1a90d55cbb5c9`，输入包约401MiB。
本地宏`.local/config/attention-kernel-v0.0.31.json`、同版本successor宏；
驱动PID2304568，终态预期路径
`/srv/multimodal-jev-xingchao/operations/attention-kernel-successor-v0.0.31/terminal.json`。
这是派发记录，不是NPU审核通过或Stage2准入。


核级v0.0.31已经exit0、独立收集与重算通过，容器
`9891ab947d196bf5349fd158f971b13716b6058a5a607b18534550a63090192d`，报告SHA256
`71c30b779612ad39c3ce14d6f4182650736ba99a3c8828cde8a85810d4384cd6`。
证据`/srv/multimodal-jev-xingchao/operations/attention-kernel-v0.0.31/jev-attention-kernel-v0-0-31-audit.json`。
四种模式的batch1与batch8均精确匹配原HF；batch32均有88个元素不同、最大绝对差0.001953125。
这88个差异与原完整模型batch32的86个差异相差2个元素，不能声称逐位复现原故障。
contiguous/模拟strided V以及inner_precise0/1/2均未改变上述结论；证据支持批量形状敏感，
尚不能确定底层tiling/累积根因，也不能推断所有长度和网络层上batch8都一致。

据此新增独立候选`bounded_prefill_attention`，仅将TND kernel分成最多8个完整请求调用，
保持服务batch32、同一完整请求边界、声学与线性投影批处理、macro128和现有租约。
CPU梯度/样本边界/连续布局与生命周期测试10项通过；未修改已封存或运行的runtime。
完整模型诊断v0.0.32仍使用同一32条有序输入、原HF对照及两种声学路径，
新增worker安装kernel分组包装；包manifest
`be63a1e714e0ef91332977ff07cc2a0e0be65729b700f76b8e4595fd3c36258f`。
原固定后继驱动PID2305083已派发，终态路径
`/srv/multimodal-jev-xingchao/operations/bounded-attention-successor-v0.0.32/terminal.json`。
仍待完整模型独立比较，未承诺修复或吞吐收益；若通过还需实际Stage1终点TRAIN面板及
新runtime两步GRPO、真实尾批、恢复与reference验收，不直接放行正式训练。


## 分组attention的CPU梯度复验（2026-11-17）

在`tests/test_bounded_prefill_attention.py`增加真实CPU因果attention oracle：
9条不等长请求、4个Q头/2个KV头、尾组1条，以随机上游梯度逐值比较分组与独立请求
的Q/K/V梯度及输出，确认边界和autograd保持。该oracle不代表Ascend核梯度数值。
本地4项通过；固定镜像候选v0.0.33准确wait、collect后4项通过、0失败/错误/跳过，
容器`bf41c80c914735e07ec99715f6befb9d66fffa6fc5fac992d5c075206de1acc9`，无NPU。
源码manifest `c2723739aad7ec272168b5355e0cdbadb33bb17bd8dce81b47cf6d4bc5d93d5c`，
报告`/srv/multimodal-jev-xingchao/operations/bounded-attention-cpu-v0.0.33/report.json`。
本次测试变更不改动正在运行的v0.0.32包及其依赖；完整NPU比较仍等待原固定驱动。
定位案例同步至`rl-train-inference-parity-guide.md`第13节，明确已验证与未验证边界。


## batch32完整模型修复案例通过（2026-11-17）

v0.0.32已完成准确wait、117文件collect、传输校验及独立张量重算：
`bounded_attention_full_model_compared_not_admission`，`exact=true`。
容器`01ac13ee126096a7b58e41179b4fc692b60bfb7ba07f6fbcec3a21f54827e074`，
2026-11-17T19:52:40Z至19:59:59Z，exit0、nonOOM。
报告SHA256 `be48f8f830ebb5df20167d2b5846a7ca3301d1a2b391b59f2ca89fadff16dbb9`，
审核`/srv/multimodal-jev-xingchao/operations/bounded-attention-v0.0.32/jev-bounded-attention-v0-0-32-audit.json`。
同一32条历史输入/策略上，原声学与批量声学两条路径的所有捕获激活及单条/batch32
logits、prob精确一致。只证明这个已知失败案例修复，未验证本轮终点、全TRAIN或吞吐。

共享服务新增显式`attention_execution={kind:bounded_prefill_v1,max_requests:8,acoustic_batch_chunks:32}`，
保留service32、macro128、原设备scope校验和LoRA/head切换；省略字段保留旧行为。
代码封包v0.0.34已隔离导入审核通过，462文件，manifest
`8f78bba6c12dbd2f842d6e2b42837a421f89aefc1d3a351336920aacdc06cbbb`，
源码manifest `786c29fc57b4906a3f936e8078b9e702c488c42434578e47776df4c537418662`。
固定来源及四卡日程已绑定到本轮318终点，包
`/srv/multimodal-jev-xingchao/operations/context-stage2-bound-v0.0.34`，manifest
`c5cc783a0b1b906232a6b1aa38a0e205e882ea2c9cdbd19269ebe00b50b37135`。
两服务明确启用该候选，actor继续原micro8数值路径。22项本地绑定/不可变封包/梯度测试通过。
正在生成同版本TRAIN256面板验收包；尚未运行新策略TRAIN parity、reference建库或GRPO。
旧v0.0.18及全部历史运行包未原位修改。


## 实际终点TRAIN256验收封包与固定驱动（2026-11-17）

v0.0.34 TRAIN256验收包已生成：
`/srv/multimodal-jev-xingchao/operations/context-train-parity-v0.0.34-prepared`，manifest
`db7787ed0f18b7cb2247e82668dd07c28077ce156defe29e2fee997afed218eb`。
父包独立审核重建真实318终点、全部40,513条两轮日程、相同runtime源码；
面板使用原冻结TRAIN256输入，macro128、rollout32、两个TP1/service32。
服务显式attention8/acoustic32，HF保持原actor路径。此封包不是NPU结果。

扩展现有`context_train_parity_operations`的`run/detach`：后台进程单次launch、准确wait、
collect后调用`audit_context_train_parity`独立重构，再写`finalization/audit.json`和终态；
失败保留原句柄、不自动重试、不派发训练。固定驱动与原生命周期16项测试通过，
远程/NPU为明确替身，源数据审核仍使用实际代码和来源。
本地宏`.local/config/context-train-parity-operations-v0.0.34.json`，驱动PID2308504，
终态路径`/srv/multimodal-jev-xingchao/operations/context-train-parity-v0.0.34/finalization/terminal.json`。
启动后首先重新校验运行包，记录时未产生NPU容器句柄；不可将后台驱动启动表述为NPU验收通过。


## 派发前设备编号修正（2026-11-17）

对v0.0.34启动器作静态检查发现：容器挂载物理4/5/6/7，服务scope配置visible ordinal2/3，
生成runner却向ASCEND_RT_VISIBLE_DEVICES写入6/7。原控制器PID2308504仍在本地准入审核，
无attempt或container；核对start ticks后通过pidfd发送SIGTERM，确认进程退出，
取消证据`context-train-parity-v0.0.34/driver/cancelled-before-dispatch.json`。
这不是NPU失败，不重写旧包或把中止当作训练错误。

统一`context_device_visibility.visibility`由实际sorted mounted集合生成编号，供TRAIN parity、
reference producer及cached GRPO使用；训练4/5→0,1，推理6/7→2/3。新增测试用真实
`context_configurable_device_scope.validate_mapping`验证偏移设备、非连续设备和排序，
并执行三个生成runner验证实际传给子进程的环境。相关91项CPU测试通过。
旧测试曾只检查字符串6/7，没有与scope校验器交叉验证，现已补齐该区分性检查。

独立v0.0.35验收包与reference包正在生成；模型执行源码继续使用v0.0.34，
仅controller生成的启动器修正可见编号，不改变已封存模型实现。
reference仅准备封包，仍必须先通过实际同endpoint/runtime的TRAIN256准入才能派发。

v0.0.35封包已完成：TRAIN256 manifest
`5f46f523b63b001a192c146d69fe49502cacc1743b64b2294d019b4ce7a7e1b4`，
全TRAIN reference manifest
`351fe1468cefa44f23de36612e33a0dae748b8b9de21cb7e8490a0d76f942abe`。
TRAIN256固定驱动PID2309476已启动并经本地ps核验，启动记录时间1794947128.4499178；
先重新审核来源与封包，再派发、准确wait、collect及独立数值审核。
本地宏`.local/config/context-train-parity-operations-v0.0.35.json`，终态路径
`/srv/multimodal-jev-xingchao/operations/context-train-parity-v0.0.35/finalization/terminal.json`。
这是控制器启动快照，不代表NPU启动或验收通过。reference仅完成准备，未派发；
Stage2训练和任务/迁移收益仍待后续证据。协作入口已去除将八卡历史方案称为最新方案的表述，
统一到当前四卡、简单任务范围。

## TRAIN验收到全量reference的固定后继（2026-11-17）

`context_reference_job_operations`新增`run/detach`，继续复用原launch/collect和独立审核。
可选`predecessor`固定TRAIN parity的driver/config路径及SHA256，并绑定相同host/lease；
后台使用`wait_exact`的pidfd等待原控制器，不增加远程轮询。仅在原终态为
`train_parity_collected_and_audited`且其audit配置完全相同时继续，launch仍重新执行
原数值与来源准入。终态字符串不能替代独立审核。建库后写入`finalization/audit-config.json`、
`reference-output-audit.json`、`audit.json`和`terminal.json`，失败保留句柄、不重试、不派发训练。

本地42项CPU测试通过，覆盖前驱路径/身份/部署绑定、失败与配置篡改拒绝、
launch及collect失败、独立输出审核失败和重复执行拒绝；远程与NPU使用显式替身。
这不代表全TRAIN reference或Stage2通过NPU验收。

本地宏`.local/config/context-reference-job-operations-v0.0.35.json`已配置上述前驱，
后继PID2309779于1794947312.122168启动，现场ps显示在pidfd等待中；
终态预期`/srv/multimodal-jev-xingchao/operations/context-reference-job-v0.0.35/finalization/terminal.json`。
reference尚未派发；后继只允许建库和审核，正式GRPO仍受两步、尾批、恢复及效率准入约束。

## 缓存GRPO有限验收固定串接（2026-11-17）

新增`tools.reports.context_cached_acceptance_sequence`，模板
`configs/portable/operations/context-cached-acceptance-sequence.json`读取
`${CACHED_ACCEPTANCE_SEQUENCE_CONFIG}`。配置仅声明固定reference前驱及三组
`package/remote_root/operations_root/tag`，不开放任意训练阶段、步数或命令。
驱动先核对reference前驱driver/config路径及hash，经准确pidfd等待后重新独立审核
TRAIN parity、reference数值来源、完整人口与传输，再顺序调用既有prepare/launch/audit：

1. `prefix`：连续0→2更新，固定完整TRAIN reference，service32、macro128、DP2。
2. `tail`：现有真实尾批窗口65真实+63零权重padding，再128真实；不冒充正式训练步。
3. `recovery`：从已审核prefix的第1步新进程恢复到第2步，逐值比较状态与输出。

生成包保持同一封存模型runtime；reference只读，checkpoint只保存adapter/head/optimizer。
任何失败立即停止下游、保留句柄、不重试；已有目录拒绝复用。固定同一host/lease，
校验全部新路径彼此隔离且不覆盖reference及源包。各阶段finalization保存解析配置、
独立audit输入、结果及终态。全部通过仍只标记`cached_finite_sequence_audited`，
正式训练不自动派发，效率对照与任务/迁移评测仍待完成。

72项本地CPU回归通过，覆盖前驱身份、失败阻断、路径隔离、准确审核口径、
普通/尾批/恢复封包与共享生命周期；远程及NPU是明确替身。首次测试命令引用了
不存在的测试文件，pytest未运行；改用现有`test_context_cached_recovery.py`后
执行上述72项，没有将未执行测试记为通过。

本地宏`.local/config/context-cached-acceptance-v0.0.36.json`，后继PID2310461，
启动时间1794947632.5070798。现场ps显示其在pidfd等待reference控制器；
终态预期`/srv/multimodal-jev-xingchao/operations/context-cached-acceptance-v0.0.36/terminal.json`。
此时三组NPU验收尚未派发，不据CPU通过或队列登记宣称NPU准入、效率或任务收益。

## v0.0.35端点校验失败与v0.0.37修复（2026-11-17）

TRAIN验收容器`e9279c86af91891b86253300d7f8b911c7d72d9ec9d6920794d0f83aba2667a1`
于20:33:43Z启动、20:33:53Z exit1/nonOOM，现场inspect已确认。HF日志显示
`validate_endpoint`调用封存`context_topology_batch_plan.derive`时拒绝Stage1的
`inference_instances=0`，未进入模型前向。原日志和收集产物位于
`/srv/multimodal-jev-xingchao/operations/context-train-parity-v0.0.35/`。
reference与有限验收后继均停止于失败前驱门禁，三个控制器现场ps确认退出，未重启旧任务。

根因是封包继承旧topology模块：当前源码支持0推理实例，但旧包只允许正数；
模块存在时依赖闭包默认保留旧版。将该模块明确列入新runtime覆盖入口，
隔离审核新增Stage1无推理实例的318端点契约实际调用。
33项相关CPU回归通过；新增单测先因旧父夹具stub缺少`file_hash`失败，
改为从封存文件提取真实validator并导入真实topology，完整隔离导入由实际包审核覆盖。

v0.0.37 runtime已审核462文件、finite/formal/tail生成及端点契约：manifest
`ec517620f0a3675866d4c6a9c48ab6e09be8407f89c7927bce7bccab7bbe8c73`，源码manifest
`77d9bd0b4aefb9966b643157f9a8b37e47d7b1bab528392fbe9b4bbbe98d1efc`。
与v0.0.34逐文件比较仅`src/multimodal_jev/context_topology_batch_plan.py`改变。
从新封存目录实际加载旧TRAIN包中真实318 commit/config，端点校验通过（无模型前向）。
新端点绑定正在准备，旧包/失败记录保留；Stage2依旧未准入。

下一效率验证口径见[缓存GRPO效率协议](../../experiments/context-cached-grpo-efficiency-v1.md)：
每配置100步，全部100步及固定剔除前5步的95步并列，单独报告reference建库和冷启动成本；
旧初始化/旧runtime的74.76秒等历史数字不充当本轮匹配对照。

v0.0.35失败归档已逐文件重算12文件及tar SHA256，均与传输收据一致；
这是失败证据完整性确认，不是数值准入。v0.0.37端点绑定已完成，manifest
`238927a47ceeb8c9c15aa8ece142e3ba349635362db2ec9b65b434f2268b9725`。
新TRAIN256与完整reference包分别使用本地宏`context-train-parity-v0.0.37.json`、
`context-reference-job-v0.0.37.json`准备中；各自重审父包和真实来源，不复用旧数值结论。

## v0.0.37封包完成与后继登记（2026-11-17）

TRAIN256包manifest `1d7968dcb0743e0b73a3148d98847b0a08fe52cb0e8b11075671fdec7a421332`，
全TRAIN reference包manifest `16b149debb91bec43af6ecd95b88820bef97bf849845d8016b76495941957e98`。
两者绑定相同318终点、runtime和输入来源，服务visible ordinal2/3、batch32、
attention最多8完整请求、声学chunk32。模型源码未再变动。

固定驱动依次登记：TRAIN验收PID2312751（1794948498.8380868），reference后继
PID2312871（1794948524.7664638），有限GRPO序列后继PID2312960（1794948545.996983）。
本地宏依次为`context-train-parity-operations-v0.0.37.json`、
`context-reference-job-operations-v0.0.37.json`、`context-cached-acceptance-v0.0.37.json`。
启动时TRAIN控制器处于本地准入审核，后两者等待准确前驱；不代表NPU启动或审核完成。
三个结果入口均在对应operations根的finalization/terminal.json（有限序列为根下terminal.json）。

控制器审核另发现有限/正式提交格式不同：有限生成器只写
`checkpoint-NNNN-committed.json`（含rank_shards）；正式生成器才写独立
`policy-NNNN-committed.json`。原通用审核器无条件读取后者，会错误拒绝有限训练。
新增`published_policy_commit`按实际阶段读取，有限路径必须有非空optimizer分片，
后续分片hash、contract、BF16执行/adapter、激活与恢复检查保留。正式路径不回退；
不改写收集产物或封存模型runtime。此前失败队列已终止，新队列使用此控制器版本。
55项CPU回归通过，其中测试执行真实生成代码的rank0提交块并原样交给审核器，
覆盖缺失分片拒绝和正式路径不得回退。NPU有限验收仍等待上游通过。

独立100步候选`context_cached_efficiency`已实现prepare/audit-package/validate/launch/audit，
沿用同一封存训练器和readonly reference；三项有限验收必须逐一重审，不能只凭队列终态。
窗口固定0→100、micro8、DP2+2、每步trainable optimizer checkpoint；完整结果审核复用
逐步schedule/credit/行为一致性/导出/激活检查，逐样本核对reference、零在线补算。
效率候选与成本统计、提交格式组合54项本地CPU回归通过，实际NPU前驱为替身，
尚未生成真实100步包或派发。配置字段和结果边界见上述效率协议。

v0.0.37 TRAIN验收准确句柄
`857c118969e724115e89d1fa3b14fcd42d27075697dc7e3ac43e912e57d74e80`已产生。
2026-11-17约20:59Z现场inspect确认固定镜像，20:56:28.85473286Z启动，
running/nonOOM，宿主PID1533493。这是启动身份观察，不是数值通过；原控制器继续
准确wait、collect与独立审核，后继仍等待。下一普通远程观察不早于15分钟后。
