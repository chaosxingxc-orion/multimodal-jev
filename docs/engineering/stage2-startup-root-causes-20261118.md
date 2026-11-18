# Stage2 启动根因分类与系统自检

2026-11-18。先分类、再自检、再修复、最后验收；本文件不是 NPU 准入证明。
范围仅限四卡简单任务：Stage2 DP2 + 两个 TP1 服务、可编辑双信用 GRPO。
不改变数据、奖励公式、精度、Stage1 初始化或固定 reference。

## 1. 证据与实际边界

已核对 v77/v79 终态与原 PID：两个控制进程均已退出。v77 的 actor 在首次训练更新前
失败；v79 因前驱失败而停止，不能描述为仍在排队或正式训练中。
Stage1 完整曲线 v76 的独立 CPU 审核终态为 `complete_infonce_source_curve_audited`。
这些是本次核对的历史运行结果，后续 live 状态仍须现场检查。

证据根 `OPS=/srv/multimodal-jev-xingchao/operations`（证据位置，不是共享配置默认值）：

- `context-stage2-parallel-v0.0.77/terminal.json`：上层错误为 `actual isolated terminal runtime differs`。
- `context-objective-prefix-v0.0.77/jev-objective-prefix-v0-0-77-collected/supervision/actor.log`：直接根因为 `parent package imported outside sealed source: tools.reports`。
- 同目录 `terminal.json`：actor exit1；两个服务均为 `oversized close acknowledgement`、exit1；非强制清理。
- `context-stage2-objective-pipeline-v0.0.79/terminal.json`：前驱失败，后继停止。
- `context-stage1-complete-curve-v0.0.76/terminal.json`：完整曲线审核成功。
- `context-audit-proof-v0.0.80/terminal.json`、`v0.0.81/terminal.json`：审核期间依赖变化，拒绝复用。
- 时延测量与采样方法见 [审核时延证据](stage2-admission-latency-20261118.md)。

## 2. 分类及因果关系

| ID / 优先级 | 分类 | 已确认原因、性质 | 自检与修复方向 | 验收条件 |
| --- | --- | --- | --- | --- |
| C1 / P0 | 封包与执行入口 | **直接故障**：直接执行 extension/bootstrap.py 会把 extension 根加入 sys.path；namespace tools.reports 同时包含扩展路径，触发 sealed-source 保护。此前 `python -c` 测试没有覆盖真实入口。 | 新包固定安全搜索路径；bootstrap 只移除自己的扩展根；用真实文件入口、torchrun 子进程、污染路径负例验证。模型加载前执行同环境 preflight。 | 宿主、固定镜像的封包入口均通过；外部模块注入仍拒绝。 |
| C2 / P0 | RPC 与进程生命周期 | **连带故障**：服务先发送 hello，旧 supervisor 把 hello 当 close ack，并用 4 KB 上限拒绝，继而 BrokenPipe。 | 复用有界 framing，辨认 hello 后才验证 close ack；保留 actor 主因与 cleanup 次因；验证部分启动、actor 失败、超时与孤儿进程。 | 大 hello、非法帧、二次 hello、断连均有确定终态；失败不重启训练、不吞主错误。 |
| C3 / P1 | 审核依赖图与复用粒度 | **时延原因**：lease 前后重复全量 validate；package/reference/parity 嵌套重建 Stage1；原复用仅覆盖 batch，不覆盖完整来源证明。 | 显式证据 DAG；每个内容身份全量计算一次，进入 lease 后重新验证全部依赖内容；采用现有 singleflight / release handle，禁止仅凭状态或 mtime 复用。 | 全量输出与旧审核逐字段相同；篡改/新增/删除/替换路径必须失败；记录计算次数与复用次数。 |
| C4 / P1 | CPU 工作量与并行 | **时延原因**：40513×60 候选反复构造，asdict/deepcopy/JSON 多次遍历相同 ontology；部分审核仍串行。 | 复用不可变 ontology 编译结果；逐音频身份、全部候选覆盖和张量证据仍检查；I/O、CPU 池从配置读取并有界。 | 同负载全量差分、宽度网格一致，实测墙钟/CPU/峰值内存；不按小样本外推生产收益。 |
| C5 / P0 | 源码身份与协作隔离 | **流程故障**：v80/v81 在可变工作树中运行，审核期间源码被修改，正确触发身份变化拒绝。也包括本次开发自身。 | 测试前复制独立源码快照、生成完整内容清单；验证与工作树编辑分离；开始/结束都校验快照，输出写到快照之外。 | 工作树变化不影响快照；快照自身变化则失败。依赖范围明确，不把 CPU 证明转授给另一组字节。 |
| C6 / P0 | 组件及发布纪律 | **系统性放大因素**：多层字符串源码改写使入口、环境、封包、审核互相漂移；本轮未发布 context_audit_release 草案也有此问题。 | 撤销草案的发布资格；控制器只组合显式接口，runner 从配置驱动，不改写已有源码；历史冻结包不改。 | 新发布入口 AST 扫描无 getsource / 源码 replace / exec；所有阶段共用生命周期，未覆盖项显式阻断。 |
| C7 / P0 | 验收覆盖与错误可见性 | **检测缺口**：fixture 成功未覆盖真实启动；外层泛化错误掩盖 actor 主因，缺阶段计时。 | 分类→检查 ID→证据→状态一一对应；固定镜像实际入口测试；结构化 primary/secondary error 与阶段耗时。 | CPU、固定镜像、有限 NPU、正式训练四级证据分开；缺项 not_exercised，不能汇总为发布 PASS。 |

因果链：C3/C4 导致审核慢；C1 导致 actor 启动失败；C2 放大退出故障；C7 让定位变慢。
C5/C6 导致修复难以稳定复现。没有证据把此次失败归因为 NPU 容量、BF16 数值、
GRPO 算法或镜像本身；processor model_type 警告也不是已证实的直接退出原因。

## 3. 按类别执行，而不是按报错重试

1. P0：固定源码快照与检查配置，保留旧失败现场；新候选未发布，不派发 NPU。
2. C1/C2：检查真实文件启动、完整协议和失败收尾；CPU preflight 必须早于服务加载。
3. C6/C7：检查 prefix、recovery、tail、efficiency、formal、evaluation 各入口，区分适用项。
   所有入口共用组件，不能只修 prefix。新增代码不能依赖字符串改写，继承债务单独列示。
4. C3/C4：在冻结源码上做真实全量 source/reference 差分及复用计时；不减检查、不采样。
5. 固定镜像验收同一封包的真实入口与故障路径，再做 4 推理 pass / 2 更新、恢复及尾批。
6. 独立 collect + 数值审核通过后进入正式实验。2026-11-18 用户取消独立效率基线，改从正式训练固定窗口统计效率；不追加100步前置实验。

## 4. 当前修复状态及禁止外推

工作树中已有其他协作者对 bootstrap 和 supervisor 的修改，须复用并验证，不能覆盖。
旧运行包保持原样。本轮新草案的 29 项宿主测试不构成发布资格，尤其不能证明 C5/C6 已解决。
v81 中 source 首次 99.17 秒、复用 2.01 秒且输出相同，仅证明该子项；reference 失败，
不能声称整条审核链加速完成。每项最终状态以分类自检报告为准。

共享检查配置与脚本将输出 `verified_cpu` / `failed` / `not_exercised`，固定镜像和 NPU
证据另行绑定。日常运行不需要模型裁决；模型本次只承担用户明确要求的工程根因分析与修复。

## 5. 分类自检入口与本轮证据

脚本 `tools.reports.startup_category_selfcheck`，模板
`configs/portable/operations/startup-category-selfcheck.json`。本地宏集中提供
`PROJECT_ROOT / STARTUP_SELFCHECK_OUTPUT / VERIFICATION_PYTHON /
VERIFICATION_SUPPORT_PATHS / VERIFICATION_SCRATCH_ROOT`。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src:.local/verification-python:tools/jev \
/usr/bin/python -B -m tools.reports.startup_category_selfcheck \
  --config configs/portable/operations/startup-category-selfcheck.json \
  --local .local/config/startup-category-selfcheck-v0.0.85.json
```

上述版本已执行，重复执行会拒绝覆盖；后续执行使用新 output 和本地宏。
脚本先冻结独立源码，I/O 哈希走配置化线程池，测试由配置化有界独立进程执行，
每进程单线程；日志和临时文件在快照外，开始/结束验证完整内容身份。
测试超时清理仅限其持有的进程组。报告明确 `release_ready=false`，不会派发训练。

v0.0.85：**74 项宿主测试通过**（C1 入口4、C2 生命周期/runner42、C3证明10、
C4输入11、C5快照7）。C6静态检查仍明确拒绝旧的未发布 `context_audit_release` 草案，
因此整份自检终态 exit1 是预期的发布阻断，不能写成“全部验收通过”。
证据 `OPS/startup-category-selfcheck-v0.0.85/report.json`；源码清单与每类日志同目录。

新增 `context_stage2_runtime_runner` 以普通函数统一 prefix/recovery/tail/efficiency/formal
窗口、配置拓扑和先 preflight 后启动服务的顺序；尚未接入正式封包，evaluation 不消费
训练 actor/reward 扩展，仍需要自身实际入口验收。旧草案派发入口显式拒绝执行。

新增 `context_composed_reference_audit` 将完整 reference 校验改为显式函数接口，保留
原始身份、候选覆盖、policy、input receipts、journal 全量检查。真实全量 source/reference
差分 v0.0.86 从独立只读源码快照启动；未取得成功终态前不声明审核加速完成。
固定镜像 CPU 回归版本 v0.0.87 使用 v0.0.85 冻结源码，结果与真实 NPU 证据分开。

自检自身的失败也归类并保留：v82 的长日志目录触发 AF_UNIX 路径长度限制（C2/C7）；
v83 的短路径不可写、v84 的 /tmp 为 tmpfs，被存储门禁拒绝（C5/C7）。v85 将短 scratch
根通过宏绑定到已经校验的本地 ext4 `/var/tmp`；没有放宽存储门禁。

## 6. 本轮修复后的复核

v0.0.88：**81 项宿主测试通过、三组静态检查通过**；源码快照、测试命令、计数、
逐类日志见 `OPS/startup-category-selfcheck-v0.0.88/report.json`。
移除了本轮 `context_audit_release` 的源码字符串改写，改为装配普通 runner 文件；
原快速派发器已退役，完整派发图仍显式拒绝未准入执行。历史训练器与下游封包中的
改写债务没有因此消失，完整各阶段集成仍为 `not_exercised`。

v0.0.86 真实全量差分终态 exit0，`full_source_reference_proof_reuse_verified`：

| 完整证明节点 | 首次计算 | 同进程内容重验后复用 | 等价性 |
| --- | ---: | ---: | --- |
| Stage1 source | 100.0649 s | 1.8428 s | 与冻结 source plan 逐字段相同 |
| 全 TRAIN reference | 68.6807 s | 0.5056 s | 与原独立 reference audit 逐字段相同 |

输入规模未缩减。开始/结束认证源码快照，每次复用重新哈希依赖；事件顺序为
`miss, hit, miss, hit`。证据 `OPS/context-audit-proof-v0.0.86/report.json`。
这是同负载单次配对的两个 CPU 节点测量，不是整条准入链、多次生产基准或训练吞吐。
v88自检中的全量差分仍记 `not_exercised`，因为它未消费v86报告；不能手改脚本结果为通过。

证明捕获器当前只用于显式注册的本地确定性审核。Python audit hook 不自动覆盖
任意网络、外部进程、原生扩展或文件描述符读取；共享进程并发捕获及父路径替换也需
独立安全契约。上述全量差分成功不授权将它包装任意函数，或直接替代 lease 内全部
依赖认证。这些是 C3/C5 发布门槛，不能用已有测试数掩盖。

下一发布顺序：冻结完整各阶段组合配置 → 绑定固定镜像实际入口/协议结果 → 验证
package/lease内容身份与传输 → 有限 NPU 更新、恢复、尾批与独立审核 → 正式训练。
当前没有发布新的 rc，也没有产生新的 Stage2 正式训练指标。

固定镜像 v87 已 collect：72通过、2失败。失败均为入口测试 `torchrun --standalone`
在 `network=none` 环境解析容器hostname超时（C7），没有进入待测 bootstrap。
生产runner本来使用静态 `--master-addr=127.0.0.1 --master-port=29617`，因此修复的是
测试与生产启动参数的一致性，不能据此归咎镜像或增加网络权限。新快照 v89 使用相同
静态rendezvous并执行81项回归；旧包未修改、未重命名，报告原样保留。

发布前还须将证据绑定到**同一最终快照**：v86全量差分和v89镜像回归并非同一源码清单，
前者之后修改了未发布控制器。不能把两份成功证明直接拼接为最终准入；最终封包需
重新绑定其实际依赖的精确身份。原开发派发器现仅保留显式拒绝入口，其旧实现已在
v85/v88等诊断快照中保留，避免工作树存在看似可运行的废弃路径。

v89最终已独立 collect + 审核：**固定镜像81项通过，0失败/0错误/0跳过，exit0、nonOOM**。
容器 `d8b1ad8f00b3e9dd00f65acb7fc6207b5b1115e2edc9be6efb907b3eeb6f73dc`，
2026-11-18 08:46:09–08:52:58 UTC；报告 `OPS/context-startup-cpu-v0.0.89/report.json`。
此次确实执行了真实脚本/torchrun子进程入口和异常路径，但模型与训练配置部分仍用
fixture，容器无NPU设备；不构成完整生产封包或NPU数值准入。

## 7. 组合式启动与正式训练内效率统计

当前候选为 v0.0.91，入口 `tools.reports.run_composed_stage2`，模板
`configs/portable/operations/composed-stage2.json`，本地宏
`.local/config/composed-stage2-v0.0.91.json`。冻结源码清单为
`c7e124900c1650a9a7c5a0bda0b59b5c948a1956a595a5b48ec67660d9294b16`。
控制进程已后台派发；准确启动身份及日志位于
`OPS/context-stage2-composed-v0.0.91-driver/`。不要重复执行 detach。

执行顺序为 prefix两更新 → 新进程恢复 → 真实尾批两更新 → formal634。
独立效率100步不再执行，也不作为正式训练准入条件。每阶段仍由脚本收集并独立审核，
失败保留准确句柄和现场，不自动重启；普通监控间隔900秒。

固定镜像证据：v90共91项通过；v91对限制部署命令替换范围的修复追加10项通过。
控制器分别绑定v91控制源码及v90实际启动组件的精确字节，不将不同快照一概视为等价。
v91实际prefix封包已独立重建成功，forward与reference保持原身份；
该CPU结果仍为 `objective_training_candidate_reconstructed_not_admitted`，不是NPU验收。

效率读数来自正式训练更新1–100、268–367、535–634，首窗口前5次更新另列为warmup。
`context_formal_efficiency`读取完整rank计时及双服务激活回执，分母为真实query曝光，
尾批padding不计入吞吐。报告步耗时、actor/rollout耗时、actor峰值内存及
导出/checkpoint/同步合计残差；嵌套耗时不相加。当前控制器在正式训练收集审核后
生成三个采样报告；训练期间保留原始逐步日志和900秒进度观察。
这些窗口不能单独证明代码加速的因果关系，也不包含启动、离线reference、评测成本。

现场启动记录：prefix容器
`e4fe949aae00809881e30819ef82b053c2f177391f687a9844907195cf2f9ff6`
于2026-11-18 10:31:58 UTC启动；随后inspect为running/nonOOM，真实actor入口
preflight通过，两个推理服务进程已启动加载。此时尚无更新完成证据，
更不等于formal634已开始。准确句柄位于
`OPS/context-stage2-prefix-v0.0.91/jev-stage2-prefix-v0-0-91-launch.json`。

后续终态复核：该容器于10:38:46 UTC正常exit0、nonOOM，两rank均有两次更新记录，
共256条query曝光；控制进程已退出。独立审核在objective trace CPU重放阶段失败：
`updated or numerically inconsistent behavior: existence`，因此恢复、尾批、formal均未派发。
这是C7审核数值后端覆盖缺口，不能将容器exit0写成独立验收通过。
只读重算全部256条轨迹：actor与behavior的logit最大差为0；CPU重算sigmoid与
记录概率最大差1.1920928955078125e-7，阈值翻转0，但log概率最大差
0.000331878662109375，67条超过原log-gap门限2e-6。靠近1的概率经log(1-p)
放大微小后端差异，触发CPU审核失败；尚不能据此宣布全部loss/metrics审核可通过。
后续应版本化修复审核的后端契约，保留真实actor/rollout对齐门限及原始轨迹，
先重新审核现有产物，避免为审核器修复重复训练或直接放宽训练侧门限。

## 8. 同后端重放修复 v0.0.92

审核器新增显式 `replay_device`。重放时仅将记录的logits送回NPU，behavior/reference
仍保持CPU张量，与训练实际计算路径一致；reward、loss、metrics均保持原严格比较，
本轮 `loss_atol=0`。训练器、模型前向、奖励插件和在线一致性门限不修改。
后端重放使用同镜像、同设备租约、单卡独立容器，原始轨迹和审核源码只读挂载，
不加载模型或执行optimizer。重复审核须核验输入、源码及回执内容身份后才能复用。

真实v91两步256条轨迹已在固定镜像NPU重放通过：32个microbatch，loss与metrics
精确一致。证据 `OPS/context-backend-trace-probe-v0.0.92/report.json`；容器
`5852015f6f2e08b0298024eb87c046f9a645b8d671f71f285f01cb0564d4f9ad`。
这是objective公式重放，不单独替代完整source/transport/optimizer审核。
同镜像回归33项通过，报告 `OPS/context-stage2-release-cpu-v0.0.92/report.json`。

后继控制器v92复用v91原prefix封包和已收集产物，从独立重审继续，不重复prefix训练；
新恢复、尾批和formal包分别版本化。准确驱动记录位于
`OPS/context-stage2-composed-v0.0.92-driver/started.json`。下游完成情况以实际终态为准。

v92已产出 `prefix-result.json`，完整独立审核状态为
`finite_objective_training_outputs_audited`，reference在线调用/策略切换均为0；
不仅是单独trace probe通过。新恢复容器
`290d0c5214a4db7ac0e5924170f6ca4cf73a281c131b0a7933ac6ae3d8026be8`
于11:13:40 UTC启动，入口preflight通过；此启动记录不等于恢复数值验收通过。

前三次正式loss由 `tools.reports.watch_stage2_first_losses` 事件监听脚本采集，
配置模板 `configs/portable/operations/stage2-first-losses.json`；只读且不重启训练。
只有正式容器、两个rank的三次更新及两个推理服务的激活回执齐全才发布报告。
microbatch loss先逐rank求和，再对DP ranks取均值；不把已归一化microbatch再次取均值。
该监听不是模型调用，也不替换900秒普通状态检查。宿主2项聚合及缺失证据负例通过。
