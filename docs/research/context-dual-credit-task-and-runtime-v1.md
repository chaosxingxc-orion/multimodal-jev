# 排序与确定性双信号：任务协议与统一工程发布

2026-11-16。用户要求先定义任务，再锁定统一工程，最后训练；优先优化大块效率。
本协议和 `configs/experiments/context-dual-credit-v1.json` 是新版设计入口。
设计已定稿，运行准入仍为 false；不能将本文件或 CPU 检查当作 NPU 发布成功。
原 arithmetic-v2 GRPO/RLOO 是对照证据，现有后台任务按原配置收尾，不原位改动。

已有 GRPO 固定终点的全 selection 分解见
[排序与接受／拒绝诊断](context-grpo-rank-existence-endpoint-v1.md)：漏检减少但误接受增加，
pair accuracy 略降，正负 NLL 均恶化。该证据支持分项观测的必要性，不证明本设计有效，
不改变本协议的任务、权重、阈值或准入规则。

## 1. 任务边界先固定

输入为原始音频及 caller 提供的完整候选定义，输出仍为共享上下文 Transformer head 的
独立 sigmoid 概率。无 ASR→文本模型，无生成 CoT，无候选特定分类 head。
当前主任务的标签表示 benchmark event membership；不是语义存在性、执行成功概率或
模型对自身正确性的确信度。本文“确定性”指有证据的接受/拒绝及概率可靠性，不指强制
概率逼近 0/1。单正例 SLURP 不证明开放世界 unknown、多意图或任意规则理解能力。

已有全量 TRAIN40513、selection9226、同 Stage1 step634 起点、两轮634更新、query
均匀曝光和零权重 padding 不变。缺标不填负例。真实标签不进入模型/rollout 请求，
只在训练端构造 reward。候选对选择使用标签属于训练监督，不在推理输入中泄漏答案。

## 2. 两个 RL 信号，一个输出头

令 head 原始 logit 为 z_j，p_j=sigmoid(z_j)。两种动作共享 z，分别记录行为概率、
动作、old log-prob、reward、advantage；不把两个信号加在原60-bit联合ratio上冒充节点信用。

### 排序分支

所有已知正负候选对都参与，每对以候选稳定ID确定左右顺序，而非总把正例放左边。
q_ij=sigmoid((z_i-z_j)/T)，T=1；b=1表示左候选优于右候选。每对G16个Bernoulli样本，
reward为是否选对；同正/同负对不建立无依据的排序。先平均G，再平均有效对，最后平均
真实query；没有有效对的query贡献0但保留曝光分母。不根据最终TEST挖掘困难对。
共同平移logits不改变此分布，因此该分支主要约束相对差距。共享网络的参数更新仍可能
影响绝对概率，不能声称两个分支参数梯度正交。

### 确定性分支

每候选从实际p_j采样G16个二元接受/拒绝动作，reward为判决是否与标签一致。
每query正例候选平均贡献1/2、负例候选平均贡献1/2；若一侧为空，另一侧占全部权重。
不按59个负例压倒1个正例，也不复制正例改变query曝光。
该效用刻意平衡两侧，不是原生出现率加权的校准目标：二元决策reward倾向端点，
不能仅凭它宣称p已校准。另保留原生候选均值binary NLL=0.3及排序监督=0.1，暂不衰减。
辅助项与旧版本不同，历史GRPO/RLOO仅为整体方案参考，不能隔离双信用的纯因果贡献。

### RLOO 主配置与 GRPO 对照

对同一query同一组件的G个reward独立计算：

```
A_rloo[g] = R[g] - mean(R[h], h != g)
A_grpo[g] = (R[g]-mean(R)) / max(population_std(R), 0.25)
rho[g] = exp(log_pi_current(action[g]) - log_pi_behavior(action[g]))
```

分别使用原生verl clipped surrogate（clip0.2、dual-clip3），组件加权后query归一化。
两分支权重各0.5，不再对合并优势白化。GRPO方差下限是本版本的明确改造，不称原生GRPO；
RLOO是组件级clipped适配，也不称原联合动作RLOO原样复现。无critic、无temporal GAE。
完整组先算优势再分DP。每批只有一次更新；clipping不保证单次optimizer移动满足KL上限。
组内恒reward仍会产生0优势，双信号不会自动消除饱和；记录正确/错误恒组及分支梯度。

合并目标：
`L = .5 L_rank_RL + .5 L_exist_RL + .1 L_pair_supervised + .3 L_binary_NLL + .01 KL`。
KL仍对候选Bernoulli分布逐位求和、每query一次；不重复加两份KL。
分支分别统计loss、reward、优势尺度/零比例、梯度范数及共享参数梯度cosine；诊断在固定
TRAIN窗口执行，正式loss仍一次backward。统计开销计入成本，不每步加多份反向。

现有vLLM只输出p，不能悄悄用logit(p)重建z并宣称与actor z_i-z_j完全相同。
新协议需传回实际head logits，验证sigmoid(logits)与原p，并统一FP32差值/sigmoid及
CPU log/log1p路径；两分支均执行更新前行为一致性校验，端点拒绝、不静默clamp。
logits仅为训练端内部附加字段，产品仍消费prob。它属于同一engine接口的版本化扩展。

## 3. 固定比较，避免边跑边改

先用RLOO改造完成同一工程上的ranking-only、existence-only、joint三臂。
单分支臂只关闭另一RL项，保留活跃权重0.5及相同辅助项，避免重新归一化造成额外混杂。
三臂同起点/日程/G/更新数，报告实际不同计算成本；不启动RL-vs-SFT比赛。
GRPO-floor作为同接口配置，在统一工程验证后另行冻结实验；不同时扩建critic-PPO。

继续11个完整selection点及旧选模规则，主指标set-F1@0.5；补充Top1、正负对排序准确率、
误接受/漏检、exact set、NLL/Brier正负分项及风险—覆盖。固定step634用于预先指定的
endpoint判断；按source group配对bootstrap2000/seed317。联合成功要求set-F1差值95%
区间下界>0、Top1下界>=0、NLL和Brier上界<=0。这是严格工程准入规则，不能把逐项区间
说成联合95%覆盖；一个seed也不证明种子稳定性。若要论文显著性，再锁多重检验/多seed协议。
joint优于起点不等于双信号协同；必须看joint与两个单分支同协议的对比。
阈值和目标集不用于调reward/权重。无明确收益继续简单任务，不先开复杂Q。

## 4. 一套工程与三个大效率改动

沿用固定52ba94e1镜像、Qwen-ASR0.6B、q/v LoRA r4、共享一层head、BF16执行/FP32优化器、
verl DP4/vLLM TP1、global128/micro8/accum4。不另建训练器；在现有batch契约、
优势/目标接口和service批处理接口上扩展。共用保存、发布、恢复、评测、输入与日志。
部署路径/机器/设备/cache/输入身份全部由本地宏绑定，发布时生成不可变解析配置及pins。

首个性能版本只包含：

1. bulk RPC + 真正vLLM合批，初值batch4/token16384，按完整文本/音频长度分组，保留顺序
   恢复、全量覆盖、padding零权重；复用既有批量候选，禁止只改max_num_seqs就称完成。
2. 同策略/processor/输入/runtime身份下共享固定reference缓存，冷建与热复用分别计费。
   当前behavior每次更新必须新算，不能共享旧behavior。已知输入前处理缓存不等于可缓存
   可训练模型的hidden，不能直接跳过音频tower而不检查冻结状态。
3. 批量输入传输和预取，减少每query落盘/hash/RPC；持久checkpoint仍保留本地fsync。
   先复用批量文件协议，不立即再造共享内存后端；只有实测传输仍是主要瓶颈才立下一版。

不改旧运行的microbatch、不同时扫描学习率/温度/组大小、不引入异步过期策略。
首版只做一次整合候选和一次验收周期，失败仅按已定位根因修复；不因微小计时波动反复调参。
升级条件为同workload端到端训练吞吐至少1.25倍，并满足完整数值与恢复要求；这是目标，
不是承诺。纯推理probe的通过不允许直接替换正式训练profile。

## 5. 发布顺序与验收停止点

| 顺序 | 产物 | 通过后允许的动作 |
|---|---|---|
| 任务定义 | 本协议+双信用配置 | 开始有限工程实现，不训练 |
| CPU数学与契约 | 枚举小空间梯度、奖励/优势独立复算、DP/micro归一化 | 固定镜像验证 |
| 真实音频推理 | serial/batch概率、排序分支、顺序和G采样；实际forward/内存/耗时 | 有限真实更新 |
| 连续更新与恢复 | 非零LoRA+head同步、两步与新进程第二步重放 | 封存唯一发布包 |
| 工程冻结 | image/source/config/input/policy pins，明确兼容范围 | 全量三臂实验 |
| 效果判断 | 全曲线、固定终点、冻结TEST/迁移及成本 | 决定下一算法版本或复杂任务 |

不得把有限两步当效果训练；它只用于运行准入。通过上述工程门槛即开展全量实验，
不继续增加无关验证成为无限准备阶段。冻结后除失败修复外不改代码/配置。

当前真实证据：旧GRPO634更新和11点selection均完成，主指标选step384；RLOO后台收尾中。
新版尚无loss接入、NPU更新或收益结果。批量推理已有封包，尚未实际NPU验收。
下一动作是实现组件动作契约与loss，在同一基座上接批量服务；不是启动新模型训练。

## 6. 固定效率统计入口

`tools.reports.context_rollout_efficiency_summary` 只读完整归档的四rank634步，不访问远端。
共享配置 `configs/portable/operations/context-rollout-efficiency-summary.json`，通过本地宏
提供 EFFICIENCY_RUN_ROOT、EFFICIENCY_REPORT。输出全部来源hash、整轮耗时分解和参考miss。

```
PYTHONPATH=src:. python -m tools.reports.context_rollout_efficiency_summary \
  --config configs/portable/operations/context-rollout-efficiency-summary.json \
  --local .local/config/dual-credit-efficiency-v1.json
```

计时口径：total取每步最慢rank，其余阶段rank0；reference属于rollout，rollout属于
preparation，不能累加；不把归档timer当纯kernel耗时或新代码加速比。
原运行中的socket文件持久化、hash、纯前向、切换没有全部独立计时；新验收需补齐，
但不因此忽略已确认的串行推理和重复reference这两个大项。

## 2026-11-16 首轮实现证据

`src/multimodal_jev/context_dual_credit.py`新增两分支组件契约、固定ID排序、逐组件独立
采样、RLOO/GRPO-floor优势、完整receipt复算、更新前概率/log一致性检查，以及调用固定
verl surrogate的RL与混合loss接口。没有修改原actor、service或冻结包；尚未接入训练器。
排序差值和sigmoid明确在CPU FP32对实际head logits计算，两端用同一路径；存在性继续
使用各自前向产生的FP32 sigmoid，并按既有门槛核验。不能从已舍入prob反推logits。

本地27项CPU检查通过，2项原生verl检查因宿主没有verl/omegaconf而跳过。19项新增契约
检查包括候选置换、平衡权重、同类无排序对、奖励/优势/行为记录篡改拒绝、过期身份拒绝、
梯度连通及全部G2动作枚举所得RLOO梯度与解析期望一致。另8项为既有候选辅助目标检查。
原生surrogate数值/梯度与DP-micro归一化测试已写，但跳过不是通过；下一步必须在固定
镜像真实verl执行。测试中的DP对比为代数分片，不能替代真实通信与两步恢复。

明确拒绝边界：单候选p未饱和不保证差值sigmoid未饱和，例如logits为10和-10时排序概率
可在FP32舍入成1。现实现拒绝而非clamp。需对真实TRAIN完整logit面板验证覆盖；如果
失败，须修订统一排序分布数值表示并重新冻结，不能删掉困难候选、调整阈值或称已准入。
排序恒reward依然可能无信号；此实现没有悄悄换成条件积分/精确期望算法。

全量效率报告已由固定脚本生成：`docs/engineering/context-full-rollout-efficiency-v1.json`，
2536份四rank更新记录逐个hash。第一/二轮最慢rank均值101.137/67.708秒，rank0 rollout
66.244/33.008秒、reference32.790/约0秒，reference miss40513/0。
没有新增远程轮询或NPU任务。原900秒收据最新四rank update0025，RLOO running。

## 固定镜像原生verl检查完成

2026-11-16 01:25 UTC，CPU容器
`2a480fb52056cf7ab422338fe7e3c1be2a2b9ec819c76bbd5375f8ed7ba138f8`
准确wait退出0，独立inspect确认非OOM。固定52ba94e1镜像、2CPU/8GiB、network none、
无NPU设备和模型挂载；363文件快照与运输后hash核对，29项测试全部通过，零跳过。
证据：`docs/engineering/context-dual-credit-native-fixed-image-cpu-v1.json`。

原生verl的compute_policy_loss_vanilla实际执行并校验源码SHA，输出loss及梯度与独立
clipped/dual-clip方程一致；混合双分支目标的整体梯度与分片平均一致，padding梯度为0。
这些是合成logits上的CPU原生运算；DP仍为代数模拟，无真实通信、Qwen前向、NPU更新或
checkpoint恢复，不放行全量训练。旧测试的skip状态保留历史，本次真实执行补齐对应门槛。

复用既有CPU封包/收集流程，新增共享入口`tools.reports.launch_fixed_cpu_tests`和
`tools.reports.collect_fixed_cpu_tests`；共享模板
`configs/portable/operations/context-dual-credit-cpu-tests.json`，本机绑定
`.local/config/dual-credit-cpu-tests-v1.json`。执行后不重跑同tag，所有结果只创建不覆盖。
收集检查完整源码hash、容器身份/命令/无设备/无网络、退出码、JUnit及预期测试数。

旧真实rollout归档的behavior_record字段只有request_id/request_identity/prob/policy_id，
不含原始logits。因此尚不能做全量真实排序差值饱和认证；反推logits不提供等价证据。
下一工程步骤将实际logits输出与bulk RPC/vLLM合批在同一服务版本集成，再进行真实音频
概率与排序分支覆盖、非零权重切换、连续更新/恢复验收，不另建一套推理工程。

## 批量服务实际logits接口已接入候选版本

新增`context_logits_wire`在同一次真实head projection捕获logits，并在原设备验证
sigmoid(logits)与原prob精确一致，输出两列[prob,logit]。readout对象及参数名称保持，
原head/marker激活事务持有的对象不被替换；不重算head，不反推logits。CPU接收端只做
2e-7兼容性检查，不以CPU sigmoid替代实际NPU概率或宣称该检查证明真实执行。

`context_dual_rollout_candidate`沿用既有批量请求校验、长度分组、去重还原、bulk RPC，
统一提供单条与批量入口；worker在原segmented profile之后显式安装wire。
`serve_context_dual_candidate`复用hash锁定的原批量service生命周期，只替换owner与
协议标识。三个候选入口均未修改旧service/worker或冻结包，尚未封存为正式训练release。

43项CPU检查通过，含BF16真实共享Transformer head、projection调用次数/参数与概率
不变、实际spawn子进程socket与张量文件往返、重复输入还原及错误logit/顺序/策略拒绝。
engine.encode与模型输出使用fixture；不是实际vLLM worker/NPU/LoRA切换证据。
报告`docs/engineering/context-dual-bulk-wire-cpu-v1.json`记录源码hash和边界。

下一步将新入口加入同一有限封包，先固定镜像执行接口检查，再真实音频串行/批量
概率和logits验收；原128条probe只能证明有限输入，不替代全TRAIN排序分布覆盖。
随后才接双分支actor、固定reference共享和更新恢复。runtime准入仍false，旧RLOO继续。

## 2026-11-16 接口固定镜像验收与有限测量包

固定镜像wire检查已完成：v3容器`88d67c9639b18460a6622c9db8810d1d2f7d5bda6477bddc182fc6fef368dd75`
准确wait0、独立inspect exit0/nonOOM，43项全过无skip，378文件hash一致。
报告 `docs/engineering/context-dual-wire-fixed-image-cpu-v3.json`。这是实际CPU BF16 head
和spawn IPC，engine仍为fixture，不等于真实NPU合批或训练发布。

失败记录保留：v1中42过、1个spawn测试在10秒启动期限内未等到socket；v3仅将测试启动
期限改为60秒并加启动耗时输出，RPC请求timeout仍10秒，数值断言不变。v3通过支持启动
期限不足的判断，但未采集到导入各阶段耗时，不能指定某个库为确定根因。v2本地绑定曾将
父目录simple-final-v1误替换为simple-final-v2，在创建容器前失败；修正明确父路径后使用
新tag v3，未覆盖原始失败证据。

`context_dual_measurement_candidate`复用原measure、compare和独立audit_measurements，
同时核验每个pass真实logit/prob、输入绑定、原Bernoulli G16、顺序/重复一致性、实际
forward长度覆盖和worker allocator峰值。新增测试与旧测量审核共32项本地通过；固定
runner复用和失败退出保留另3项通过。合成夹具中的逆sigmoid只用于构造测试记录，实际
producer仍捕获projection，不从模型prob反推logits。

共享入口和本地宏：

```
PYTHONPATH=src:.:tools/jev python -m tools.reports.prepare_context_dual_probe_candidate \
  --config configs/portable/operations/context-dual-probe-prepare.json \
  --local .local/config/context-dual-probe-prepare-v1.json
```

该命令已执行，输出只创建一次，勿原路径重跑。1095文件候选包继承原TRAIN128输入、
Stage1 policy/adapter与固定基础依赖；完整payload pins和父包未改已独立核验。
manifest `3691b50a8c0d3fa8f3f41f91bccf2635fe30e1acb858ac0e51f3fd30d49582f1`。
证据 `docs/engineering/context-dual-probe-local-readiness-v1.json`。

这是prepare-only候选，尚未远程派发；launch/collection的宏化适配、原队列完成与设备
lease仍是执行前提。有限128输入不替代全TRAIN排序分布覆盖，也不替代双信用actor、
非零LoRA/head连续更新、新进程恢复和端到端吞吐验收。全量三臂训练仍未放行。
只在上述必要项完成后封存一个正式release，不因CPU通过扩展实验矩阵。

## 配置化有限推理操作入口已补齐

`tools.reports.context_dual_probe_operations`通过同一local_config宏读取机器、路径、
原共享设备租约、实际恢复队列及准确候选manifest，默认只本地验证。复用既有归档、
准确容器wait、终态收集以及hash锁定的独立auditor；不另建自动队列或修改原训练。
启动先检查原队列所有owner locks已释放、完整终态和每阶段结果hash，再获取设备租约。
旧MINDS queue-status为历史stopped状态，不能作当前队列进度；本入口改读正在运行的
`continue_recovered_research`真实input/status/stage-complete证据，未移除评测优先级。

启动尝试、准确容器句柄先持久记录，start响应超时继续wait原句柄，失败结果照常收集；
同tag已有尝试时拒绝重建。审核增加实际镜像/设备/网络/只读挂载/环境/命令与exit检查，
再做原输入、G16、logit和prob的独立复算。runtime成功不自动放行训练。

```
PYTHONPATH=src:.:tools/jev python -m tools.reports.context_dual_probe_operations \
  --config configs/portable/operations/context-dual-probe-operations.json \
  --local .local/config/context-dual-probe-operations-v1.json
```

已实际验证1095文件候选manifest不变。本地43项通过；固定镜像容器
`6d4f04437f89da99f2865977dc04cf940bf99ddc706289e5bbf350f50c09befd`准确wait0，
独立收集确认exit0/nonOOM、43项通过零skip，392文件快照。
报告`docs/engineering/context-dual-operations-fixed-image-cpu-v1.json`及
`docs/engineering/context-dual-probe-operations-readiness-v1.json`。
本地首轮有1个失败来自测试夹具缺少封包storage源文件，补齐夹具后通过；未放松执行门槛。
SSH、engine和容器输出为测试替身；本轮没有NPU推理/更新或效率证据。

有限推理dispatch配置仍false，原队列/租约尚未满足。源候选的历史PREPARED pending字段
保持不改，本节及operations-readiness是操作入口现状。后续工程重点转向双信用batch与
actor接入、按实际runtime身份隔离的共享reference，复用同一前向/优化器/恢复循环；
不能把旧contract相同当作新segmented执行下reference可直接跨runtime复用的证明。

## 双信用批量训练记录与共享reference接入候选

`tools.reports.context_dual_session_candidate`复用原ContextRolloutSession激活、确认更新和
失败状态机，以及ReferenceCache原子持久化实现。每批固定reference缺失项只发一次
bulk RPC，当前behavior另发一次；长度分组和实际batch4由原service负责，不另造调度器。
组大小G16为小张量采样次数，不增加16次模型前向。全量工程仍需真实吞吐计量。

`DualContextGroups`先确认候选顺序、完整布尔监督、当前policy、全局槽位唯一及零padding，
只将inputs/positions送入engine。实际prob/logits返回后按完整输入身份核验，调用已验证的
dual_credit.prepare构造独立排序/存在性动作、reward、advantage与绑定记录。reference
只保留候选概率供KL，不能用其替代下一步当前策略前向。原Frozen运行不使用此候选。

`bind_candidates`消费原CachedContextBatches已经编译并与编码输入核对的target metadata，
检查问句候选顺序、标签与query一致后补入candidate_ids。它不按标签选择问题，不新建
数据预取器，也不重新解码音频；正式全量population与来源审核仍由既有loader负责。

RuntimeReferenceCache按image/source manifest/processor SHA、执行profile及output protocol
分离namespace，并把身份写入每条缓存。共享范围是同runtime下相同固定Stage1策略和
实际输入，跨实验可复用；不同runtime即使policy相同也miss。保持本地fsync、不可覆盖
写入及冲突拒绝。runtime身份由调用方核验真实服务后提供，本类不独立证明远程来源。

本地42项检查通过：两步不同policy、新session参考复用及G16重放、候选ID规范排序、
padding槽位保留、参考去重、目标不送engine、错序/缺记录/错logit/切换/saturation拒绝、
runtime namespace隔离、不可覆盖冲突，以及真实CachedContextBatches代码接线。
模型输出是fixture，恢复检查只是session与采样，不是optimizer/NPU恢复。
首轮有一项fixture的饱和方向写反：规范ID为(a,b)、输入顺序为(b,a)，将fixture改为
[-10,10]后产生预期FP32正向饱和拒绝；实现、概率门槛和数值协议未改变。

待完成的主要接线是verl DataProto/actor混合loss，以及真实服务runtime身份认证；现有
session产物尚未进入optimizer。必须保持原生verl更新/梯度累积/恢复循环，并在更新前
同时核验排序和存在性的行为概率，不能只验旧60-bit联合概率就称双信用训推对齐。

固定镜像CPU检查已闭合：容器`3c4e1fa1bd17b771e7b5fd4a20b9b47278ad69acbb19e27ba4043bf20caed632`
准确wait0，独立收集exit0/nonOOM、42过零skip，369文件快照。
证据`docs/engineering/context-dual-session-fixed-image-cpu-v1.json`。
此检查没有挂载NPU或模型；不代表真实优化器更新、数值准入或收益。

## 原生verl actor双信用接口已接入候选

`tools.reports.context_dual_actor_candidate`复用SHA锁定的原context_native_actor构造及
原生DataParallelPPOActor.update_policy，不修改原文件。独立MODE/ContextVar避免污染旧
训练；替换DataProto装配、集体验证、前向后双分支parity及registered_loss四个接口。
一次microbatch前向输出共享logits，双分支混合loss只回传一次，复用原优化器与梯度累积。

传输检查重算完整输入tensor identity并绑定dual receipt、候选顺序、标签、policy、update、
G16/seed317，拒绝旧记录、错误全局分母、非零padding及篡改动作/优势。原数据分区的
global_rows/global_real_requests保留到各rank；loss先按真实全局query归一化，再补偿
原生verl的accumulation除法一次。原有binary旧概率列只作传输，不冒充排序动作的ratio。
真实reference provenance仍由session/launcher负责，本接口只能核验其policy标签与数值。

前向后以实际logits分别核验排序和存在性，使用原固定2e-7概率、2e-6 log概率与零阈值
翻转门槛，所有rank汇总后再进入loss/backward；不能仅比较边际prob就放行排序分支。
rank异常在集体门槛报告，不允许其他rank继续optimizer。尚需实际四rank故障/恢复验收。

固定镜像CPU容器`79abfcba1b6f3726f152559de3099b0e761e6b6e2346a62d5201ce460b4cdc23`
准确wait0、独立收集exit0/nonOOM，366文件快照、15项全过无skip。
报告`docs/engineering/context-dual-actor-fixed-image-cpu-v1.json`。
其中RLOO与GRPO-floor各测试micro1/4，实际单rank Gloo/DDP及原生verl update_policy执行；
梯度与完整batch hybrid_loss一致，零padding保留、optimizer step=1，真实forward次数
为4或1。前向替换为合成双logit参数，非Qwen音频；这不是四卡NPU或真实优化器恢复。
梯度参考仍使用同一hybrid_loss，验证的是actor接线/累积；损失独立方程证据见此前29项。

宿主9项通过、4项verl缺失skip由上述镜像实测补齐。镜像快照后只补强一项parity测试：
[-8,-10]的小幅反向扰动使存在性两项门槛均通过、排序失败；本地1项通过，actor源码
与镜像快照完全相同。单独记录`context-dual-actor-parity-followup-v1.json`，不将此测试
修订冒称已在镜像执行。原镜像15项报告/源码快照均保留。

下一主要动作是同一actor下真实四rank连续更新与恢复，然后接统一trainer/session/service
及实际runtime认证。尚未把候选替换进原RLOO或启动全量三臂训练；完整音频NPU、全TRAIN
排序饱和覆盖和端到端效率仍未通过，不能以本轮CPU优化器证据宣布工程release完成。

## 四rank连续更新与独立进程恢复

新增`tests/test_context_dual_actor_dp_resume.py`在固定镜像启动实际四rank Gloo/DDP，
RLOO/GRPO-floor各执行micro1/2两种配置；每rank两条query、全局8条/5条real，rank3
全为padding。两步都调用原生verl update_policy，验证全局梯度与完整8条混合loss一致，
Adam步数正确；单rank错误分母及行为概率错误经collective在全体rank拒绝。

第一次四rank进程结束后重新spawn四个解释器，使用同一context_training_checkpoint
capture/restore接口恢复第一步FP32参数、Adam状态及实验身份，重放第二步；逐张量
精确比较第二步参数、Adam moments/step、两路G16动作/奖励/优势/绑定及梯度。
测试使用合成logit参数，并非Qwen音频前向、BF16 FSDP或真实service切换。独立梯度
参考使用同一hybrid_loss，验证分布式接线与归一化，不冒充独立推导loss正确性。

v1容器`615989407c75aa836994c3f4081d19e7ffd17a0c70f4ffd42bc9279cbc0a6fd6`
准确wait0、收集exit0/nonOOM、14项全过无skip，366文件快照；四rank测试内每rank
连续8个case、重放4个case。报告`context-dual-dp-resume-fixed-image-cpu-v1.json`。
其中另13项是actor接口检查，包括前轮补强的存在性通过但排序拒绝案例。

并行代码审查发现前向异常顺序：原候选在集体parity前构造binary log-prob，单rank
sigmoid饱和可能先抛错、其他rank进入collective而等待。候选已将check_dual_forward
移到log-prob构造之前。新增真实actor.update_policy路径中的仅rank3饱和案例，要求
所有rank集体拒绝、参数及optimizer不变。不是放宽概率门槛，也没有改冻结actor源。

这个修订与v1执行快照不同，正在用新tag v2验收，不能用v1成功证明修订已执行。
v2容器`1dda47d8f87fbf13b3422e45192c324efa063feb2983860c19907e5542f967d2`，
CPU-only，准确wait由当前session持有；终态后用同一collector读取，勿重派。
shared config为`configs/portable/operations/context-dual-dp-resume-cpu-tests-v2.json`，
local binding为`.local/config/dual-dp-resume-cpu-tests-v2.json`。

后续仍须把actor/session接到同一正式trainer，并完成真实音频合批、非零权重发布/激活、
BF16四卡更新恢复和整体效率验收。当前仅CPU分布式工程证据，不代表学术指标或泛化收益。

v2于2026-11-16 02:08:43UTC完成：准确wait0、独立收集exit0/nonOOM，14项全部通过、
零skip。报告`docs/engineering/context-dual-dp-resume-fixed-image-cpu-v2.json`。
当前actor SHA为`c24012e738e02dcfd8dc2c465cecb59a9d15ad7d9183db30b0905fe2f363dbd0`，
与执行快照一致。单rank饱和实际通过actor.update_policy集体拒绝，无参数或Adam状态
改变；随后正常两步及全新进程恢复重放通过。前述pending状态已结束，不重派该tag。

## 统一trainer与服务绑定候选

`context_dual_trainer_candidate`从hash锁定原trainer生成集成入口，替换actor/session/
cache引用、双分支checkpoint契约、candidate metadata绑定和group归档字段。原NPU/FSDP
优化器、保存、BF16导出、激活和恢复循环保留。`context_dual_runtime_candidate`核验完整
Python源码manifest及实际processor文件，并在socket首消息绑定runtime/policy/contract，
之后沿用原bulk协议。image身份仍由launcher Docker inspect确认，进程不能自证镜像。
服务恢复可以使用不同snapshot目录，但processor内容hash必须相同。

固定镜像05cafa5ff5033474b17b0802df272330d4fdb1536d6cdcac9f75d1a32b56d384已
准确wait0、exit0/nonOOM，28项过无skip，402文件。证据
`docs/engineering/context-dual-trainer-fixed-image-cpu-v1.json`。含实际socket greeting及
现有session检查，engine为fixture，尚非真实NPU trainer运行。

宏入口prepare_context_dual_training_candidate生成初版397文件整合包，原前缀来源、
Stage1、全量schedule及共享supervisor保留，参考缓存独立按runtime复用，dispatch=false。
该包早于用户新增LoRA保存规约防护的源码修改；后续使用新版本封包，不改旧package。
共享模板`configs/portable/operations/context-dual-training-prepare.json`。

用户明确禁止反复保存主模型，新增白名单并接trainer启动前检查；本地6项scope+12项
集成检查通过，修改后的trainer尚未固定镜像验证。详见
`docs/engineering/lora-checkpoint-storage-policy.md`。原保存代码及已检查实际执行文件
均只保存LoRA/head/必要marker；不得把适配器多格式导出称为主模型完整权重。
