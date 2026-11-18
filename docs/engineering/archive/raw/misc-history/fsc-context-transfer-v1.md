# 当前概率头的FSC原生任务迁移输入

## 2026-11-15 完整迁移接续已排队（尚无目标成绩）

当前执行入口为`run_fsc_context_experiment.py`，默认只验证无标签输入身份，
显式`--launch`后顺序调用源策略封存、运行包准备、原设备锁下的推理、独立
收集和评分。实际spec为
`.local/decision-objective-research/fsc-context-experiment-v1-spec.json`。
完整3793条、31候选、定义/人口/processor收据已绑定，训练源固定为
`context-baseline-sequence-v3/grpo_v1-experiment`，不按目标表现切换算法。

`prepare_fsc_context_source_v2.py`只更新源曲线审核依赖到已修复的v2，
原策略/adapter/processor复制及身份校验语义保持。`prepare_fsc_context_package.py`、
`launch_fsc_context_pair.py`、`collect_fsc_context_pair.py`已有固定镜像、只读输入、
短IPC、Ascend入口、准确句柄及独立运行来源核验；不是新训练栈。

`score_fsc_context_pair.py`先重跑独立推理审核，验证实际容器检查回执、两份
完整预测、源策略和预先冻结的目标规约一致，才调用原评分器读取TEST标签。
`runtime_provenance_verified=True`只能来自这条重验路径，不能仅修改原评分器
返回标志。两策略固定为源域step0与SLURP选中点；选中step0合法，不强求正收益。
目标结果只做完整配对描述，不能冒称semantic-presence迁移或纯RL因果收益。

本地52项测试通过；固定镜像无NPU容器9dc363f16f47正常退出并独立收集，
115项测试通过/52.82秒，见`fsc-context-experiment-cpu-v1.json`。
测试包含真实小型BF16资产和文件校验，源完整审核、模型前向、SSH和目标读取
在编排夹具中替代；这不是实际NPU迁移或目标域成绩。

driver582739持本地owner锁，1071项依赖身份固定，等待现有comparison v5
成功终态后才启动；状态在
`.local/decision-objective-research/fsc-context-experiment-v1-queue-status.json`。
前序失败即停止并保留句柄，不轮询NPU、不重启训练、不抢占RLOO资源。
尚无实际GRPO源包、31候选模型parity或3793条真实迁移预测。
本接续不增加SLURP准入条件。以下为此前阶段记录。

## 2026-11-15 源策略封存与双策略执行候选

`prepare_fsc_context_source.py`已补齐源策略封存：只接受GRPO完整训练和11点
SLURP选择曲线，重新调用独立训练/曲线审核，固定step0与源域选中checkpoint。
从已审核评测产物复制原始policy ID、BF16执行张量、native adapter和processor，
并核验其processor与FSC既有缓存一致。允许选中step0，不强制报告正收益；
不根据FSC选择算法、checkpoint、阈值或候选文字。尚无真实完整GRPO源包。

`run_fsc_context_pair.py`是同容器顺序执行候选，不负责远程启动或设备分配。
输入spec含source_package/source_manifest_sha256、model、physical_devices、
pins，以及inputs中的input_manifest、input_audit、requests、input_receipts、cache_root。
子进程共享同一封存策略与缓存；运行环境需包含工程根、src、tools/jev的PYTHONPATH。
运行命令为`python -m tools.jev.run_fsc_context_pair --spec SPEC --output NEW_DIRECTORY`。
须由后续固定镜像launcher持原设备租约执行，本轮未运行此NPU命令。

执行顺序为initial HF/vLLM面板、selected HF/vLLM面板、initial全3793推理、
selected全3793推理。两份原始面板输出均按确定的输入索引、真实processor收据、
policy ID及31概率重验；两策略全部exact通过后才开始任何全量推理。
全量输出重新检查3793条顺序/身份/概率/文件hash；失败保留现有输出且不自动重试。
仅成功完成推理仍需独立collector及既有评分入口，不能作为任务成绩或运行溯源认证。

46项CPU测试通过/9.80秒，涵盖真实小型BF16策略与adapter复制、合成3793条
预测完整性和双策略失败编排。完整源域审核在打包测试中替代，模型子进程与
存储准入在编排测试中替代；没有实际31候选NPU前向或目标标签评分。
证据`fsc-context-pair-orchestration-cpu-v1.json`。NPU打包/launcher/collector仍待，
这部分不是当前SLURP简单RL发布的前置条件；以下各阶段状态保留作历史。

当前context-prob-v1简单主线采用exclusive_event，先将同一输入契约迁移到
FSC原生31命令。保留transfer-protocol-v3独立语义存在性路线；其未审核标签
不从原生单标签投影，也不因本输入包完成而获得准入。

`tools/jev/prepare_fsc_context_transfer.py`复用现有
`configs/tasks/fsc-span-transfer-v1.json`原生类别描述，不根据TEST表现调整文字。
每音频供给完整31候选与规约，一次输出31个独立概率；无需31类专属参数或
softmax。31候选是新输入长度/读出数，仍需实际processor与HF/vLLM验收。
none是原生未指定类别，原生域默认值不被声称为真实语义存在性或执行可行性。

输入包位于`.local/decision-objective-research/fsc-context-transfer-inputs-v1/`，
完整3793条官方TEST人口、10位说话者，来源为既有label-free roster。
本轮没有读取TEST标签/参考转写或运行模型；项目历史FSC接触仍须披露。
每条只含音频路径/hash、录音/说话者身份、候选输入及request/render hash。
独立核对全人口顺序与全部候选定义，2项测试覆盖31合法命令绑定、候选置换、
缺候选拒绝以及exclusive标签不能冒充candidate_applicability语义标签。
证据`fsc-context-transfer-inputs-v1.json`；未重新解码音频/验证processor tensors。

后续评测冻结本Stage1终点与仅由SLURP选择集选出的RL checkpoint；权重、阈值0.5、
候选定义均不根据FSC结果调整。以完整人口集合F1、exact-set、NLL/Brier与top1
原生三元组准确率/固定31类macro-F1报告；empty/multiple selections单独显示。
分数的差异只称原生任务配置迁移，不能称已证明I_present或槽位语义迁移。
官方监督模型仅作能力参照，其训练条件不同，不能混称零训练基座。
当前training/evaluation dispatch均false，尚无source权重身份、processor、
后端一致性或实际迁移收益；不阻塞SLURP简单任务训练。

## 固定镜像真实processor检查（运行中）

`tools/jev/preflight_fsc_context_inputs.py`在固定镜像CPU容器
b55c28bd465c78643d3bc2b39f1a1e86b63a7a306f4034e6ab270217dc45bf66
执行全3793条音频hash/解码/processor，31个marker位置和4096上下文上限检查，
不截断、不加载模型、不打开target CSV。14:32:30 UTC启动，2CPU/8GiB、无NPU，
/data ext4热路径预检通过。输出保留输入张量hash及可复用纯输入cache。
launcher PID375240准确wait；独立collector driver见
`.local/decision-objective-research/fsc-context-processor-cpu-v1-status.json`。
首次普通观察不早于14:42:30 UTC；准确wait可等待终态，不按超时重启。
collector将独立逐cache核验全部张量/绑定与完整人口，尚无完成报告。

终态更新：该容器14:35:33 UTC exit0/nonOOM，全部3793条完成，真实展开输入最长
2478 tokens，全部31读出位置通过4096上限检查，无截断。独立collector已重新读取
每个cache的文件hash/张量hash、request绑定和完整官方人口，证据
`fsc-context-processor-cpu-v1.json`（manifest a2d090fc，独立stream43223914）。
driver375992正常收尾。尚无模型前向、31候选HF/vLLM一致性或迁移收益；
不得把processor成功当作目标域任务性能。

## 概率读出评分候选

`tools/jev/score_fsc_context_transfer.py`接收manifest绑定的source_selection、population、
task、input_audit、input_package_manifest（均为path/sha256对象），input_receipts路径，
以及policies.initial/selected的policy_id与predictions(path/sha256)。source_selection
必须是完整SLURP task曲线，两个policy ID分别等于initial_policy_id与selected_checkpoint。
源域权重及运行来源仍需独立collector认证，评分工具不把自报来源当作运行验收。

先校验源域选择、输入包与processor审计及逐条request hash，再读取官方target CSV。
全部3793条预测绑定policy/input identity，固定31类和阈值.5；拒绝缺失/重复/非有限
概率或身份不匹配，不能丢坏例后报告全量分数。输出集合F1/exact-set、top1准确率、
固定31类macro-F1、NLL/Brier、空/多选比例及逐说话者结果；两个预选模型只做配对
描述性差异，不选择FSC最优模型，不声称显著性。

14项相关CPU测试通过，证据`fsc-context-metrics-cpu-v1.json`；包含集合与top1分离、
人口完整性、策略/输入错配、非有限概率拒绝和配对gold身份检查。
本轮仅合成数据测试及CLI编译，没有读取真实TEST标签执行评分。实际来源选择与
目标预测尚未完成，该入口不等同端到端迁移验收。

## 全人口推理候选入口

`tools/jev/infer_fsc_context_transfer.py`复用固定`FixedContextRollout`与已审核输入cache，
max_decisions=31、max_model_len=4096，一次概率读出；无target CSV读取、转写或优化器。
先绑定完整source_selection，只接受其中initial或selected策略ID，再核验输入包和
全3793条独立processor收据。源processor必须与cache审计一致，缺cache拒绝重建；
每条重新校验音频hash、cache字节/张量/marker身份，逐条保存预测及输入receipt。
最终只输出完整预测，不在推理入口选择目标域模型或计算分数。

真实3793条请求/独立收据绑定已在CPU检查通过；新增9项测试联合已有评分10项共
19项通过，覆盖源策略预选限制、错误输入/策略/digest、异常概率及缺决策输出。
证据`fsc-context-inference-input-check-v1.json`。没有实际模型前向或读取target gold。
候选尚无NPU launcher/collector、31读出HF-vLLM一致性与真实完整推理证据，不能
据此解除迁移派发限制。后续须完成源域选择和31读出后端验收，再封装固定镜像运行。
已有Stage1/连续RL排队源码和配置保持不变，此入口不增加SLURP主线前置要求。

## 31候选后端检查候选

`probe_fsc_context_parity.py`采用与完整推理相同的`audited_encoder`，该纯输入函数
从尚未发布的FSC候选入口提取复用，未改冻结SLURP或排队源码。HF/vLLM分别运行，
同一原始snapshot、同一缓存BF16特征与marker，比较完整31概率及0.5阈值翻转；
不重导出或更换policy ID。HF沿用既有named packed-row profile、LoRA/head加载方式，
vLLM仍用FixedContextRollout；原始双侧输出保留，不一致拒绝。

从全3793输入按固定规则选每说话者首条、最短16条和最长16条，合并去重得到42条，
覆盖全部10说话者及2418–2478 tokens，不按target标签/结果选样。
真实输入面板已在CPU绑定，见`fsc-context-parity-panel-v1.json`；新增8项测试与
推理/评分合计27项CPUtests过。覆盖输入/策略错配、缺失面板记录、非有限概率、
可见概率偏差与阈值翻转。CPU检查不代表HF或vLLM实际执行成功。

两进程候选尚未封装NPU launcher/collector，也未占用排队SLURP设备；
尚无42条后端实际概率或全3793迁移结果。正式迁移使用源域预选策略，
短工程检查若提前使用诊断权重必须另外标识，不能替代最终源策略验收。

## 实际缓存消费验证完成

固定镜像CPU容器b550b6d5已exit0/nonOOM；用候选`audited_encoder`读取全部3793条
原缓存，3793次命中、无重建，总9258074输入tokens，消费环节111.93秒。
cache挂载只读，每条实际读取核验原音频hash、cache文件hash、完整张量及marker身份。
独立collector重新核对成功终态、源码/配置hash、全部消费ID/request/cache key顺序及
token总数，证据`fsc-context-cache-consumer-cpu-v1.json`；collector driver412085已收尾。
这项验证支持实际输入消费路径可用，不包含模型前向、HF/vLLM概率或迁移表现。
全量SLURP和连续RL原设备队列未受影响；仍需实际42条后端检查及源域权重选择。
