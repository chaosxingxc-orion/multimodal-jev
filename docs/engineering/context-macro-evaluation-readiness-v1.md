# 双信用 GRPO 的效果与迁移评测接入边界

2026-11-16 13:27 UTC：新增共用`context_transfer_macro_parity.py`及portable宏入口，
支持HF独立进程与实际macro双服务的目标宽度验证，保留实际logits/prob、独立输入
绑定、journal重建及精确差异报告。真实FSC/MINDS/SLURP TEST全人口准入器已生成
64/35/46条无标签固定面板；FSC覆盖10说话人，MINDS无语言字段且未打开gold补取，
不声称全语言覆盖。50tests2.84s过，真实pool/日志流程、模型替身；未派发NPU。
详见`context-transfer-macro-parity-v1.md`。当前服务batch32的100步对照优先，原驱动
及尾批后继保持；后文micro8/16/32研究安排为历史记录，micro16已OOM且micro32未派发。

当前100步microbatch8/16/32是效率实验，不能直接替代正式任务收益曲线，也不能用
目标集成绩挑选microbatch。冻结前已声明的源/迁移指标与隔离原则继续有效，见
`../research/context-grpo-transfer-macro-batch-v1.md`。

已核对的可复用部分：

- SLURP selection完整人口9,226，60候选独立概率，阈值0.5；评分复用
  `ContextEventMetrics`，报告set-F1、top1、macro-F1、binary NLL/Brier及覆盖。
- `context_selection_curve.py`接受预声明的检查点列表；必须完成列表才选checkpoint，
  按set-F1优先、NLL和较早step破平局。固定终点单列，不能以事后最优点掩盖退化。
- `evaluate_context_event_records.py`已接便携v2加载器，v1兼容；同一预测在v1/v2
  得到完全相同指标，搬迁v2并删除开发审核目录后仍可评分。推理和v2独立审计入口
  同步接便携加载器，冻结旧运行包不变。此项只验证发布兼容，不代表新NPU评测已通过。
- FSC、clean TEST、MINDS已有输入/评分工具；源域选定的initial/selected策略对
  必须保持原身份，目标集不得参与reward、难例采样、阈值或模板选择。

仍需接入的具体工作：

1. 正式DP2训练需要独立预算和发布身份。当前trainer只允许工程/效率模式，
   `training_dispatch_allowed=False`；不能修改这个布尔值就将100步称为完整研究实验。
2. `context_rl_evaluation_plan.py`仍要求旧`grpo_v1/rloo_v1`及634步、11点完整曲线；
   `prepare_fsc_context_source_v2.py`也绑定旧GRPO曲线。新增双信用算法需显式适配
   `dual_credit_grpo_floor_clipped_v1`和DP2 checkpoint来源，不能重命名为旧算法绕过校验。
3. 新评测需消费训练已验收的实际执行runtime。旧selection入口使用`FixedContextRollout`，
   不等于当前macro批推理runtime已验收；接入时保留实际输入张量、marker位置、
   candidate顺序、LoRA/head身份及HF/vLLM对齐检查。
4. 完整源selection曲线完成后，按预声明规则封存initial/selected对，再跑FSC、
   clean TEST、MINDS；目标分数仅报告，不反向影响本轮模型选择。SFC函数适用性
   需要独立输入/标签审核，SLURP→FSC/MINDS不能单独证明跨任务能力。

工程推进顺序：完成相同条件的microbatch效率比较，选可行且有实测收益的配置；
在同一基座上接正式训练预算、评测计划及策略对发布；完成源域与迁移证据后讨论复杂任务。
保持现有两路排序/存在性信号，算法改动与吞吐改动分开登记；历史GRPO集合F1提高而
NLL/Brier恶化的结果继续保留，不能用训练reward或新运行完成量代替概率质量评测。

## 批量评测入口实现进展

新增`tools/reports/context_macro_evaluation.py`复用训练侧`MacroRolloutPool.score_macro`，
按完整人口顺序提交最多128条query，末批保留实际条数，不填充、不丢样本。编码回调
只接受id/audio hash/request hash，候选顺序、marker位置、输入张量hash绑定收据；
返回值核验实际logits/prob、policy/request identity、完整macro收据及结果hash。
每批成功后原子保存预测和输入证据，完整人口完成前不发布总预测与成功结果。
失败保留已完成批次，不自动重试、切策略或消费部分结果。运行目录只能在已准入本地盘。

新增`tools/reports/evaluate_context_selection_macro.py`接SLURP便携v2及既有评分器，
完整读取9,226条selection后剥离标签，仅音频、固定全60候选和要求进入编码/推理。
本地宏模板`configs/portable/evaluation/context-selection-macro.json`统一策略、cache、
服务连接和输出目录。它连接已启动的隔离服务，不创建服务或使用训练中服务，不修改
任何冻结实验包。正式部署仍需版本化封包/operations、固定镜像检查和独立收集审计。

共享入口支持固定的14/31/60候选面板；该支持不代表FSC/MINDS已完成新runtime接入。
FSC/MINDS还需将各自已审计cache收据适配给编码回调，并通过各自真实候选宽度的
HF/vLLM与macro一致性验收。SFC变长函数面板暂不适用此入口，不因此缩小跨任务目标。
当前进展仅CPU合成输入与真实macro调度/原子文件边界验证，没有新增NPU评测成绩。
35项测试通过（23.09秒）：共享macro评测、固定14/31/60宽度、完整9,226合成人口
标签隔离、原macro调度以及便携发布评分。首次测试发现可变progress误用不可变
commit接口，已改为既有atomic_json；结果/批次证据继续不可变提交，fsync保留。

独立输出审核`tools/reports/audit_context_macro_evaluation.py`现已实现，宏入口为
`configs/portable/operations/context-macro-evaluation-audit.json`。必须显式提供独立认证的
population、candidate_ids和processor输入收据及各自hash；不能从本次推理保存的收据
反推“独立预期”。逐batch核验实际INPUT、每个rollout的完整归属/结果、COMPLETE与
总预测，拒绝缺样本、重复槽、错误副本、输入错绑、策略变化、错误logits及多余batch。
即使篡改者同步重算结果文件hash，也须与独立输入一致。输出为不可覆盖的审计收据。

44项CPU测试通过（2.78秒），覆盖真实MacroRolloutPool调度/落盘与独立重建、尾批、
重新hash后的破坏拒绝、宏配置解析和不可覆盖审计。模型及音频为合成替身，尚未证明
NPU概率、HF/vLLM对齐、实际容器/传输身份或目标任务收益。完整部署collector仍待。
selection cache准备入口也切换到portable loader，以免未来v2预处理重现开发路径依赖；
旧冻结包不改。已有processor收据格式到此审核入口的适配与实际完整人口复核仍需完成。

## 全量selection旧缓存与v2请求绑定

`context_selection_audited_cache.py`现在提供固定宏入口和只读编码适配。selection评测
入口要求事先认证的cache审计与独立收据，不再允许评测时调用processor补建缓存。
消费时逐文件校验cache key/文件hash，核对实际所有输入张量与独立hash、完整marker
位置和上下文上限；缓存缺失直接拒绝，不重新打开原始音频或下载资源。

实际固定脚本已将9,226条v2 selection请求（逐id/audio/request hash及顺序）与旧版
独立cache收据完整对应，processor身份与当前micro8封包的Stage1资产一致。
通过v2迁移声明以及selection.jsonl、selection.audio.jsonl、tasks.json的未变hash
显式关联旧release，不重标旧审计。结果：
`/srv/multimodal-jev-xingchao/operations/selection-macro-input-binding-v1/binding.json`。
入口配置`configs/portable/operations/context-selection-cache-binding.json`，本地宏
`.local/config/selection-macro-input-binding-v1.json`。旧关闭诊断按原证据保留。

该结果证明全量请求与既有独立缓存证据兼容，没有访问远端缓存文件，不能据此声称
当前远端文件仍齐全。实际消费会再次检查字节；独立审核所需positions具体值仍需从
认证缓存提取并对原positions hash验证。部署封包、真实容器/传输和NPU对齐仍待。

实际只读缓存核验现已派发：`verify_selection_macro_cache.py`在固定镜像CPU容器中
逐条检查完整9,226缓存文件、request/processor/encoding身份、全部tensor hashes及
真实60个marker位置，导出新审核器需要的具体positions和输入收据。operations入口
`selection_macro_cache_operations.py`/宏模板`selection-macro-cache-verification.json`
限制无网络、无NPU、只读cache、两CPU/8GB，本地独立输出；不访问训练进程或修改缓存。
收集时验证准确容器/镜像/命令/挂载与源码hash，下载后再次逐文件核验，独立对原
tensor证据复算positions hash。全量合成人口真实torch缓存读取及破坏拒绝2tests通过
（42.00秒），不是实际缓存完成证据。

当前实际容器`1e28e37bf1059ae982aab1778b155f7f469e091605c49de4ee75511b5b17a1ee`
已确认running，准确docker wait等待；ops为
`/srv/multimodal-jev-xingchao/operations/selection-macro-cache-verify-v1`，
远端`/data/xingchao/jev-selection-cache-verify-v1`，本地宏
`.local/config/selection-macro-cache-verify-v1.json`。尚未完成收集/独立审核，不重复派发。

终态更新：上述容器exit0、非OOM；完整9,226条实际cache字节/张量/marker检查完成，
固定脚本收集并独立比对原审计tensor hashes通过。新收据hash
`f9cc47e5d58e23d777431369e048554376c959eb3c81c2955d334e033ee2cf9a`，
证据`selection-macro-cache-verify-v1-observed.json`，ops下`audit.json`及`transport.json`。
此次没有重编码音频或执行模型；已解除实际缓存存在性与positions具体值的缺口。
后续部署仍要绑定这些输入证据、实际policy/runtime，并独立核验NPU概率与完整评分。

## 全量Stage1评测包

`prepare_context_selection_macro.py`和两个portable operations模板准备正式全量selection
基线；`context_selection_macro_operations.py`复用原双TP1生命周期、准确句柄和独立
transport collector。预检无NPU，加载完整v2/独立cache证据和实际首尾cache、校验
policy/native adapter/runtime，再建服务。收集重新核对service启动身份、完整batch
journal、独立processor收据，并用现有scorer复算全部指标。客户端只读挂载发布与cache。

实际已封包并validate：
`/srv/multimodal-jev-xingchao/operations/selection-macro-stage1-v1-prepared`，
manifest `d38ab9ffb0b2645cb0d729578ad7ff6d56a06d8e0905d8c19480f61e2b0c80a0`。
本地宏`.local/config/selection-macro-stage1-v1.json`，预定远端
`/data/xingchao/jev-selection-macro-v1`。仅共同Stage1策略，尚未派发，dispatch=false；
当前训练占用6/7且持独占lease，不能借用训练服务或绕过lease。

实际源码比较确认service/runtime Python全部沿用v6；仅新增3个评测client模块及
旧scorer的portable loader变更，policy/adapter逐文件字节不变。运行身份显式换为
新完整源码清单，不冒用旧manifest；证据包旁`source-comparison.json`。
28启动scope/挂载/设备/transport测试通过（2.38秒），实际封包验证通过。
尚缺真实评测执行及新版完整源曲线/迁移策略绑定，此Stage1基线不证明RL收益。

## FSC/MINDS复用统一macro输入

`context_transfer_macro_inputs.py`复用各自既有完整输入准入器，输出相同格式的
population/candidate-ids/input-receipts，供共享macro推理及独立审核消费；只读cache
编码器复用selection实现，并将实际输出与各目标原独立收据比较。每个请求核验
request/rendered hash、固定全候选及顺序，positions具体值再对tensor hash复算。
metadata中的speaker/language不进入模型输入；不读gold，不新增目标模板或词卡。

实际全量转换已通过：FSC官方TEST 3,793×31，MINDS冻结上游TRAIN 8,168×14。
共同processor为`41f8edc45719bb2d02b385f57520b862f605df05d38fea60788eed8115ee96fe`。
产物在`/srv/multimodal-jev-xingchao/operations/fsc-macro-inputs-v1`和`minds-macro-inputs-v1`，
本地宏分别`.local/config/fsc-macro-inputs-v1.json`及`minds-macro-inputs-v1.json`，
共用`configs/portable/operations/prepare-transfer-macro-inputs.json`。
此处验证旧独立输入证据，没有重新访问远端cache，也未评测目标表现。

MINDS既有评分函数现支持显式`request_identity_protocol=macro_compact_json_v1`，
验证原始macro请求ID和actual logits/prob。默认legacy模式保持不变；不重写预测ID、
不自动猜协议，语言聚合与所有指标复用原函数。同一概率两协议评分完全一致，
拒绝混协议/输入错绑/logit破坏。新输入+selection cache27tests通过（4.63秒）。
连既有MINDS评分、macro协议与目标输入适配共25项回归测试通过（2.26秒）。
目标权重仍需等完整源域曲线封存；macro目标部署、实际14/31宽度的后端一致性和完整
推理/独立评分尚待。它们只能证明跨数据集/领域迁移，跨任务仍需SFC等独立监督准入。

## 三策略迁移客户端与输出审核

`evaluate_context_transfer_macro.py`接入现有FSC/MINDS全量输入准入、只读cache编码及
同一`MacroRolloutPool`，不在推理进程读取目标标签。portable入口为
`configs/portable/evaluation/context-transfer-macro.json`。推理前先读取由上游独立
源域评测收集器提供的`jev-macro-transfer-source-evaluation-v1`绑定：算法明确为
`dual_credit_grpo_floor_clipped_v1`，634更新、11个完整selection点、9,226人口、
阈值0.5；各点评分文件绑定hash，用既有`summarize`重算set-F1→NLL→较早step选模。
接受共同Stage1、源选新版GRPO和固定634终点三角色，拒绝缺点、换角色或改选模结果。
目标三角色可相同，但每次实际推理的策略身份不可重标。

`audit_context_transfer_macro.py`以独立全量输入证据重建全部macro journal/输出，
再次核对源角色和最终产物hash。支持证据路径搬迁，内容hash必须不变；不把推理自报
的input evidence代替独立输入。portable审核入口为
`configs/portable/operations/context-transfer-macro-audit.json`。

66项相关CPU回归通过（3.29秒）：新三角色/源曲线拒绝、真实macro调度与独立日志
重建、14/31候选面板及尾批、聚合结果和角色破坏拒绝、既有输入与MINDS评分。
模型后端、源评分和输入在集成测试中为替身；没有实际NPU迁移推理或目标评分。
这两个入口仅核对绑定内容，不把一个声明了hash的源评测文件当作已完成真实源曲线
审核；上游完整源模型/容器/权重/评分收集、实际14/31后端一致性、目标部署封包及
独立目标评分连接仍待。当前不生成假的source-evaluation文件或派发目标推理。

## 目标评分连接（离线收集器组件）

`score_context_transfer_macro.py`补齐三角色评分矩阵，全部唯一策略经source binding和
完整macro输出重新审核后才打开gold。相同policy必须复用同一collection/journal，
保存的执行角色不重写；报告仍保留起点、源选、固定终点三栏及两组相对起点的变化。
矩阵要求同一源曲线、输入及runtime；不做目标winner选择，也不混合数据集平均分。

FSC复用原生31类评分，评分task/population hash必须等于实际请求清单，逐条核对
问句、音频、说话者；再由官方CSV和实际音频hash提取标签。MINDS复用原生14类评分，
显式使用macro请求ID协议，校验冻结inventory及原音频→16k音频conversion ledger，
保留14变体/12语言报告。两个数据集均保留NLL/Brier、top1和set指标及负迁移信息。

该模块没有独立CLI或派发入口，供后续实际runtime collector调用：它不认证容器、
源权重或14/31后端parity，结果明确`runtime_provenance_verified=false`。不能在缺少
这些外部验收时把局部评分当作已完成真实迁移评测。

验证：评分矩阵/既有迁移客户端及指标50tests通过（30.69秒）；完整3793条合成FSC
CSV/音频、8168条合成MINDS原始与转换清单及输入适配20tests通过（22.97秒）。
矩阵测试的预测与runtime/source审核为替身，独立macro日志重建由前一轮测试覆盖。
真实目标标签与预测本轮未读取；没有新的目标指标。输入适配仅额外返回原始无标签
source_rows供评分连接，既有population/candidate/input-receipt格式与内容保持不变。

实际无标签输入回归也已完成：重新经原准入器加载FSC3,793与MINDS8,168，分别与
`operations/{fsc,minds}-macro-inputs-v1`已有三份population/candidate/input-receipts
逐内容比较，并验证原manifest记录的文件hash，全部一致；新增source_rows顺序与
人口一致。未重新读取远程cache或真实目标gold，也没有目标指标。
