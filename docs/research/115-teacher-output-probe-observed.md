# 效率提示对照终态与任务冻结缺口

2026-11-13本地时间，unix1794526812.2410135核验PID3832790已消失，6个逐请求及
整体终态齐全，距启动604.60秒。此前一次时间守卫在间隔未满时退出，未查询进程或
结果；本次是启动后首次实际状态观察。旧进程不再轮询，不自动重试。

6条HTTP200、executor退出0，4条stop且结构完整，2条length且正文空。3条历史成功
请求继续成功，24个判决与历史一致；3条历史失败请求仅1条恢复。7个作者参考目标中
5个返回且一致、2个无输出保留分母；总计32个完整判决，其余27个没有目标评分。
不得将返回目标一致或24个结果不变称为独立语义准确率。

输入133,683、输出14,566、合计148,249 token。两条失败各用满4096输出，其中推理
分别4094、4096。失败参考表达是“please play my pop playlist”和“play that last song”；
恢复项是“how old is cher”，8个类型均为negative，已阅读证据说明。历史成功项没有
语义状态变化。此处阅读为作者检查，不构成独立审核者裁决。

[观察报告](teacher-concise-output-observed-v1.json)与
[独立结构检查](teacher-concise-output-observed-check-v1.json)绑定全部原始请求、响应、
usage、终态、参考分母及旧结果。检查重建确认实际发送仅追加系统效率提示，其余
参数、用户问题和语义规约一致。固定成功/失败分层仅用于诊断，非随机质量队列；
模型随机性未控制，不能把净增1条成功解释为可靠的因果收益。

结论是当前thinking-enabled/4096执行方式仍不能保证完整输出。软提示不是服务端
推理预算，不能据此释放128组质量队列或全人口标注。当前新增语义训练准入0。
本轮不增加预算、不改本体、不扩模型任务，也不派发训练。

后续收口以doc23为唯一基线入口：先确定可完整返回的标注通路及独立审核身份，完成
固定质量队列，再进行全量语义标注和审核；同时完成现有任务的输入表示与上下文预算。
随后冻结任务包，进入一个共享runtime的GRPO/RLOO工程验收，最后才是两阶段训练。
不得用继续堆叠规则、校准探针或两步CE记录替代这些退出条件。


## 另一教师路由的有界诊断收尾

原接口的模型列表仅提供别名，不证明模型可调用或其上游身份。独立只读GET第一次
requests客户端CA失败，第二次复用固定executor的urllib系统CA成功，未禁用TLS；
私有原始响应保存在.local/research/teacher-provider-inventory-v2。最小请求
qwen3-235b-a22b-instruct-2507返回404 model_not_found；qwen-plus返回200/stop，
只证明该提供者别名的最小请求成功，不表示已锁定上游revision或独立审核身份。

随后同一固定六案例仅将原请求model改为qwen-plus并移除提供者thinking字段，
原messages/本体/问题和每条4096输出上限保持不变；并发2，无重试。
2026-11-13 00:24:13 UTC（启动后617.31秒）首次普通观察session79004，进程exit0，
6请求及总终态齐全，不再轮询该session。独立原wire、响应、终态、usage和全部判决
重建见qwen-plus-reviewer-route-check-v1.json，观察记录见
qwen-plus-reviewer-route-observed-v1.json。

6/6 HTTP200/stop，但严格格式仅3/6合格，另3条返回裸数组而非decisions对象；
不自动包裹数组挽救结果。7个作者参考目标：3一致、3格式失败未返回、1分歧。
分歧为来源1887的music_genre：作者negative，模型unavailable；模型理由自身说
未表达音乐风格类别，标签与理由需进一步语义审查，不能因理由似合理就改答案。
旧3个有效请求共24判决，当前只有8判决可比较且全部相同，16因格式失败保分母。
本次input136827/output3593/total140420；不是代表性准确率或全量成本估计。

结论：没有通过监督或审核通路准入；仅提供有界诊断证据。默认教师配置不切换，
不批量派发128组或全人口、不继续自动重试，不改变本体/标签以追随教师输出。
既有GLM校准与效率提示结果继续保留。新增语义准入0，训练未派发。

补充逐条读取S04/来源1887全部八个输出及其原始共享规则展开：music_genre的
unavailable与“未表达genre”的理由冲突；music_descriptor还声称history-relative
selection不属定义，然而输入规则明确包含该项。后者是未预登记的额外答案诊断，
不加入原7个参考目标的分母。作者建议分别negative/positive，仍待独立裁决，
未改规则或训练标签；记录configs/reviews/qwen-slot-reviewer-disagreement-triage-v1.json。

针对裸数组另建六例通用JSON外层说明对照，仅系统提示追加结构说明、原user所有
字段不变；并发2、4096/请求、无重试，与原qwen-plus探针逐字段身份对齐。
session12918于1794529696.637826启动，下一普通观察不早于1794530296.637826。
本对照与2048条质量输入包区分：前者已派发六例，后者API_sent=false。
不自动放宽parser挽救旧输出；格式修复也不免除上述语义审查。


### JSON外层说明对照终态

启动后631.97秒首次观察session12918，exit0，6请求及总终态齐全，不再轮询。
HTTP及严格格式均6/6，通过本六例的结构条件；原规则、user消息及参数逐字段核验，
仅系统序列化说明改变。原始输出、终态、usage和完整判决独立重建，见
qwen-envelope-contract-check-v1.json。共137607输入/1875输出/139482 total tokens。

语义校准未通过：7参考只有2一致、0失败、5分歧，五项均expected negative而返回
unavailable，理由却描述缺席或被排除的角色。全部48判决为25negative/21unavailable/
2positive，S03两条均八项unavailable。与原GLM有效24判决相比13同11变；与前轮
qwen-plus可比较的24判决相比21同3变，详见qwen-envelope-contract-paired-output-v1。
这些比较来源不同，不能混成一个分母；六个定向案例也不能估计总体准确率。

逐项理由见configs/reviews/qwen-envelope-semantic-triage-v1.json。不能全局把
unavailable转成negative挽救指标；真正歧义仍须保留。格式和语义的收益不能混称。
当前prompt/model不准入128组审核或全量标注；已冻结质量输入仅保留为完整输入证据，
没有执行。默认教师、本体、parser及训练状态均不变，不继续自动重试该六例。
后续须先确定能正确消费三值规则的审核配置与独立裁决通路，再冻结预算执行代表性审核。

### 三值覆盖校准（输出观察前登记）

六例对照的7个预登记参考全为negative，不足以检查正例和真实歧义是否被破坏。
因此从当前意图inventory-v5与槽位current-rules-v14中仅取规则兼容且无相互冲突的
作者参考，按固定hash为每族positive/negative/unavailable各选两个不同来源。
共12目标、12完整面板请求；未按教师输出选择，所有参考仍为作者建议而非独立gold。
未决只从已评估的明确unavailable取样，不从未评估或过期记录填充。

`tools/jev/build_three_state_calibration.py`生成
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/three-state-calibration-v1`，
report SHA256为`9ac7580990dbeb4dfc757709f26b93ab2ff2774b7e2b4e57ed15ca7cf9a07755`。
全部筛选、完整面板及原始TRAIN metadata独立核验；与128组质量审核按source_group
无重叠，与此前六例有来源2768重叠，不称holdout。证据为
three-state-calibration-input-check-v1.json及three-state-original-metadata-check-v1.json。

固定两臂：envelope沿用上一轮完整系统提示；envelope_three_state仅追加通用三值
判决说明，区分角色/意图存在、明确缺席与真实歧义，以及值不可解析和类型可识别。
没有案例答案或样本主题提示；原user中的规则、完整问题、参考文本完全不变。
原本体、actor二元输出、mask/reward不变；教师unavailable不成为actor新增类别。

24请求、qwen-plus提供者别名、并发2、4096输出/请求、累计98304输出上限、无自动
重试。session91568于1794530707.7099957启动，第一普通观察不早于1794531307.7099957。
配置见configs/reviews/slurp-three-state-calibration-candidate-v1.json。
预登记要求：所有请求完整；候选12个参考目标全部一致，包含4正例、4负例、4未决，
才能通过这一小型校准条件。即使通过也不是独立质量审核或全量准入。

配对评分工具score_three_state_calibration.py要求全部终态、真实wire及一致传输配置；
按族/状态报告三值混淆、失败分母、目标配对及全输出变化，不挽救裸数组/截断。
38项相关测试通过，含冲突与过期参考排除、不完整运行拒绝、失败保分母及完整128组
质量链路。当前无本次模型输出结论；所有标签仍未新增准入，不训练。


### 三值配对校准终态

session91568于启动后636.36秒首次普通观察，exit0；24/24请求和总终态齐全，
不再轮询或重启。两臂各12/12严格格式合格，各返回95判决；12个作者参考分别
7/12与8/12一致。四正例两臂均4/4，四负例均3/4；四未决从0/4到1/4，
两臂均未通过预登记校准条件。完整面板95个可比较判决中13个改变，不能仅报告
一个目标恢复。原wire/response/usage/终态与全部分母独立复算，见
three-state-calibration-observed-check-v1.json及观察报告。

envelope输入160539/输出3581，补三值说明输入162771/输出3738；
总输入323310、输出7319。该提供者别名未认证上游revision；定向作者参考不是
独立准确率，单次配对也不证明确定性的提示因果收益。

逐项理由见configs/reviews/three-state-calibration-triage-v1.json：
来源2768仍用“没有特定playlist”的理由返回unavailable；12868从裸Uber推组织/
provider，12855新臂靠IRCTC外部交通知识和隐含服务推positive；2401两臂选定
一种saving附着解释而作者保留歧义。869恢复作者未决只是局部现象。
不按教师多数改gold，不全局重映射未决，也不通过扩充规则追随这些输出。

结论：格式通路可完成，但这两个prompt/model配置均未取得语义准入；不批量派发
128组审核或全人口，不继续自动重试或提示词扫描。独立审核/裁决通路与完整规则
消费仍未解决；全量监督、工程发布及正式训练仍未完成。本轮无训练。
