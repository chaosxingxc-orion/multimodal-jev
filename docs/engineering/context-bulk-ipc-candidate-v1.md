# 逐条前向不变的整批IPC候选

当前正式runtime有两层串行：`ContextEngineClient.score`每输入提交一次本地tensor文件、
一次socket往返；worker逐条调用`engine.score`，vLLM的`max_num_seqs=1`。
本候选仅处理前一层，不改变模型batch或声学特征布局。

独立实现 `tools/reports/context_bulk_rpc_candidate.py` 增加score_many：最多128个
无标签CPU输入以一个不可变.pt提交，服务端先验证完整payload字段与hash，再顺序逐条
调用现有engine.score，并逐条验证policy/input/position/probability绑定。响应顺序保留；
重复输入仍对应独立query槽位，后续采样seed由原训练器决定，不在服务端生成动作或奖励。

客户端复用现有策略切换认证与sequence协议，一批内固定policy；服务端在每次forward前
及整批结束检查policy未变化。每个record都完成身份验证后才向客户端返回整批。
任一失败后不允许重试或单条fallback；部分已执行时保留payload和失败状态，不能把后缀
当作新成功批次。关闭失败连接不发送额外close命令。

完整128输入在调用score_many时：文件持久提交128→1、socket请求/响应往返128→1，
但模型forward仍128次。既有单条score兼容入口仍单条往返；现有session调用方式尚未改，
因此**当前正式训练没有获得这项优化**。reference misses和behavior可分别按组提交，
策略切换边界仍须保留；把两者混成一个策略批次是错误的。

## 本轮验证

本地torch CPU，候选9项+既有RPC8项共17项检查通过/2.43秒。
使用真实Unix socket、torch序列化、文件hash及原bind_rollout；引擎、策略权重认证和
测试临时目录的存储准入使用替身。覆盖3/128输入、变长输入、重复输入、切换后策略、
逐条record和G16采样动作相同、半批失败不重试/保留文件、payload越界/哈希/标签注入/
空批/超容量拒绝。没有NPU forward、实际vLLM加载或吞吐测量。

## 接续边界

先在独立候选中接session的“整批前向→逐条绑定采样”接口，再用同一checkpoint和真实
音频检验单条与整批传输概率、采样、更新及恢复。此方案不改变vLLM max_num_seqs，
若后续启用实际模型并行batch，需作为另一数值版本处理，不能沿用本传输测试的结论。
只有实测表明文件提交/RPC占时显著，才能报告其吞吐价值；此前reference约32.61秒
含模型计算、切换、缓存写入，不等于可由整批IPC省去32.61秒。

当前运行/排队的源码、batch、奖励和学习人口不改；固定reference跨实验复用仍是另一
候选，与本优化分别计成本，不叠加未测量收益。候选不能成为已准入简单RL继续执行的门槛。

## 独立session接入已完成

`tools/reports/context_bulk_session_candidate.py`继承原session的初始化、策略接受与失败
生命周期，只在独立prepare中使用score_many。预先验证当前policy、完整truth、真实query
权重、padding与无标签wire；缺失reference按真实输入去重后整批计算，整批结果全部绑定
成功才开始缓存写入。缓存IO失败仍可能留下已验证的不可变前缀，不保证跨文件事务原子性。
behavior保留所有槽位，包括重复padding，不跨更新复用。

behavior响应通过一次性有序记录对象交给原OnlineContextBatches.prepare；原sampling seed、
概率绑定、G16动作、辅助系数及优势配置没有复制改写。记录对象仅存于本次prepare调用，
每个槽位验证输入/位置后消费一次；多余、缺失或乱序记录均失败。服务端只接inputs和positions，
不发送标签、mask、奖励或来源类别。

session9项、bulk IPC9项与原session5项共23项CPU检查通过/4.38秒。包含128槽位、127个
不同输入+一个padding，两次更新对比原session的全部query receipts/分布/G16动作/选项，
原engine调用顺序也一致；新session恢复后动作一致且只做behavior前向。恢复不声称真实
optimizer或NPU状态恢复，因为此处尚未接actor。

冷批事务数为2（reference、behavior），warm批为1。另检查不完整/乱序reference不发布缓存，
behavior失败保留已验证reference但不能ack/retry，非法query在forward前拒绝。
测试使用真实本地reference文件缓存及原绑定/采样逻辑，engine和存储准入为替身；socket
测试与session测试分开进行，尚未做真实服务+音频+actor的联合验收。

正式session、RPC和已发布包未变。后续需要固定镜像联合路径与真实NPU/恢复/耗时对比，
不能由减少事务数直接宣布训练效率提升，也不插队现有GRPO→RLOO评测流程。

## 跨进程联合检查

新增 `tests/test_context_bulk_process_integration.py`，使用spawn独立worker进程、真实
Unix socket、tensor文件提交与reference文件缓存，串起完整session的128槽位两次更新。
首批4个命令（切reference、reference整批、切behavior、behavior整批）；第二次更新前
切新策略，再执行一个behavior整批，总6命令。概率记录、原G16动作及辅助选项与原
session相同；127个reference条目及重复padding身份不变，正常退出后交换文件清空。
模型引擎、激活权重认证及测试临时路径准入仍为替身；不宣称真实模型加载或actor训练。
本地新旧合计32项通过/9.16秒。

固定镜像首次检查 `cc79f506c4977b7a92c8bcbf03faa5e10b9b0b31a50a5ef643cb7ac5b3bae846`
exit1、非OOM，28通过/4失败。四项均在AF_UNIX bind时因pytest临时目录过长失败，
跨进程联合项通过；见 `docs/research/bulk-context-tools-fixed-image-cpu-v1.json`。
后续v2按既有规约将相同本地cache额外挂载到/jev-tmp并设置TMPDIR，1,076项实际包文件
hash与v1完全相同，未改算法或测试来绕过错误。固定镜像终态以独立收集报告为准。

v2已独立收集完成：容器`36d1df79505f2cb4a166253672d7c49c6851b29aa6e433ca61f102883182b253`
exit0、非OOM，32项通过/60.56秒，无失败、错误或跳过；1,076项源码身份已绑定。
收据为`docs/research/bulk-context-tools-fixed-image-cpu-v2.json`，原始日志位于
`.local/decision-objective-research/bulk-context-tools-cpu-v2-collected/run/pytest.log`。
这闭合了固定镜像CPU跨进程检查；引擎与激活仍为替身，没有实际NPU概率、actor恢复
或吞吐收益证据。候选维持独立，正式GRPO与后续排队包保持原版本。

## 实际服务入口与有限对照准备

新增独立`tools/reports/serve_context_transport_candidate.py`，明确选择原单条协议或整批协议，
均实例化既有`SwitchingContextRollout`。不改变max_num_seqs=1、数值profile或模型forward；
启动前核对自身pin、存储与短socket路径，一个服务仅接一个控制器，异常不自动重试。
`measure_context_transport_candidate.py`核对完整策略、服务ready身份及无标签输入hash，
一轮warmup后固定两轮计时，比较完整record和固定seed的G16采样收据。
计时包括文件/RPC/顺序forward/绑定，不含启动、策略切换、actor或checkpoint成本。

服务真实socket路径、测量比较器及既有IPC/session在固定52ba94e1镜像CPU共45tests通过，
60.99秒、无skip，容器52677bb768992307 exit0/非OOM，1080文件身份已独立收集。
见`docs/research/context-transport-tools-fixed-image-cpu-v1.json`；模型/激活/临时路径准入
仍使用替身，未构成真实NPU测量或CLI全链准入。

真实输入采用冻结TRAIN epoch0首个global batch原顺序128条，不按标签或模型概率选样。
从已独立审核的缓存只读收集，逐项核对缓存字节、binding、五种tensor hash，物化为仅含
inputs/positions的payload。token范围2808–2911；不覆盖全量极端长度。
输入SHA256为`4b8791ee0abfe3c40613c24e80737f89630e56321f8f8e14c2cce28237cc9234`，
目录`.local/decision-objective-research/context-transport-audio128-v2`。
首次宿主tar读取root所有缓存遇权限失败已保留；v2改用固定镜像无NPU、只读挂载收集，
没有修改源文件权限或重新做processor。准备收据为`context-transport-candidate-readiness-v1.json`。

181只读观察显示八卡均被两个现有服务占用，且该固定镜像不存在；本轮没有在181创建
NPU容器，也不抢占175正式租约。真实音频对照、独立结果收集和必要的actor验收仍待。
正式驱动1078项依赖再次复核无变化，现有训练/评测继续。
