# 当前语义输入编译与上下文实测

2026-11-13（Asia/Shanghai）。本记录是任务输入诊断证据；统一基线仍以[doc23](23-task-engineering-rl-baseline-lock.md)收口，不新增工程发布或算法配置。

## 本次完成

新增设计编译器 tools/jev/compile_semantic_actor_input.py，读取 menu-v3/readout-v2、意图 catalog-v5、槽位 package-v14。原生 Final60 描述、65 个 I_present 和55个 S_present 的语义命名空间分离。17固定面板每轮覆盖全部120谓词；运行输入不含 gold、参考转写或审核来源，问题不随音频路径或样本标签改变。

每面板保留完整选中意图卡及共同政策；槽位保留选中类型、相邻边界及一跳邻居的完整配置字段。shared_rules_v1 只共享完全相同的字符串，可逐字段还原，没有改写或删除规则。Direct/Q_input/Q_credit 知识相同；Q_input/Q_credit 整个请求相同。Final候选按实际输出符号绑定，不把显示顺序当作输出顺序。该编译器尚未接入生产训练器。

5个新增编译测试与既有读出/面板测试共17项通过。独立脚本直接从原目录、政策及边界重建34个模板全部字段，核对25个来源pin、120谓词和153种布局/实验臂绑定，见 semantic-actor-input-check-v1.json。这是结构复核，不是独立语义审核或音频复听。

## CPU processor实测

冻结目录：/mnt/cephfs/xingchao/experiments/multimodal-jev/semantic-actor-context-v1。
使用17面板 × Direct/Q_input × 两条真实学习音频，共68请求。Q_credit已核对与Q_input请求完全相同，没有重复执行。
两条音频依据历史81,026条完整processor记录选出：40,513学习录音的中位长度41,979采样点及最长373,760采样点（16kHz，最长23.36秒）；历史记录与当前学习人口音频hash一致。

| 实验臂 | 请求数 | 最小tokens | 最大tokens | 超8192 | 超16384 | 超32768 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 34 | 6501 | 30264 | 28 | 26 | 0 |
| Q_input | 34 | 6589 | 30972 | 28 | 26 | 0 |

最长为 semantic-07 / Q_input / 最长原始音频，30,972 tokens，含304 audio tokens。各面板两音频的总长度差和所有读出位置差均恰好等于音频token差；全部68条映射、位置、模板绑定与笛卡尔覆盖已独立复算。

容器退出码0，运行约31.97秒；首次实际观察在启动后628.33秒，遵守十分钟间隔。容器已终态，不再轮询或重启。17面板中13个连16K都不满足，因此旧8K不能作为当前实验默认。32K仅覆盖本次测得的68个输入，不是完整学习人口、增强波形、原生符号干预或训练显存验收。

## 镜像与证据边界

本地固定训练镜像未发现，ssh npu 连接超时；连接失败不能推断175服务器或原作业停止。本次使用本地独立CPU镜像，不挂NPU、不联网、不加载模型权重：

- CPU镜像：sha256:fdd4a28bb7515717e9ff1988669b4ae7c24588091cf0b73f05abff1f6ff6f87b。
- 原固定训练镜像：sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c，未替换。
- CPU包版本：transformers5.17.0 / tokenizers0.23.2 / torch2.13.0+cpu。
- 运行记录的processor源码SHA256：5ff7ace057ae959fb0cd551775cbbfdad8c003eb731bf0f1f39bfc6452ce86fa；版本和hash由该诊断运行报告，不等于训练镜像已复验。
- config SHA256：78393702043538b0fb7bb3a6adcf471d73e0d27f1ab11c1a6a43d5fc2639435d。
- records SHA256：b91f8d7694213a145f088d7fd8bce88e1df6f3e93490d6dfcbdeb74505bb46c1。
- result SHA256：846ec52993c64f136b086933d69a3294ebe82d41c125328c2f4ac7cdde14512c。
- 独立复算：semantic-actor-context-check-v1.json；核验271个文件pin、终态及全部68记录。

这些结果只证明当前冻结候选输入在该CPU processor上的长度和结构。没有模型前向、训推概率对齐、语义正确性或任务指标结果。selected_context_limit仍为null，context_accepted=false，training_dispatch_allowed=false。

## 对稳定基线的影响

先完成任务输入与语义监督发布，再锁工程。当前长度主要来自规则内容：最长面板除4个选中槽位外需供给23个相邻类型与19条边界；共享字符串后仍很长。不能因进程exit0就宣布上下文通过，也不能截断输入、删长音频或跳过困难面板。

下一步集中做两项：一是收敛可执行标注及独立复核通路，完成全量语义监督准入；二是比较保持语义的输入表示与定稿上下文的资源预算。规则简化如改变语义必须走任务修订及对应参考复核；纯表示变化须通过字段可逆和行为一致性检查。32K不自动获选，也不因模型位置上限65536就默认其训练成本可接受。

这两项完成后，才按doc23验收统一BF16 LoRA运行时、GRPO、RLOO-clipped以及可选PPO-critic，最后发布数值配置并开展两阶段训练。本次没有更改训练损失、算法实现、模型权重或训练启动状态。

## 完整重复块共享的收尾对照

在相同CPU镜像、processor、最长原始学习音频、17面板及两臂上，比较完整重复
dict/list的可逆共享。34请求完成，容器exit0，耗时26.20秒；完成通知后核验，未轮询。
Direct最大27,318、Q_input最大28,026 tokens；分别12/17和13/17仍超过16K。
相对原34条配对请求总计少54,894 tokens，最坏输入30,972→28,026，约减少9.5%。
面板14–16没有可共享块，包装反而各增加85 tokens。

工具share_semantic_blocks.py和编译器可选shared_blocks_v1保留作诊断；默认仍是
shared_rules_v1。26项相关测试通过；独立重建34份完整契约，核验25任务pin、274运行
文件pin及全部配对位置/映射，见[独立记录](semantic-block-context-check-v1.json)。
结果目录为`/mnt/cephfs/xingchao/experiments/multimodal-jev/semantic-block-context-v1`，
result SHA256为`fba3e087a1ce62d814638f940855b98ddbaee97bc6f4a2f3faea030d4e556f46`。

结论：没有降低所需上下文档位，不选为主线，不继续扩展压缩变体。字段可逆不等于
模型理解等价；上下文选择仍null。后续针对定稿任务输入评估训练资源，不能凭本次
CPU长度诊断自动选择32K、宣布全量/NPU验收或启动训练。

## 全量原始音频的长度与输入规模投影

2026-11-13补充：在同一CPU镜像检查实际processor及feature extractor源码，使用
已固定learning/selection音频元数据和当前68条实测记录，分离固定文本长度与音频
占位长度。全部音频为原始16kHz、0.5–30秒范围，默认480,000样本padding且不截断：
有效mel帧为ceil(samples/160)，每100帧输出13音频token，余段为ceil(余帧/8)。
34个面板/实验臂各自两个实测锚点的非音频长度与读出偏移完全相同。

| 人口/实验臂 | 录音 | 一轮17面板请求 | 输入token总量 | 中位长度 | P95 | 最大长度 |
|---|---:|---:|---:|---:|---:|---:|
| learning Direct | 40,513 | 688,721 | 12,849,921,960 | 20,509 | 29,983 | 30,264 |
| learning Q-input/Q-credit各自 | 40,513 | 688,721 | 13,279,967,455 | 21,217 | 30,691 | 30,972 |
| selection Direct | 9,226 | 156,842 | 2,926,307,004 | 20,509 | 29,983 | 30,113 |
| selection Q-input/Q-credit各自 | 9,226 | 156,842 | 3,024,240,994 | 21,217 | 30,691 | 30,821 |

所有实验臂均14/17面板超过8K、13/17超过16K，比例分别82.35%/76.47%；
当前规范布局的长度投影没有超过32K。Q-input学习输入中音频token仅0.1944%，
因此删长音频或做静音裁剪不能解决规则占主导的长度问题。一次完整三臂selection
为470,526请求、8,974,788,992输入token；尚未计非规范布局或更多检查点。

这是基于全部元数据的**投影**，不是对全部请求重新运行processor。工具
`project_semantic_context_population.py`核对固定发布hash、learning/selection身份、
当前任务依赖及原探针终态，不读取标签选择面板。独立脚本用逐块三层stride2公式
重新计算长度直方图、分位数和各面板统计，并核对40,513条learning音频长度/hash与
历史两轮81,026条processor记录一致；3项测试验证边界、分母及拒绝不一致锚点。
见[独立核验](semantic-context-population-projection-check-v1.json)。

共享报告位于`experiments/multimodal-jev/semantic-context-population-projection-v1/report.json`。
没有新音频解码、全量processor执行、增强波形验收、模型前向或固定训练镜像验收。
报告中的sum(length²)仅为注意力规模代理量，不能换算为实测显存、FLOPs或耗时；
token总量也未计padding/packing、缓存和训练反向。G16不会机械乘成16次音频前向。

对任务冻结的新约束：保持完整人口、语义规则和现有17面板候选，正式发布前必须
显式核算一轮完整谓词曝光及评测的预算。当前完整日程不能默认是低成本的分类训练；
不以小参考集、gold选面板、截断规则或重复压缩分支绕过。上下文仍未选定，下一步
需要在任务输入收口后评估这一真实长度分布的四卡资源可行性及有效batch，当前不训练。

## 固定训练镜像的配对诊断：已启动，尚未观察结果

175的SSH和固定镜像52ba94e1已实际核验。原68请求的模板、音频和源码快照保持，
仅修改诊断runner的镜像身份报告：不再硬编码fixed_training_image_used=false，
且该字段须与外部docker inspect的真实镜像一致。本轮没有更改任务内容或生产trainer。

新目录为`experiments/multimodal-jev/semantic-fixed-image-context-v1`，
容器`85f192c1f982a8be1be770b7142124831b8b57495d261a7221d3032264b9ceaf`。
启动核验为running，无NPU设备挂载、网络关闭，4CPU/16GB内存/300秒进程期限，
不加载模型权重。记录见[启动证据](semantic-fixed-image-context-launch-v1.json)。
启动状态观察时间1794533168.6680999，首次普通状态观察不早于1794533768.6680999；
期限不代表已经退出，连接超时也不推断终态，不自动重启。

配对比较工具`compare_semantic_context_runs.py`检查双方成功终态、真实镜像、
全部依赖hash、输入身份与完整笛卡尔覆盖，再比较长度、音频token、Span位置、
候选映射和完整token序列摘要。不同输入拒绝作为镜像效果比较；相同长度而token
或位置不同也会报告。11项相关测试通过，当前尚无真实配对结果。
即使上述比较一致，也不证明声学特征张量、模型概率、完整人口或NPU资源一致。
上下文仍null，任务/工程/训练准入状态不变。

### 固定镜像v1失败与CPU环境修正

首次普通观察距启动状态检查612.956秒，原容器已exit1、无OOM，未生成result或
执行processor请求。错误为torch自动加载torch_npu时缺少libascend_hal.so；
这是无驱动挂载CPU容器的导入失败，不能解释为模型/输入不支持。
[失败证据](semantic-fixed-image-context-failed-v1.json)保留原容器、日志及终态。

新目录semantic-fixed-image-context-v2保持原镜像、源码和68个输入，只在CPU诊断
环境加入TORCH_DEVICE_BACKEND_AUTOLOAD=0，遵循torch错误提示关闭加速后端自动加载。
不修改NPU训练配置、不挂载驱动、不另建镜像。v1已终态，不再轮询或原位覆盖。

v2容器2368b7443633e0ed4bc1ddf0c13a9b9c6fed2387e4d5adfc64a89019ab1189db
于1794533823.491032核验running、设备为空；首次普通检查不早于1794534423.491032。
[启动证据](semantic-fixed-image-context-launch-v2.json)只证明派发，尚无输出观察。
比较器补真实音频hash及launch/config/raw-inspect/container/command交叉核验；4项测试
通过，覆盖成功文件链及哈希、失败退出、输入变化和错误命令拒绝。没有当前输入长度
或模型一致性的新结论，不启动训练。

## 显式eager attention的容量下限

对当前`span_attention_alignment.eager_causal_attention`逐行检查：TP1下16个查询头
完整物化BF16分数矩阵，随后`scores.softmax(..., dtype=torch.float32)`产生FP32
概率；局部变量scores在该表达式期间仍存在。仅这两个张量需要
`batch × 16 × length² × (2+4)`字节。此处batch=1，不计BF16概率转换、mask、
Q/K/V、权重、优化器、其他层、反向或allocator，因而不是峰值估计或可用容量保证。

| 规范Q-input长度 | 两张量合计GiB |
|---|---:|
| 8,192 | 6.00 |
| 中位21,217 | 40.25 |
| P95 30,691 | 84.22 |
| 最大30,972 | 85.76 |
| 32,768 | 96.00 |

按175每卡名义64GiB容量，上述两张量可容纳的最大整数长度仅26,754。
全量投影中的semantic-07对所有学习/选择录音、Direct/Q-input都超出该下限；
每臂分别影响40,513/9,226请求，占完整17面板日程5.88%。即使其他面板未超这个
粗下限，也不代表完整前向/反向能放入显存。四卡数据并行及FSDP参数分片不分摊
单条请求的注意力矩阵；microbatch已为1，梯度累积也不降低这次前向的矩阵尺寸。
这不是NPU实测OOM，也不将TP4作为当前DP4/TP1配置的替代。

工具`audit_semantic_attention_capacity.py`及[独立复算](semantic-attention-capacity-check-v1.json)
绑定当前源码、模型config及全量长度直方图。共享report SHA256为
`0ad1129bf7028adf95882705e4c14ca8d736e2fd6979888072dd8e30195cc206`。
本地pytest为1通过/1因无torch跳过；固定CPU诊断镜像另用三个真实小张量核对
BF16分数与FP32概率的字节数及batch/长度缩放，全部一致，不宣称NPU容量验收。

**工程结论改变：当前完整语义输入不能直接继承旧显式eager TP1运行时。**
保留任务人口和规则；任务冻结后需在共享工程中实现内存可承受的attention，随后
重新验收actor/vLLM概率、训练更新和恢复。融合/分块实现的数值顺序可能不同，
不能只替换rollout kernel或降低dtype后沿用历史一致性凭证。当前不改算法、
不执行NPU试错、不自动选择32K。固定镜像processor复测仍是独立待收尾事项。

## 固定镜像v2首次观察未取得远端状态

2026-11-13 01:47:03 UTC，在规定首次观察时间后执行原容器inspect，SSH于banner
exchange阶段超时，exit255。未取得docker状态，不能推断容器成功、失败或仍在运行；
没有执行配对比较，也没有重启或重复派发。单独源码检查的既有ps会话92418同样
exit255且无输出，源码检查远端进程状态未知。见
[连接失败记录](semantic-fixed-image-context-observation-unavailable-v1.json)。
下次普通连接尝试至少距该记录600秒；仍检查相同容器，不因客户端超时新建实验。
该连接问题不改变已完成的源码容量核算，也不提供固定镜像输入一致性证据。

## 服务器ready通知后的核验

用户通知175/181 ready后立即只读核验：181（A800I-03）可登录，8张910B4-1健康，
但每卡约61,200/65,536MB HBM占用，两个saea共享推理容器仍在运行；固定52ba94e1
镜像不存在。没有停止服务、改变用卡或改用其他镜像。175直连、禁用连接复用、经181
跳转均SSH banner超时，原v2容器终态仍未知，未重启或重复派发。

另发现clean发布目录700阻止181普通账号访问；依此前共享权限授权将该单目录修复
为777，MANIFEST内容hash不变。181普通账号现可读取clean MANIFEST、当前评测名单
report与v2配置，三者hash与本地完全相同。这只证明三个文件可跨主机读取，不等于
全部模型/音频权限、镜像、空闲设备或训练准入。详见
[核验记录](../engineering/host-readiness-20261113-observed-v1.json)。下一普通连接尝试
按该记录的next_regular_attempt_not_before执行；任务→工程→训练顺序不变。

## 固定镜像v2终态已确认（取代此前未知状态）

主机故障取证时已核实原2368b744容器01:35:57 UTC退出1，OOMKilled=false。
随后仅收取该终态容器日志，没有重新派发。报错发生在Qwen3ASRProcessor导入链：
Transformers的flash-attention集成显式import torch_npu，因CPU容器没有宿主机
libascend_hal.so而ImportError。`TORCH_DEVICE_BACKEND_AUTOLOAD=0`只关闭自动
加载，不能阻止该显式导入。完成请求0，未生成可配对测量；不执行成功结果比较器。

见[失败证据](semantic-fixed-image-context-failed-v2.json)。当前context配置已标终态、
不再轮询/重启，旧连接失败和等待计划保留为历史。失败并不证明固定镜像在正确挂载
驱动的NPU环境不可用，也不能把它归为上下文OOM或175宿主机Ceph崩溃原因。
后续诊断必须先符合新的本地热状态规约、核对实际依赖挂载；本轮不新建第三次探针，
上下文与工程准入继续未完成。
