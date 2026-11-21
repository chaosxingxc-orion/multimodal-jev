# Stage1 终点 → Stage2 难例与简单回放训练集

> 难例视图训练的当前工程候选入口及证据见 [Stage2 v0.0.125 启动交接](stage2-view-launch-v0.0.125.md)。旧全量后端保持原准入边界。

本次交付是确定性数据工程组件：使用完成审核的 Stage1 终点全 TRAIN 概率缓存，制作冻结
训练视图、独立复算审核、生成 Stage2 数据配置并供 batch 消费器读取。无需模型人工挑样本。
不启动/更改当前 Stage1，不启动 Stage2；真实难例数量要等本轮终点评分完成才能确定。

## 数据流和选择标准

```
本轮 Stage1 最后 checkpoint + 全 TRAIN 声学输入
  → 既有 reference 构建/审核入口（模型推理仅执行一次）
  → 固定只读 reference（覆盖完整 TRAIN）
  → prepare_stage2_dataset prepare / audit
  → selection.json + batches.json + consumer.json + MANIFEST.json
  → prepare_stage2_dataset bind → 新 Stage2 数据配置
  → TrainingDataset / training_inputs → CachedContextBatches
```

模型必须直接听音频；采样使用这个模型的完整候选概率。只接受可信 clean TRAIN，禁止
selection/test/FSC/MINDS进入采样、阈值或比例优化。全部错误/边界样本保留：

| 分组 | 判定 |
| --- | --- |
| error | 正确候选不是严格top1（并列算错），或阈值输出集合不等于唯一gold |
| boundary | 输出正确，但正确概率−最强错误概率<M，或正确概率<t+M/2，或最强错误概率>t−M/2 |
| easy | 排序、集合、间隔、双侧阈值全部满足 |

默认t=0.5、M=0.1，必须与实际Stage1 checkpoint中的loss配置一致，否则拒绝。
按样本ID与seed317生成确定性顺序，不受输入文件行序或线程数影响。easy目标占最终样本数20%，
各个有easy样本的类别至少保留16条（不足则全部保留），之后按固定随机顺序补足。
若难例H条，目标easy数为ceil(H×0.2/0.8)，类别保底可提高此数，不足不复制。
类保底不是强制均匀类别重加权；调度把各类铺开但不改变总次数。

全难时保留全量；全简单时只保留各类保底并报告no_difficult_samples，不凭空制造难例。
初值不是最优比例，不自动用selection/test调整。难例偏噪声、类别分布改变和简单能力遗忘
需要后续完整评测验证；这里不承诺难例训练优于全量或迁移收益。

## 产物、复用与可信度

- selection：选中ID、全TRAIN难度诊断、各类数量及实际easy比例、margin/anchor覆盖。
- batches：每轮每个选中ID恰好一次，确定性类间铺开；尾批补齐但padding权重0。
- consumer：global batch=64×DP=128，total updates=epochs×ceil(N_selected/128)，
  checkpoint间隔由配置给出，始终包含0和实际终点。不会沿用旧634步预算。
- manifest：绑定完整source rows/task、reference及policy、Stage1 commit、loss及阈值、seed、
  采样实现及依赖源码hash。相同内容重复执行先重新核验再复用；不同内容拒绝覆盖。

reference继续完整覆盖TRAIN。训练视图只存ID及小型元数据，不复制音频、特征、reference或
模型。stage2配置引用原资产。完整候选问题不按gold删减，难度/预测概率不放进actor输入。
训练参数/奖励插件配置不受数据绑定改写。不同Stage2 adapter身份仍需独立LoRA迁移准入，
不能靠随意填stage1_policy_id绕过；当前数据绑定只放行同源policy身份。

使用同一文件锁做单实例，写入`.building`后独立复算，再原子重命名发布；崩溃中断目录
不可消费，也不自动删除不完整证据。正式产物必须有固定manifest hash。输出需在本地持久盘，
Ceph仅供共享输入只读；不得在Ceph做热写入或fsync。

## 入口和宏

模板：`configs/portable/operations/stage2-dataset.json`。
默认本地宏：`.local/config/macros.json`；项目路径、主机、输出路径均在宏中绑定。

| 宏 | 内容 |
| --- | --- |
| AUDITED_STAGE1_REFERENCE_JOB | 既有reference job的完整审核配置（不是PASS字符串） |
| STAGE1_DATASET_ENDPOINT | policy_id、stage1_commit_sha256、endpoint，来自本轮已审核终点 |
| STAGE2_DATASET_OUTPUT | 本地不可变发布目录 |
| STAGE2_DATASET_SHA256 | prepare返回的manifest hash；prepare时可设空字符串 |
| STAGE2_DATASET_RECEIPTS | 标准路径使用 release 收据；分步诊断时提供对应动作的收据路径 |
| STAGE2_INPUT_CONFIG | 待绑定Stage2配置的path与sha256 |
| STAGE2_BOUND_CONFIG | 新配置输出路径；不会覆盖旧运行配置 |

日常实验使用[统一实验入口](simple-experiment-admission-v1.md)，选 `stage2_dataset`。数据后端也可单独执行：

```bash
PYTHONPATH=src:.:tools/jev "$EXPERIMENT_PYTHON" -B -m tools.reports.prepare_stage2_dataset \
  --config configs/portable/operations/stage2-dataset.json \
  --local "$LOCAL_MACROS" --action release
```

同一进程只审核一次 producer，基于本轮实际发布且已独立复算的内容生成绑定，无需先写回 hash。独立取证可显式运行 audit；已有已发布数据可按指定 SHA 运行 bind。两者都重新验证内容，不以旧 PASS 收据替代。
绑定完成不等于NPU准入，输出始终`training_dispatch_allowed=false`。

新版成对证据采样使用同一发布与消费协议。入口为
`tools.reports.prepare_stage2_paired_dataset` 和
[`stage2-paired-dataset-dp6-v1.json`](../../configs/portable/operations/stage2-paired-dataset-dp6-v1.json)。
绑定已独立审核的当前终点 reference、可信 clean TRAIN 与原生 TRAIN 元数据、局部线索规则、受保护探针和独立输出路径。
原生转写仅用于离线匹配同词录音或寻找局部词编辑；actor 仍直接听原始语音，训练标签不变。
同词保持对要求不同说话者；局部词对要求同一说话者。难例保留可核对的对照录音 ID，
简单回放从双边严格首选正确的保持证据中按种子与类别优先级抽取，数量向上取整，不被类别保底抬高。
完整候选排名用于发现错序或并列；NLL、Brier 分开记录，无固定分数阈值、Top‑K 或概率间隔准入门槛。
缺少独立判决依据的录音只保留诊断记录，没有新增 UNKNOWN 或组合能力监督。

```bash
PYTHONPATH=.:src "$EXPERIMENT_PYTHON" -Bm tools.reports.prepare_stage2_paired_dataset \
  --config configs/portable/operations/stage2-paired-dataset-dp6-v1.json \
  --local "$LOCAL_MACROS"
```

加 `--audit-manifest-sha256 <sha256>` 从同一输入独立重建证据、清单和训练配置并校验已有发布。

## 其他因素的固定脚本采集

样本入选、归类和审核只读冻结配置、可信 clean TRAIN、原生语料元数据及固定检查点原始概率。
脚本输出全部匹配 ID、原始分母和证据身份。语音模型直接消费音频与任务规约并返回概率。
文字元数据只供离线规则筛选，人工和 AI 均不逐条挑选或改标。诊断产物独立发布。

| 因素 | 固定配置与采集口径 |
| --- | --- |
| 候选定义和读出的位置 | [全 TRAIN 等义换序](../../configs/portable/operations/next-simple-full-train-unseen-probe-v1.json)复用训练编译器，对本轮全量录音编译原生和训练未见顺序。两个输入共用录音和候选定义。用同一检查点与同一推理后端收集原始概率，[评分配置](../../configs/portable/operations/next-simple-full-train-unseen-score-v1.json)按语义 ID 对齐，输出原生正确而换序失败、反向恢复、双对和双错的录音 ID。 |
| 上下文决定答案或证据不足 | [日历规则](../../configs/portable/operations/jev-evidence-calendar-train-v1.json)筛选可信的单日存在性问题。同一段音频配固定的完备、事实反转、缺决定性事实和缺无关事实四臂。缺决定性事实时依据两个相容世界给可判性监督，答案掩码保留为 null。[评分配置](../../configs/portable/operations/jev-calendar-contrast-score-v1.json)输出各臂指标及基准正确条件下的失败 ID，同一录音可以触发多个因素。 |
| 组合判决和模型容量 | [逻辑支持统计](../../configs/portable/operations/jev-logic-support-clean-train-v1.json)从全 TRAIN 原生类别投影 `IS/OR/NOT` 的真值覆盖。当前只计支持度。组合失败需要配对的模型输出，容量归因需要同输入、同训练协议的容量对照。 |

准备和重跑沿用同一套 CLI，分别使用
`tools.reports.prepare_next_simple_probe`、`tools.reports.build_jev_evidence_panel`、
`tools.reports.jev_logic_train_support`，均传入 `--config` 和 `--local`。
换序评分由 `tools.reports.score_next_simple_pair` 消费配对 HF 收据。日历评分由
`tools.reports.score_jev_boundary_panel` 消费四臂完整 HF 收据。
路径、清洗视图与原生元数据的 manifest 哈希、本轮训练布局与策略 ID 在本地宏绑定。
缺少变体预测时只发布诊断输入与人口。真实世界证据不足样本须有独立真值才可纳入训练。
视图复用原音频、特征和全量只读 reference；不改变已有冻结运行。

## 组件接口与并行

新组件注册为`release/stage2_difficulty_release_v2`，解析到`TrainingDataset`。
训练器/审核器可调用`load_training_dataset(config, rows=..., task=..., reference=..., execution=...)`，
或共享`training_inputs`；必须传入已认证完整reference与显式execution。
返回population/batches供原batch消费器使用，`TrainingDataset.stream(preparer)`直接复用
CachedContextBatches。不存在新源码字符串改写或动态导入配置路径。

io_workers控制release/reference哈希线程池；cpu_workers、chunk、window、min_parallel控制
概率分类和reference记录审核的spawn进程池。小任务串行，线程/进程数不参与采样结果身份。
每次报告实际执行宽度、在飞上限和墙钟，不保存逐条计时大数组。

40513条×60候选合成全规模一次配对测量：采样串行2.04秒，四进程3.47秒，结果hash完全
相同。此轻量分类负载的进程启动成本超过收益，所以默认min_parallel=50000，让当前规模
走串行快速路径；并行能力保留且有等价测试。这不是其他负载或规模的交叉点估计，也不是
真实模型推理速度。reference构建、历史producer日志审核成本不包含在这两个秒数中。

## 验收边界

CPU覆盖：原子发布/重复复用、全量与候选顺序、旧policy/commit/阈值拒绝、重算篡改检测、
串并行逐字节一致、各类保底、全难/全简单、真实batch消费与零padding、动态预算拒绝634。
本轮margin loss终点已接入reference构建器合法目标校验；旧纯InfoNCE历史读取保持兼容。

当前交付完成到**训练数据生产、审核与batch消费接口**。旧正式Stage2启动链仍有固定
40513/634的封包/独立输出审核假设，不可直接用它派发这个新数据配置。后续Stage2发布需把
已交付组件接到正式trainer/输出审核共同入口，并完成两步NPU验收；数据准备不冒充这一步。
本轮没有本轮真实endpoint/reference的新产物，不报告真实难例数量，不启动重复评分任务。

本次定向CPU回归：95 passed（18.93秒），JUnit证据
`/srv/multimodal-jev-xingchao/operations/stage2-dataset-v2/tests.xml`。
全规模合成对照报告：同目录`sampling-measurement.json`，其中样本数/难例比例均为合成构造，
不能作为本轮Stage1质量结论。未修改当前冻结运行包。
