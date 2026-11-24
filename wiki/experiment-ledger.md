# Jev 实验台账

近期横向结论见[近期实验汇总](reports/recent-experiments-2026-11-22.md)。

2026-11-22 UTC，JEV-206 的 **34/34 步**、源 TRAIN 阈值校准和五组留出评测均已审核。
JEV-204 正式重启的 **212/212 步**和三点 SLURP / MINDS 评测已审核；补充评测及全链路成本对照待完成。
JEV-201 已审核 **34/34 步**训练、全部登记的末点评测及变长 Choice 诊断，评测租约已释放。
后续补充评测固定为 SIB-FLEURS 英语话题和 MMSU 多义词。WolBanking77 仅保留历史诊断，YUE-PUB-Speech 不加入队列。[评测范围](../experiments/simple-speech-baselines.md)

## 编号和状态

- 讨论中的候选方案沿用当前研究 ID；只有用户明确决定另立实验时才分配新 ID。
- **实验 ID** 跟随研究问题和预登记协议，例如 `JEV-201`。封包修复、重新派发、
  断点续跑及同一协议的评测都追加到同一个 ID；增加对照臂用臂名区分。改变主要
  假设、目标、训练人口、起点或比较预算时，登记新实验。历史显示名保留，不追改冻结产物。
- **包版本** 继续使用 `v0.0.N`（开发态）或 `v0.0.rcN`（锁定基线），总是与
  运行家族一起写。历史上两条队列可使用同一数字；只写“198”之类的数字不足以标识产物。
  **运行句柄** 再区分具体窗口、容器、续跑和评测。历史回填按首个有效候选保留熟悉的
  数字称呼：199–201 记入 `JEV-201`，202–204 记入 `JEV-204`，后续封包继续沿用实验 ID。
- **分别登记训练、评测、研究结论。** 训练可记录拟定、CPU 就绪、有限验收、正式运行、
  退出待审核、已审、失败或停止；评测另记未派发、运行、待审、已审或结果缺失。
  当前运行须按带时间的现场核验；退出码 0 只说明进程结束。终态和独立收据齐备后，
  再填完成量、成本和结果。结论可为通过、未通过或证据不足。

## 实验队列

从此表选一条待开工的实验，按[五步实验包规约](experiment-package.md)执行。优先级数字
越小越先；先完成已开工实验的后续阶段。选中一行用于准备，须由现有发布入口确认依赖、
设备和派发范围。队列状态是规划记录，现场运行以对应句柄和收据为准。

| 实验 ID | 优先级 | 状态 | 协议 | 下一步 |
| --- | ---: | --- | --- | --- |
| JEV-205 | 40 | complete | [194 固定目标的排序修正](../experiments/tpo-ranking-repair-205.md) | 五组评测已审核；源域 selection 未达标，保留 194。[执行报告](reports/JEV-205.md) |
| JEV-206 | 41 | complete | [逐类概率、双向对比奖励与 harness 阈值](../experiments/tpo-choice-coverage-206.md) | 五组评测已审核；排序较 201 改善，MINDS 阈值迁移未证实。[执行报告](reports/JEV-206.md) |
| JEV-207 | 42 | complete | [新构造的源域音频对与 actor 成对排序](../experiments/tpo-actor-pair-margin-207.md) | 五个面板已审核；源域选模未达标，保留 206。[执行报告](reports/JEV-207.md) |
| JEV-208 | 43 | complete | [Stage1 错例覆盖与当前概率间隔](../experiments/tpo-probability-gap-208.md) | 全部登记步骤已审核。[执行报告](reports/JEV-208.md)、[三点验证](reports/JEV-208-curve.md)、[尺寸与知识卡复核](reports/JEV-208-card-review.md) |
| JEV-209 | 44 | complete | [0.6B r128 知识卡与融合 TPO](../experiments/stage1-cards-tpo-209.md) | 模型实验已审核。MINDS 推理锁定 `.25` 为 `v0.0.rc25`：20 块/端点、两轮各端点 17,920 个概率零差异；Stage1/Stage2 评分吞吐为原生的 1.825 / 1.805 倍，以各轮原生归一化后较上一版提高 49.6% / 48.2%。181 八卡已释放。[执行报告](reports/JEV-209.md) · [推理报告](reports/JEV-209-inference-latency.md) |
| JEV-210 | 45 | planned | [0.6B r128 多负卡与类别 logits EMA](../experiments/stage1-cards-tpo-210.md) | 旧派发在 305 步停止、无 checkpoint，181 八卡已释放。v8 设计改为训练 logits 按真实类别累计 EMA 向量，以余弦相似度 softmax 驱动负类别采样；取消轮间验证/全量打分。末点全量 TRAIN 打分仍用于 Stage2 视图。v8 已接入统一入口，预取最多落后一批，CPU 双进程验证通过；尚未派发，实际八卡吞吐待有限验收。[执行报告](reports/JEV-210.md) |
| JEV-201 | 10 | complete | [联合 TPO 协议](../experiments/tpo-joint-comparison.md) | 全部登记步骤已审核。[执行报告](reports/JEV-201.md) |
| JEV-204 | 20 | in_progress | [全量单阶段 TPO 协议](../experiments/tpo-joint-comparison.md) | 训练和原生三点评测已审核；待补配对诊断、跨任务面板及全链路成本。[执行报告](reports/JEV-204.md) |
| JEV-R64-S2 | 25 | complete | [Stage2 selection 补评](../experiments/r64-stage2-selection.md) | 全部登记步骤已审核。[执行报告](reports/JEV-R64-S2.md) |

状态使用 `planned`、`in_progress`、`blocked`、`complete`、`failed` 或 `stopped`；
`planned` 专用于协议、依赖与预算已登记、可准备且尚未开工的行。领取时保存对应台账提交、行和冻结协议。

## 已登记的实验

| 实验 ID | 研究问题与预算 | 训练 / 评测状态 | 已知结果与决定 |
| --- | --- | --- | --- |
| JEV-R64-S1 | 0.6B r64，40,513 条 clean TRAIN、160 步；作为同轮 Stage2/TPO 的起点 | 正式端点及源域、迁移评测已审 | SLURP TEST / MINDS Noul macro-F1 为 80.97% / 71.64%；保留作为配对起点 |
| JEV-R64-S2 | 两阶段 r64 对照，分层难例加回放，正式 68 步 | 训练、SLURP TEST、MINDS 及 selection step 0/68 已审 | 80.43% / 66.05%；比 Stage1 分别低 0.53 / 5.59 个百分点（原始值计算），selection macro-F1 为 82.78%（−0.27 点）。[详细报告](reports/JEV-R64-S2.md) |
| JEV-194 | 34 步 TPO 投影与 34 步 direct；同起点、同数据及日程，只关闭投影项作为对照 | 两臂训练和完整留出评测已审；193 有限窗口失败，194 完成 | TPO 为 81.66% / 71.82%，direct 为 79.14% / 68.14%；投影在该种子有效，源域概率损失仍略升，采样贡献未隔离。[详细报告](reports/JEV-194.md) |
| JEV-195 | 改为独立 Bernoulli 的固定奖励投影；同起点、34 步 | 训练及 Noul 评测已审 | 81.39% / 70.42%；SLURP 的 NLL 比 194 低，但 MINDS macro-F1 低 1.40 点，不能视作全面提升 |
| JEV-196 | 交替训练 Noul、封闭 Choice 与 5 项开放 Choice；34 步 | 训练及留出评测已审 | Noul 81.04% / 72.08%；5 项开放面板的整体正确率恰等于 gold 缺席率，不能用零误接证明会选标签 |
| JEV-197 | 变长 Choice 统一支持 NONE，固定 reference；34 步 | 训练及完整原生面板评测已审 | Noul 81.32% / 71.94%；完整 Choice 的误拒为 99.87% / 100%，因此保留为 Noul 对照 |
| JEV-198 | 相对 197 逐步用上一次提交的 actor 刷新先验，保持冻结奖励位移和 34 步预算 | **34/34，训练和 SLURP TEST、MINDS 评测均已审** | Choice 误拒下降；Noul 和 Choice 候选内排序退化。保留为动态参考的反例和 201 的 Choice 对照。[详细报告](reports/JEV-198.md) |
| JEV-201 | 从 0.6B Stage1 step 160 接续、6,248 条 TRAIN × 2 轮、34 步；用配对 Noul/Choice、动态竞争者与固定长期锚共同训练 | **34/34，末点评测及变长 Choice 已审**。199/200 未进入正式训练 | SLURP 和 MINDS 的 Noul macro-F1 为 80.33% / 69.74%。变长 Choice 虽改善候选排序，但因误拒增加，整体准确率下降 13.12 / 17.15 点。SIB-FLEURS 为 73.46%，提升区间跨零；MMSU 准确率 47.96%。[报告](reports/JEV-201.md) |
| JEV-204 | 取消独立 Stage1；从 Qwen3-ASR-0.6B 加新初始化的任务参数起步，一套目标跑全量 clean TRAIN，40,513 条 × 2 轮，212 步 | 首次运行在第 4 步停止；正式重启 **212/212 步**和 SLURP / MINDS 的 step 0、106、212 已审 | 逐类概率末点 macro-F1 为 74.75% / 73.40%。相对 201，源域 −5.58 点、MINDS +3.66 点；第二轮 MINDS 排序退化，全候选 Choice 误拒增加。不替换现有对照。[报告](reports/JEV-204.md) |
| JEV-205 | 从 194 的 Stage1 起点接续，复用其训练人口、批次与固定参考，只修正固定排序目标 | **34/34 步**与五组评测均审核通过 | SLURP selection macro-F1 −0.59 个百分点，未达标；TEST +0.16、MINDS −0.15，未证实迁移收益。[报告](reports/JEV-205.md) |
| JEV-206 | 与 201 同起点、同训练录音、34 步；只输出逐类 sigmoid，用音频↔标签双向对比和 Brier 效用构造 TPO 目标 | **34/34 步**与五组评测已审；源 TRAIN 校准 21,244 条、59/60 类 | SLURP TEST / MINDS 为 81.58% / 71.89%，比 201 高 1.25 / 2.16 点；较共同 Stage1 的配对区间跨零。候选级 Brier 改善，MINDS 冻结阈值的覆盖与接受错误仍有取舍。[报告](reports/JEV-206.md) |
| JEV-207 | 从共同 Stage1 起点验证新训练集 + actor 排序项；按新视图定步数，206 为历史对照 | 2,750 条、62 对；16 步训练及五个面板已审核，八卡评测用时 22 分 54 秒 | 选模 macro-F1 80.10%（−3.08 点），未晋级。SLURP TEST 79.11%（−2.47 点），MINDS 73.46%（+1.56 点），SIB-FLEURS 持平，MMSU 少答对 11 题。[报告](reports/JEV-207.md) |
| JEV-208 | 从共同 Stage1 接续，覆盖全部准入 TRAIN 录音，以 TPO 和当前最强错类的概率间隔训练两轮 | 19,269 条，102 步与五面板已审 | 源域 selection +1.03 点但区间跨零；MINDS −0.17 点，话题 −4.57 点。保留 206。[错例分析](reports/JEV-208-errors.md) [报告](reports/JEV-208.md) |
| JEV-209 | 0.6B r128 知识卡 Stage1 全量两轮，单份分层视图，一轮双向 TPO 加按样本启用的动态间隔 | 混合版已冻结 109,610 条、68 类，真人端点五面板已归档 | [真人版归档](archive/JEV-209-real-only/report.md)；混合版单列完整官方 TEST 与五面板。[报告](reports/JEV-209.md) |
| JEV-210 | 0.6B r128，同锚点 K=16 张不同错卡 | 对照 209 混合版，八卡两轮 3,426 步预算 | [执行报告](reports/JEV-210.md) |

未逐号回填更早的工程包。旧实验以[归档](archive/)和既有研究文档为准；FSC 数字留在
历史收据里，当前主参考为 SLURP 与 MINDS。仅比较相同封存人口、候选和指标；
此表保留跨实验的读数，因果结论只用相应登记的配对对照。

## 198、199 和 200

198 vs 197 的数据人口、成对指标、成本和迁移限制见 [JEV-198 报告](reports/JEV-198.md)。
SLURP 和 MINDS 的 Noul macro-F1 均下降；完整 Choice 的误拒下降，但候选内准确率也下降。
199 和 200 是 201 的工程版本，均未进入正式训练，没有评测分数。

| 运行家族与包 | 最后到达阶段 | 证据与处理 |
| --- | --- | --- |
| `tpo-stage2-contrast-v1/v0.0.193` | 有限窗口失败 | `ops-193/finite-wait.json`、`terminal.json`；同一研究问题在 194 重封后完成 |
| `tpo-joint-top1/v0.0.199` | 固定参考失败，无正式训练、无对应评测 | `ops-199/anchor.log` 记录 `KeyError: 'contract'`；终态非零 |
| `tpo-joint-top1/v0.0.200` | 固定参考失败，无正式训练、无对应评测 | `ops-200/anchor.log` 记录字符串参数没有 `read_bytes`；终态非零 |
| `tpo-joint-top1/v0.0.201` | 固定参考、两步有限验收与正式训练均已审核通过 | `ops-201/{anchor-collected,finite-collected,tpo-collected,terminal}.json`；按实际容器与更新收据核对 |
| `tpo-single-stage-full/v0.0.202` / `203` / `204` | 202 已有旧包；203 的看护加载配置失败；204 参考已审，正式训练保留四步收据后停止 | `package-202/`、`custody-203/driver.log`；`ops-204/{finite-collected,anchor-collected,tpo-inspect,tpo-wait}.json` |

## 后续实验怎样调整

1. **短对照先回答一个问题。** 194 的同起点、同数据、等预算开关对照可归因；195–197
   同时变更多个目标或输入，只能评价组合。201 也是组合验证。先按原协议评第 34 步：
   Noul 以 197 为主对照，Choice 以 198 为主对照，同时列另一端点。若 201 未守住，
   用独立 clean TRAIN 诊断区分新负例与概率漂移，再分别改动态竞争项或长期锚。
   TEST 和迁移数据不用于改权重、采样、阈值或选 checkpoint。
2. **把“愿意选”和“选得对”各自验收。** 保留 SLURP、MINDS 的同音频对照，分别报
   两种接口的 Top-1；加入 gold 缺席的变长候选面板，分别报误拒、误接和候选内正确率。
   NLL/Brier、正负项及最强错类间隔用于诊断。只有独立话题任务 SIB-FLEURS 的预登记
   结果支持时才报告跨任务改善；MINDS 只说明跨数据集迁移。
3. **扩人口时，把成本和效果一起对齐。** 204 的 212 步从未训练任务的起点跑全量；
   与 201 的 34 步对比的是两条路线。比较时把旧路线 Stage1、全量 TRAIN 打分、
   后段训练和对应评测的卡时都计入，分开列参考生成、正式训练、评测与审核。
   用 step 0、轮末、终点和实际曝光报告曲线。预登记协议明确等预算后，
   才能用单因素对照回答“去掉阶段边界”的独立作用。单种子结果有价值，复现后再扩大。
4. **登记按事件更新。** 用[模板](TEMPLATE.md)预登记；有限验收失败、正式端点收集、
   评测独立审核或停止时更新台账，并随仓库同步。工程修复另封包、沿用实验 ID；
   版本、分母、时点、成本及结果只从已核验证据读，避免人工重复统计。

## 完整报告

[报告目录](reports/README.md)保留每次实验的协议、复用记录、训练与评测证据、指标与成本，
尤其是每个跨数据集目标的结果和结论。正式结果、部分结果和缺失证据分别标注；
实验上线或晋级结论与包执行状态分开记录。

## 证据定位

以下路径相对于调度机的 `next-simple-r64-dp8-v1` 运行证据根，由本地宏解析。
引用是供复核的索引；完整预测、日志、权重和本地宏留在 Git 外。

| 核查项 | 收据 | SHA-256 |
| --- | --- | --- |
| 194 TPO / direct / Stage1 的留出比较 | `tpo-stage2-contrast-v1/comparison-194.json` | `b795e2f344fd0f96f0c922b3aaca718a898c674c96b38e40e940cb1cf4a7e896` |
| 195 独立评分 | `tpo-stage2-reward-projection-v1/eval-195-ops/score-collected.json` | `fc7290f4194f77ad7f108c6d460d734317a86137ca6d57fa6616c5e40a4f57df` |
| 196 独立评分 | `tpo-choices-v1/eval-ops/score-collected.json` | `dabb52375bff1c6635a8b94f37ab716cf865109386b0bbd4e0a9ec950a9702aa` |
| 196/197 完整面板的独立评分 | `tpo-variable-choice-v2/eval-ops-full-native/score-collected.json` | `1a5703bd9b9d6fc55c3db41067565b1b7d8a3a0938ab02466e6b07da06f2cabc` |
| 198 正式训练收据 | `tpo-variable-choice-v2/ops-198/tpo-collected.json` | `c104140c5ccece0704b57959f28686549ff80747106a275719110070bb777cc0` |
| 197/198 完整面板的独立评分 | `tpo-variable-choice-v2/eval-ops-full-native-198/score-collected.json` | `52ee4f19d4d2c47c715661bd7ad183f95ebdfed7799e4a7362802408bd70b618` |
| 201 冻结执行计划 | `tpo-joint-top1/package-201/entry/joint.json` | `03705498230f7ac1b3a237b9f8c6fe4af695d98923ce27cc378537243e33b457` |
| 204 CPU 封包训练目标清单 | `tpo-single-stage-full/package-204/entry/targets/MANIFEST.json` | `9090d4c9db13c297a4228fc8f482a16afca32d380c9a9805a070e976dff3e95a` |
| 201 第 34 步独立训练审核 | `tpo-joint-top1/ops-201/tpo-collected.json` | `1ed0ad04dbc6c8a74f479443213d044207bdc0c36785fea7e4544ad8fd80ef48` |
| 204 两步固定参考审核 | `tpo-single-stage-full/ops-204/finite-anchor-collected.json` | `dfbea688139215591ad81972d349f3fa8583a7882013b7a5ef5eb92119efc659` |
| 204 两步有限训练审核 | `tpo-single-stage-full/ops-204/finite-collected.json` | `b43c728eb13f3aa74f4b67a50bacb472ce6945d919d809f3e3eabb8b22bcf5b4` |
