# Stage1 概率间隔目标与 Stage2 难例训练视图

> 数据生产新入口已收敛到[Stage2 dataset release v2](stage2-dataset-release-v2.md)，包含原子发布、独立复算、宏配置与实际batch消费。下方v1接口是初版开发记录。

本轮用户要求：废弃纯 InfoNCE Stage1，使用新的区分性目标从原始初始化重新开始；
Stage2 根据这个新终点在完整 clean TRAIN 上的预测构造难例加简单回放训练视图。
旧运行不作为新运行的初始化、reference 或正式曲线。当前实现与 NPU 状态分开记录。

## 为什么调整 Stage1

模型输出每个候选的独立 sigmoid 概率。纯 InfoNCE 对所有 logit 加同一个常数不敏感，
但 sigmoid 概率会改变；因此正确排序并不保证概率间隔，也不保证正确阈值判决。
概率间隔本身也不足：0.95/0.80 的间隔够大，但两个候选都会被阈值0.5选中。
本轮仍是完整候选面板的区分性学习，不增加二元 CE。

令 z+ 为真实意图 logit，z- 为最强错误候选 logit，p±=sigmoid(z±)：

```
L_NCE = logsumexp(z / T) - z+ / T
L_gap = relu(M - (p+ - p-))
a = logit(t + M/2); b = logit(t - M/2)
L_anchor = [relu(a - z+) + relu(z- - b)] / 2
L = L_NCE + lambda_gap * L_gap + lambda_anchor * L_anchor
```

初值 T=1、M=0.1、t=0.5、lambda_gap=1、lambda_anchor=0.25。
anchor 要求正例≥0.55且所有负例≤0.45；在正负概率饱和且预测错误时，logit hinge仍有
纠正梯度。InfoNCE提供全候选竞争，概率 hinge 直接约束实际读出，anchor稳定阈值两侧。
这些是待实验验证的初值，不保证每条样本都达标，也不等同于概率校准。极难或有噪声的
样本不能通过无穷扩大M/训练轮数强行拟合。多正例任务不在本轮支持范围。

BF16执行、FP32 loss/主参数/Adam不变。所有项按真实音频数归一化；DP乘world size后
由分布式梯度平均消去，零权重padding不产生梯度。已满足间隔的样本margin项归零，
InfoNCE仍继续学习。候选完整面板保留，绝不依据gold改变问题或输入。

## 验收与日志

源域完整评测应报告：top1准确率、阈值exact-set、margin达标率、双侧anchor达标率、
margin分位数、逐类覆盖、NLL/Brier以及未达标样本数。TRAIN负责采样，selection只负责
候选选择，test/FSC/MINDS不参与阈值和权重调整。

训练日志提供 infonce_numerator、probability_margin_numerator、threshold_anchor_numerator、
probability_gap_sum、margin_satisfied_count、anchored_count、local_real_queries。
这些是微批内累积量，verl可能再取平均；不能直接把count当作完整训练集样本数。
完整概率质量与难例比例应从终点全TRAIN缓存独立统计，不能用前三步证明收益。

Stage1验收分为工程通过和效果通过；工程通过不承诺margin覆盖率达标。当前未设任意
总体达标率作为已验证门槛，应先报告完整分布、冻结对照和逐类退化，再决定是否继续。

## Stage2 采样：有条件认可难例优先

保留所有错误和边界样本，加入目标20%简单回放；每个有简单样本的类别至少保留16条，
固定seed317，无放回。类保底优先，实际比例可能超过20%；没有足够简单样本时不复制。

- error：gold不是严格top1（包含并列），或阈值输出不等于唯一gold。
- boundary：输出正确，但gold相对最强错误的概率间隔不足M，或双侧anchor未满足。
- easy：严格top1、阈值判决、概率间隔、双侧anchor均满足。

这个比例不是最优结论。只留错误会放大残留标注噪声和类别偏差、造成简单能力遗忘，
并改变概率校准的最优分布；保留回放和逐类报告用于发现这些风险。GRPO一组全错且奖励
完全相同仍可能没有相对优势，难例采样不能替代连续概率/排序优化信号。
若大部分样本仍为难例，视图接近全量是合法结果，不应为了缩小数据集丢弃难例。

终点全TRAIN评分只执行一次，复用同一reference。视图只保存ID、诊断、日程及hash，
不复制音频、特征或reference。按新样本数推导轮次预算：2*ceil(N_selected/128)，
不把小集合机械重复到旧634步。同一个Stage2运行内视图冻结；若重采样需新版本。
这有意改变优化分布，不宣称是全TRAIN的无偏估计。迁移收益仍须独立FSC/MINDS评测。

## 接口与发布边界

- Stage1 loss：`src/multimodal_jev/context_infonce.py`，显式kind
  `candidate_infonce_probability_margin_v1`；旧kind仍保持旧数学行为。
- 候选设计：`configs/experiments/context-two-stage-margin-v0.0.123.json`。
- 固定镜像验证：`configs/portable/operations/context-stage1-margin-cpu-tests.json`。
- 启动复用 `tools.reports.context_stage1_start_pipeline`，路径/设备绑定在本地宏。
- Stage2视图：`tools.reports.prepare_stage2_training_view`，操作`prepare`/`audit`，
  模板`configs/portable/operations/stage2-training-view.json`。AUDITED_STAGE1_REFERENCE_JOB绑定
  新终点的完整reference生产配置；审计重新计算抽样，拒绝缺失、混split、候选或policy错配。
- `tools.reports.context_training_view_trainer`是显式选择的共享trainer候选适配层。
  全reference/音频镜像仍完整核验，只有调度population缩小。

Stage2数据视图及trainer适配已具备CPU验证接口，正式封包和独立输出审核仍需接入动态
population/step预算并在NPU验收；现有40513/634固定发布链不得直接放行子集。
在新Stage1终点与reference产生前，不发布真实难例集合，不填猜测的样本数或提速倍数。

## 本轮工程证据与运行身份（2026-11-19）

旧运行`context-infonce-prefix-resume-v0.0.122`已主动停止，非OOM；取消收据在
`/srv/multimodal-jev-xingchao/operations/context-infonce-prefix-resume-v0.0.122/cancelled-by-user.json`。
仅保留待清理结论与证据，不作为新Stage1/Stage2初始化。

新候选`context-margin-finite-v0.0.123`已通过固定镜像87项CPU测试和DP4两步真实NPU审核。
两步全局总loss=4.31127573、4.31448619。恢复候选四rank主参数、Adam、batch、BF16导出
完全一致。上述数字是工程验收，不是完整训练效果。

尾批准入首次拦截的根因是工作区Stage2采样文件在CPU快照和封包之间发生修改；
loss文件身份无差异。独立CPU16项对精确封包验证通过，保留原包，另行绑定补验收据。
后续封包入口已改为直接取经过hash验证的CPU源码快照，防止复发。
这次继续入口是`tools.reports.resume_stage1_admission`，已通过的prefix/replay不重复执行。

正式候选名称`context-margin-formal-v0.0.rc7`，从原始policy初始化，global256/micro8、
warmup16、LR1e-4、40513条两轮318步；不能使用旧loss的optimizer续训。
启动宏：`.local/config/context-margin-admission-v0.0.124.json`。
进度/审核：`/srv/multimodal-jev-xingchao/operations/context-margin-admission-v0.0.124`。
正式训练是否已启动以及完成多少步必须查询当前容器与事件证据。

各项loss按四rank真实样本汇总的固定入口为`tools.reports.summarize_stage1_margin`，
模板`configs/portable/operations/stage1-margin-summary.json`。它还核对分项重建loss与verl
actor原生loss一致；缺rank或错误分母直接失败。

2026-11-19 08:03:34 UTC 正式容器已启动，现场inspect为running/nonOOM；
容器ID `b3a900d5dfadb7d374b7312b556cbd1e9392289134315988c90cb224d0b1a00a`。
真实尾批审核通过，两个验收批次分别为65和256条真实音频。
正式宏为`/srv/multimodal-jev-xingchao/operations/context-margin-admission-v0.0.124/formal-operations.json`，
配合`configs/portable/operations/context-infonce-formal.json`；已启动的任务不得重复launch。
该状态是启动时快照，完成量仍以现场证据为准。

2026-11-19 08:07:16 UTC 首三步事件验收：四rank均完成1–3步，容器running/nonOOM。
全局loss依次4.31127573、4.31448619、4.30866394；慢rank步时40.32、35.42、35.14秒。
当前初始化附近的margin/anchor覆盖仍为0，不据此前三步宣称收益。首三步汇总与原始记录：
`/srv/multimodal-jev-xingchao/operations/context-margin-admission-v0.0.124-first-updates/summary.json`。
按约35秒/步估算318步训练约3.1小时，checkpoint及终态收集/评测另计，非保证终止时间。
