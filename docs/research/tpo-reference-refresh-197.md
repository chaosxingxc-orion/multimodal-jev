# TPO 的 reference 什么时候更新？197 配对对照与下一阶段路线

**状态：实验预登记，只做设计，没有改变 197 的训练或派发新任务。** 先用当前有可靠起点的 197 测出刷新锚点的贡献，再独立验证从基座直接 TPO。两者有效性不能互相替代。

**对照。** 已核对目标视图及 batches 实体：0.6B、BF16、q/v LoRA r64/alpha128、DP6，6,248 条 TRAIN 录音，两个 epoch，各 17 步，各曝光全部 6,248 条；后半轮 11 个 Choice batch、6 个 Noul batch。复用配置所钉的真实列表和顺序，包括 NONE、配对、尾批与 padding。A 全程固定原始 reference；B 从同一起点运行相同 34 步。第 t 步只用完成第 t−1 次 optimizer.step 后的 actor 来评分第 t 个 macro batch。首步均为初始化策略；第 2–34 步可以出现差异。预测：锚点转移允许第二轮继续拉开真类与混淆类差距，也可能放大第一轮的错误确信；都要用同一端点评测。

**究竟改变哪一项。** 初版只研究 TPO 投影的策略先验。Noul 用 logit(q)=logit(p_ref)+A_197/τ；需要保留同一条录音、同一 epoch 的 A_197。优先用冻结源分数和既定配对在 CPU 编译原始 A；重构固定臂的 target 并与 197 数值对齐。不得把原始 targets.pt 当成 A。Choice 在原列表上用 softmax(z_ref/T+R/β)，覆盖 marker 用 logit(g_ref)+ΔR/β。缺席样本的条件排序仍不产生梯度。B 只换三个公式中的 ref，同样 beta/T、同样标注和奖励。197 的 Noul 部分奖励是 reference 处的局部导数，因此“重算全部奖励”会改变两个变量。若 A/B 有效，后续才另设 C（同一次刷新，同时重算与 reference 有关的奖励），以 C−B 区分额外贡献。候选生成器永不随本次刷新改变。

**一步一刷新，一个 macro 一份缓存。** 197 的完整 macro 有 384 个槽位，DP6 每卡 64、每个 micro 8，每卡累积 8 次后才做一次 optimizer.step。第 17、34 步均只有 104 个真实样本，280 个 padding 的权重为零。动态臂在更新提交、各 rank 权重版本一致后，无梯度地用同版本计算下一整个 macro 的 reference；可物理分块，无需一次塞进单卡。全部有效请求及其原始 logits、条件候选支持、覆盖 marker 和目标身份齐全才允许训练消费。一个 macro 内各 micro 只读，targets detach；若直接对当前参数和 reference 两边同时反传，梯度不再对应预注册目标。训练前可预取下一批音频、模板和候选；上一更新提交前算出的 logits 不得冒用为下一版本。完成第 34 步后不再生成训练缓存。无需另存 34 份模型；在同一 actor 未更新的时间窗内 eval/no_grad 前向即可。缓存和训练图分开，只暂留当前 macro，崩溃恢复用准确策略、步号、输入身份重建；跨步误用、缺失、非有限值或训练时概率不一致要报错。

**成本与验收。** 第一次评分相同；34 份逻辑批缓存，首份可按完整身份复用，后续 33 次更新后换版本。无需每步重扫 6,248 条。固定臂若已有身份相同的 197 封存结果可复用，否则同预算重跑；新动态臂 34 次更新。单独记录 reference 前向 wall time、NPU 卡分钟、等待、训练时长与总耗时；匹配的是训练更新数，而不是计算成本。首步在相同执行路径测逐请求 prior、target、loss 和梯度对齐。之后先验只从上一步提交的权重产生，验收版本没有滞后；检查非零奖励的更新、累计相对初始策略的漂移以及负样本概率。冻结奖励时，逐步移动先验使每一步 KL 约束相对于近期策略，不能视为仍约束到初始策略。动态臂即使 loss 更小也不证明末点评测更好。

**怎样判断。** 首要检验是 SIB-FLEURS 英语 7 类话题 Choice macro-F1 的 B−A；按 177 个原句成组、组内所有录音共同重采样，报告配对 95% 区间。预设上线候选条件：主指标差值区间下界大于 0、SLURP Choice macro-F1 点估计退化不超过 0.5 个百分点、SIB 多类 Brier 点估计恶化不超过 0.005。两个护栏是本轮拟定的工程容忍度，不是学界阈值；同时给出差值区间。MMSU 98 道词义题按题配对报准确率，MINDS 报意图迁移；各集报告 Noul/Choice 概率指标和误拒，不以 loss 降低代替成效。混合或区间跨零就报告不确定，数据少不能解释为无效。端点选择、刷新时点、阈值都不随测试结果调整；独立种子复核训练波动后才称稳定。

**下一阶段。** 0.6B 的直接 TPO 从未做过任务训练的基座、固定种子新建的 Jev head、初始 LoRA 出发，step 0 用完全相同的只读策略作初始锚点。使用完整 clean SLURP TRAIN 与脚本构造的有/无覆盖、乱序变长候选；不能复用 Stage1 的难例人口、提案分数或参考缓存。无可信召回器时先用不依赖 gold 的可复现列表，先出命中与缺席分层统计，防止未训练 head 的分数支配训练人口。冷启动时先测目标熵、分类别概率和真实梯度，特别是大 K 时弱奖励能否提供足够区分。若 197 刷新呈阳性，在单阶段独立复验，不直接迁移结论。先做 0.6B 同协议的重复种子及全量迁移，再允许 1.7B：第一组固定 r64/alpha128，保持数据、曝光、head 规则和评价相同，按模型几何重建参数及缓存；确有容量或成本需要，再只调整 rank（例如 r32/alpha64 或 r128/alpha256），分别报告可训练参数、吞吐和效果。明确全量冷启动和 197 的优化预算不同，不能把效果差全归给少一个阶段。

**执行登记：v0.0.198。** 上面的预登记保留原样。动态臂另用 [可执行配置](../../configs/portable/operations/tpo-next-macro-window.json)，CPU 封包复算 12,496 次曝光 × 60 个候选的原始 target，749,760 个 float32 项全部与 197 相等。运行时取训练 actor 的 BF16 音频前向；按步记录 reference 所用的策略版本、全局槽数、最慢 rank 的缓存时间及真实 loss。第一个 macro 在每卡逐项比较固定臂的 Choice logits，首步 loss 和梯度范数与 197 正式 step 1 比较。实际 NPU 结果以收据为准。复现的三个入口如下（本机宏和发布包路径由协作团队绑定）：

```bash
python -m tools.reports.prepare_tpo_windows configs/portable/operations/tpo-next-macro-window.json "$LOCAL_MACROS"
python -m tools.reports.native_seal_operations --config configs/portable/operations/native-seal-operations.json --local "$PACKAGE/operations-local.json" --detach
python -m tools.reports.collect_tpo_first_updates "$PACKAGE"
```

采集器等待正式容器和前三次原子写入，核对 image、容器 ID、有限验收、正式配置及 reference 步号。输出为运行目录的 `first-three.json`；CPU 收据、排队句柄、有限窗口以及 197 的 loss 均不可替代它。
