# 第256步阶段分析与后续准入

用户最新要求：简单任务出现明确训练收益后才启动复杂训练；普通观察每15分钟。
配置记录在configs/experiments/context-monitoring-and-complex-admission-v1.json。
尚无完整RL选择集收益，复杂训练保持禁止。当前原队列只包含简单算法、固定评测和迁移，
属性processor及其finalizer仅准备数据，不触发复杂训练。

## 第256步实际证据

使用此前已独立收集的四rank update-0256.json，重验其SHA并确认四rank全局rollout一致；
重新绑定全量TRAIN、日程、audio/tensor、候选及gold，128组实际动作hash重建匹配。
分析的是第256次更新前的behavior255及同批Stage1 reference，不是更新后的checkpoint256效果。
新增派生JSON只使用本地归档，不增加远程观察或训练。

| 同批128条TRAIN | Stage1 reference | 当前behavior |
| --- | ---: | ---: |
| 阈值集合F1 | 73.9583% | 77.3438% |
| top1准确率 | 83.5938% | 78.9063% |
| 精确集合 | 94/128 | 95/128 |
| binary NLL | 0.0161812 | 0.0535061 |
| Brier | 0.0042154 | 0.0067591 |
| 采样分布期望集合F1 | 0.660020 | 0.772404 |

11条从精确错误变正确，10条从正确变错误；仅净增1条。set-F1包含部分正确决策，
其增益不能自动解释为原生单意图识别提高。NLL约增至3.31倍，Brier约1.60倍。
不能据此认定泛化退化或提升：这只是同批TRAIN配对分析，且不同训练步使用不同query，
不能将第74步和第256步直接拼成同一评测曲线。

实际G16中92组全1、25组全0、5组恒部分reward、6组变化；有效差异组仅4.6875%。
33条阈值错误中30条恒奖励。原生组优势在恒奖励组为零，辅助监督和KL仍可能更新。
第74步固定128条、每条512次重新采样的补充诊断也显示RLOO零方向频率87.67%，
LEG式条件积分为0%；输出logit方向平均MSE分别2.48266e-4和3.23754e-7。
这比较的是原始reward梯度，不能等同GRPO目标、模型梯度或相同reward计算预算；
该候选不是新估计器创新，也未证明训练收益。

## 四项改进判断

1. **优化函数：**概率校准值得优先关注。排序、集合reward和NLL的目标并不相同，
   当前辅助系数衰减可能影响相对强度，但本次不能单独归因。后续简单候选可比较更慢衰减
   或保持概率辅助权重；须先定义独立实验并报告Final与NLL/Brier，不能中途改系数。
2. **reward：**集合F1奖励有部分正确信用，但当前单标签top1和集合F1走势不一致。
   先坚持harness集合语义及冻结评分，再判断该效用是否符合业务；不能为提高top1而静默
   改成强制单选。概率校准由proper scoring辅助和独立指标约束，F1本身不保证校准。
3. **算法：**全等奖励使GRPO/RLOO均无采样RL信号。条件积分是有依据的后续简单任务
   候选；不能仅增加G就承诺消除饱和，也不能把PPO-critic当作自动解决方案。
4. **工程：**该步104.49秒中preparation75.87秒、actor24.76秒；rollout66.85秒已包含在
   preparation内，不能重复相加。reference计算33.16秒且128条reference miss；输入缓存
   128/128命中，两类cache不可混淆。首轮遇到新query的reference miss不等于缓存bug。
   优先验证已有批量rollout候选及reference复用，再评估microbatch；micro32不是已证明加速。

actor日志中的actor/kl_loss=0是原生外层KL关闭的字段；自定义混合loss仍包含reference KL。
当前registered_loss丢弃了内部组件metrics，只保留总context_loss。因此不能把日志0解释成
真实KL为0，也不能从当前记录精确分解每项loss或梯度。未来版本应增加分项观测，
在独立候选验证后发布；不改运行依赖。grad_norm=16.80也不能未经语义核对称为裁剪后值。

## 观察与准入

最新普通GRPO观察live-v44为四rank272/634、running/nonOOM；下一观察按新规则不早于
1794722060.694339。TRAIN输入下次不早于1794721860.9853616。旧600秒字段保留为历史收据。
原driver在训练终态后依次评估固定11点，每点完整9226 selection；当前没有在线选择集曲线。
在不占用未释放设备、不破坏原租约和来源审核前提下继续阶段性已提交产物分析。

复杂训练准入需完整selection的Final增益、完整曲线和退化/不确定性审查，单批TRAIN增益
不予放行。当前不虚构最小收益阈值或容忍校准退化值，也不使用TEST来决定训练方案。
若证据不足，继续简单任务版本化优化；不因数据准备完成自动开始复杂任务。

证据：context-committed-step256-reward-support-v1.json、context-committed-step256-reward-groups-v1.json、
context-credit-resampling-step74-v1.json；计时来源为工程step256独立收集。
