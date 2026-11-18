# 第384步：简单任务阶段诊断

定时观察1794730236.5961814确认原GRPO四rank384/634、running/nonOOM，checkpoint384已提交。
只读收集该检查点；四rank master/Adam、执行与发布策略、adapter、冻结marker、激活和记录的
parity审核通过。128参数、6,878,209元素的FP32 master重组转BF16与执行张量精确一致。
本地首次审核遗漏既有safetensors依赖路径，补齐PYTHONPATH后复用原运输文件完成；
失败记录保留。没有重传、模型前向或新增NPU恢复。

## 同批TRAIN128，固定Stage1参考与behavior383

| 指标 | 固定参考 | 当前行为策略 |
| --- | ---: | ---: |
| 集合F1 | 84.3750% | 86.4583% |
| top1 | 90.6250% | 85.9375% |
| 精确集合 | 108/128 | 106/128 |
| binary NLL | 0.0111797 | 0.0399406 |
| Brier | 0.00290678 | 0.00479274 |

精确集合4条修复、6条退化。集合F1允许部分正确的多选获得分数，因此F1与精确集合、top1
方向不同不构成评分矛盾；保持冻结harness契约，不能事后换主指标。NLL/Brier也变差。
这是同批训练诊断，不是heldout收益；不同步的128条query不同，不能拼接成固定样本趋势。

G16的128条动作hash均重建一致：103组全1、14组全0、5组恒部分奖励、6组有差异，
有效差异组占4.6875%。22条阈值错误中19条恒reward。组标准化或leave-one-out都不能
直接给恒reward组提供RL优势；辅助监督及KL仍可更新共享参数。

rank0总69.43秒、准备42.61秒、actor22.84秒，input cache128命中。耗时含checkpoint步骤，
不是无checkpoint吞吐基准；嵌套时间不可直接相加。原生pg_loss是混合目标，不能解释为纯RL loss。

完整634步和11点selection继续。若完整选择集重复概率质量退化，下一版优先比较概率监督
衰减或KL单变量；若奖励信号问题妨碍收益，再以独立目标评估既有条件/精确积分候选。
当前证据不支持复杂任务准入，不改运行配置或既有后继队列。

证据：
- [检查点审核](../engineering/context-arithmetic-step384-independent-v1.json)
- [master与BF16执行逐项比较](../engineering/context-arithmetic-step384-master-export-v1.json)
- [同批输出与理论奖励支持](context-committed-step384-reward-support-v1.json)
- [实际G16奖励组](context-committed-step384-reward-groups-v1.json)

## 概率错误分解

重新绑定上述来源hash及四rank原始更新文件后，按同一128条query分解binary NLL。
每query的正例贡献为-log(p_gold)/60，负例贡献为-sum(log(1-p_negative))/60，
两项相加复现原报告；没有裁剪概率或修改阈值。

| 项目 | 固定参考 | behavior383 |
| --- | ---: | ---: |
| 正例NLL贡献（全batch均值） | 0.00705066 | 0.01958957 |
| 负例NLL贡献（全batch均值） | 0.00412907 | 0.02035098 |
| gold概率低于0.01的query | 1 | 14 |
| 存在错误候选概率高于0.99的query | 0 | 17 |
| 空集 / 错误非空集合 | 14 / 6 | 2 / 13 |
| 包含gold但多选 | 0 | 7 |

106条query的NLL下降，22条上升。当前22条非精确集合贡献约99.9226%的总NLL，
其中13条错误非空集合贡献约79.5001%。这表明该批的主要问题是错误预测也趋于极端自信；
不能仅看多数query改善或平均reward而忽视错误尾部。高低概率阈值仅用于诊断，没有调参选模。

当前set-F1=106/128精确正确贡献 + 7条多选的部分得分贡献0.0364583=0.8645833。
相比参考108/128精确正确，部分得分解释了F1上升而精确集合下降。原14条空集中4条转精确、
7条转错误非空、1条转含gold多选、2条仍空；原108条精确中5条转多选、1条转错误非空。
这属于同批配对结果，不是任意调阈值的依据，也不证明heldout校准已经全面退化。

完整selection若复现这一现象，应优先检查概率监督衰减与KL强度的单变量候选。
增加G或替换优势估计器不能自动修复概率质量；精确F1期望同样不保证校准。

详见[分解证据](context-step384-probability-error-decomposition-v1.json)。

## 保留NLL作为错误饱和的纠正候选

正式封包与当前辅助源码逐字相同，使用binary_cross_entropy_with_logits，按60候选取均值。
理想实数链路中，NLL对logit的导数是(p-y)/60；Brier是2(p-y)p(1-p)/60。
因此错误饱和时NLL没有额外p(1-p)因子，Brier则会缩小纠正信号。

该批20条query共32个高置信错误坐标（gold<.01或negative>.99），等系数下Brier/NLL
输出logit梯度绝对值比为6.51e-7至0.0117522。这里只用已记录概率解析计算，
不是实际BF16计算图重放、参数梯度或混合loss强度比较，也不能确定最佳辅助权重。
Brier保留为概率质量评测指标；优先检查NLL权重/衰减，不能据“Brier变差”直接换成Brier loss。

证据：[输出梯度诊断](context-step384-proper-score-gradient-diagnostic-v1.json)。
