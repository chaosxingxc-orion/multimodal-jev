# 第320步：提交状态与简单任务阶段诊断

原GRPO定时器local1794726636.5727654确认四rank331/634，running/nonOOM，step320已提交。
下一普通观察不早于1794727536.5727654。只读运输该已提交checkpoint，没有新增latest查询。

四rank commit/FP32 master及Adam、BF16 execution及发布策略、原生adapter、冻结marker、
策略激活和已记录行为parity检查通过。128参数/6,878,209元素逐项重组master转BF16，
与execution精确相同。此次是CPU归档审核，不是新的NPU恢复或完整训练验收。

## 同批TRAIN128

以下是第320次更新前behavior319相对同批固定Stage1 reference，不能作选择集收益。

| 指标 | 固定参考 | 当前行为策略 |
| --- | ---: | ---: |
| 集合F1 | 77.8646% | 84.1146% |
| top1 | 88.2813% | 84.3750% |
| 精确集合 | 99/128 | 105/128 |
| binary NLL | 0.0152193 | 0.0459341 |
| Brier | 0.00361907 | 0.00525215 |

8条精确错误被修复，2条正确退化。集合F1改善而top1与概率质量变差，再次说明这些指标
不可互换；不要为了增加某一指标静默改harness契约。不同步骤query不同，不能把step256、
318、320拼成固定样本趋势，亦不能把混合目标变化单归因于RL。

G16实际动作hash128条重建一致：103组全1、19组全0、2组恒部分奖励、4组有差异。
23条阈值错误中21条恒reward。RL信号稀疏问题仍存在，辅助监督/KL不因其恒reward而消失。

rank0总67.51秒，准备40.07秒、actor23.55秒；input cache128命中，reference miss0。
这些细分时间互有包含，不能重复相加。已观察的第二轮步骤保持约67秒，但还不是独立
稳定吞吐基准或batch候选加速证据。

## 下一步

原完整634步和11×9226选择集曲线继续；无完整泛化收益前不启动复杂训练。
保留概率NLL权重候选和条件积分候选，待完整曲线审查后按单变量新实验决定。
新分项日志已通过真实verl固定镜像CPU16项检查，未安装到本次训练；下一版本验收时纳入，
帮助区分RL/排序/NLL/KL强度，不能从当前缺失字段反推为原生记录。

证据：engineering/context-arithmetic-step320-{independent,master-export,runtime-summary}-v1.json；
research/context-committed-step320-reward-{support,groups}-v1.json。
