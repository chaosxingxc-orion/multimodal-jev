# 上下文候选概率：对比辅助目标与RL组合

状态：可执行候选与数值验证，未锁定正式训练超参/奖励，未更改正在运行的Stage1。
数据语义见`context-candidate-task-integration-v1.md`。输出仍是逐候选sigmoid prob。

## 可执行的辅助目标

对每个原query，P为有标注正候选，N为有标注负候选，O为有标注候选：

```
Lrank = mean_{p in P,n in N} softplus((z_n-z_p+margin)/temperature)
Lprob = mean_{i in O} binary_NLL(z_i,y_i)  或  Brier(sigmoid(z_i),y_i)
Laux  = ranking_coefficient * Lrank + probability_coefficient * Lprob
```

只比较有效/无效候选，不对两个合法候选编造偏好。全正或全负时Lrank=0，保留
Lprob；全缺标时两项为0，但原query仍留在全局分母。缺标不补负例，不靠缩小分母
重新分配其预算。外层为`sum_query weight*Laux/global_real_queries`，跨DP平均梯度
时本地乘world_size，microbatch累计求和，不额外除累积次数；padding必须零权重。
候选数不同先按各query内均值，再按query总体归一化，不让大菜单自动获得更多权重。

该对比项只约束差值，对所有logit共同平移不敏感；它不能独自确定prob的绝对标尺。
概率监督是独立的辅助项，可按预先登记的更新日程衰减，不能因为使用pairwise就把
sigmoid值称为已校准概率。NLL/Brier选型与系数仍需在学习/校准协议内预先固定。

一正59负时，pairwise对该正例与59个负例逐对比较，避免单纯“多数负例正确”的
训练信号占据全部目标；它仍不能保证模型或优化稳定性。重加权/采样会影响概率
含义，报告自然分布Brier/NLL及原query效果，不能在测试上调系数/阈值。

## 与通用RL基线的组合

主候选仍使用固定verl的GRPO或RLOO优势与joint PPO clipping，采样动作是所有
真实问题的二元判决tuple，joint logprob为各位logprob之和。对外语义概率与这个
因子化探索策略分开，不声称语义事件独立。现有`span_sampled_hybrid`可以设置其
旧CE系数为0，再加上述Laux，避免重复计入监督损失；也不把对比项冒称为RL。
参考KL若使用仍绑定固定Stage1策略；概率输出结构变化后旧Final60参考不能直接复用。

`probe_context_hybrid_loss_cpu.py`在固定镜像调用真实verl实现，使用构造logits和
仅作算术fixture的奖励，比较独立clipped公式、完整组优势、全局query归一化以及
模拟rank平均/micro拆分梯度。包含零信号组、缺标、全正和padding。不是多进程
分布式验收，也没有冻结候选集合奖励。仍须真实Qwen/NPU、snapshot/恢复、vLLM。

## 必须另行确定的奖励

逐位correctness在一正59负时全负可获59/60，不能直接作主集合目标。整集合exact
对60个随机二元动作几乎全零，也不能未经诊断直接派发。候选集合F1/代价敏感效用
可避免部分退化，但需要明确空集、部分标注、hard availability与不同问题预算的
语义。先用已有全量控制报告检查返回全/空/单项oracle，再固定真实任务奖励。
本文件没有将构造奖励升级为正式RL奖励，也不添加未经监督的过程reward。

固定镜像CPU aaa9d42f已exit0/nonOOM。GRPO与RLOO两项实际调用均通过，独立
clipped公式梯度误差为0，模拟rank/micro拆分最大7.45e-9，57/64动作ratio落在
clip区间外，确实覆盖clipping。证据`context-hybrid-loss-cpu-v1.json`；无真实
模型训练和任务奖励收益。22项本地目标/接口/任务边界测试通过。

## 集合奖励的精确探索诊断

`binary-set-reward-diagnostic-v1.json`与`analyze_binary_set_rewards.py`计算已知
完整标签、每个bit以q独立采样的有限分布；通过4bit逐动作穷举独立校验。这是
探索策略算术，不是实际模型数据。q=.5、G=16时：

- 60候选一正例：exact奖励均值2^-60≈8.67e-19，组内恒奖励概率浮点精度下为1，
  数学上略小于1；全负bit准确率59/60。set-F1组内恒奖励概率约1.526e-5。
- 6候选一/两正例：exact奖励均值1/64，组内恒奖励概率约.7773。
- 6候选空集：显式定义empty-empty F1=1，此时F1退化为exact，组内恒奖励同样.7773。
- 多正例只返回一个oracle候选：两正例F1为2/3，完整oracle为1；集合命中无法区分。

因此不能把“存在组内奖励差异”当任务目标正确性的证明：bit正确率有差异却可能
奖励多数负例；F1可区分覆盖但不校准prob；空集任务仍可能稀疏。比较应报告组内
奖励方差/恒组比例、空/非空分层效果、返回数量和自然分布概率质量。部分标签不
计算完整集合奖励，不把缺标补负以制造可计分样本。正式reward仍未冻结。
