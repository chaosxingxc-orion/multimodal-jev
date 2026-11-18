# 简单决策奖励候选的完整短程曲线

2026-11-12 UTC。原三臂两步计划只验证真实更新、恢复及后端一致性。
本版在三臂尚未派发、未观察其结果时，预先登记每臂累计12步，不改变原两步协议。
扩展是否派发只取决于工程验收，不依据前两步的学习分数选择赢家。

## 固定研究因素

| 臂 | 信用 | 作用 |
|---|---|---|
| correctness | 原0/1正确性条件优势 | 同源码控制 |
| smoothed-utility | U'(p_old)乘原正确性优势，tau=1/K² | 测试弱正确概率来源的信用重分配 |
| uniform-credit | 固定8倍原正确性优势 | 区分统一尺度变化与相对信用变化 |

三臂使用同一初始化、fresh Adam起点、seed317、lr1e-4、每步8来源。
第3–12步从各自第2步FP32 master/Adam继续，不重置动量，不切换目标。
累计每臂96来源使用次数，固定循环日程覆盖89来源；不称96独立样本。
所有新增训练/suite使用对应两步试验的原源码快照，避免同时改变实现。
BF16执行/导出、FP32 master/Adam、DP4和vLLM TP1保持原准入要求。

固定scale8只作为已登记尺度控制，不能声称精确匹配每批梯度或Adam位移。
平滑效用改变优化偏好，不是原平均正确率目标的无偏修正；不宣称新RL理论。

## 验收与解释

必须报告step0–12的全部学习人口结果：Final accuracy、macro-F1、预测类别覆盖、
正确概率、NLL、margin，及各步KL、梯度裁剪和实际Adam位移。不同目标的
总reward数值不可直接横比，主要比较同一正确性评分和Final判决。
允许得出无提升、波动、类别塌缩或负收益，不能挑最好一步代替曲线。

扩展前要求三臂各两次真实更新与独立suite成功、精确恢复，且correctness第2步
通过与原D2的实际master/Adam/导出/概率重放比较。已存在比较工具不等于已通过。
工程失败阻止派发，不能改写成功记录或自动重启状态不明的任务。

当前89条学习来源只经参考文本意图审核，未复听；没有已准入的独立开发评测。
因此本研究首先回答优化是否学起来，不能据学习曲线宣称泛化。
冻结FSC矩阵仍只包含原D/Q/P检查点，不能把新utility检查点悄悄加入其测试选择。
后续独立评测和迁移需另立预注册版本并完成监督准入。

## 准备产物

生成器：`tools/jev/prepare_intent_utility_curves.py`。
根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-utility-curves12-prepared-v1/`。
新增30个训练/suite配对；连同既有前两步共36配对。准备不派发，不改现有队列。
协议检查覆盖缺失实验臂、减少检查点、改变学习率/目标、重复来源/输出和改写
原前两步。后续调度需逐项验证真实工程验收后再使用这些prepared计划。

准备已完成；独立逐项核验36组既有/新建训练suite、72个独立容器名称、
源码一致性、全部已知pins、真实来源日程及连续状态配置通过。24项协议和
延续相关测试通过。[准备验收记录](../engineering/intent-utility-curves12-prepared-v1-verified.json)。
复算脚本`tools/jev/audit_intent_utility_curves12.py`固定审计本版根目录；
运行后会重新写审计结果，不会派发NPU或修改实验协议。

后续自动执行入口已排队，工程验收和correctness实际状态重放通过后才继续，
完整FSC矩阵安排在其后；参见[控制器证据](../engineering/archive/raw/intent-rl/intent-utility-curves12-controller-v1.md)。
本段更新调度状态，不改变冻结曲线协议或说明训练已完成。

补充只读报告入口已实现：`tools/jev/report_intent_utility_curves.py`，将每步
gold margin、更新后KL/ratio裁剪质量、四rank更新前梯度及裁剪日志汇总，
要求所有36组完整且逐项独立验收。已用实际Q7验证单步读取，完整效用曲线
尚未执行；未修改冻结队列。参见[报告验证](../engineering/archive/raw/intent-rl/intent-update-metrics-v1.md)。
