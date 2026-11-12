# 可复用常驻 Span rollout

2026-11-12 UTC。连续RL缺口之一是每次策略更新后为新导出重新运行vLLM。
新增`ResidentSpanRollout`，将原A/B/A探针内的实际推理实现提取为可接收新
策略配置的常驻组件。开发版探针已调用此组件；已冻结排队的探针源码不变。

## 已实现

- 同一TP1引擎使用BF16基座/LoRA/head、完整非缓存prefill、原运行profile。
- 每个不可变adapter内容身份分配唯一slot；新策略不用旧slot冒充，返回旧策略
  时保留原slot身份。实际worker必须只注册当前adapter，并报告BF16缓冲与head。
- 每轮输入可使用新策略独立HF参考；完整人口、请求ID、音频特征张量摘要、
  候选顺序、Span位置及策略身份逐条匹配，worker展开后的token也必须完全相同。
- 原全候选/联合概率阈值不放宽，前后重算adapter内容身份，防止运行时改写。
- 任一输入、推理、回调或worker检查失败使session失效，禁止直接重试。
  close显式关闭engine，不自动创建替代引擎。

该接口不接收监督目标或reward；不负责数据准入、设备租约、优化器更新或
接受训练策略。调用者仍须认证独立参考的来源和实际进程退出。

## 验证和剩余工作

固定镜像无设备CPU测试使用真实torch张量及pooling输出解析，engine由受控
替身提供，测试A/B/A身份与输出、音频压缩/展开、错误候选与特征、缺失/重复
人口、陈旧结果、运行期权重变化、错误worker槽位/dtype及失败后禁止重试。
同时运行既有探针配置、独立切换验收与候选传输测试。

这不是实际NPU热切换验收，也没有证明actor与rollout可以共用设备或常驻并行。
现有冻结NPU探针继续原队列执行。后续连接需完成：actor导出新策略、在明确
设备所有权边界运行新HF/vLLM概率、独立验收后更新session身份，再消费新
行为概率执行下一步；保存FP32 master/Adam和来源游标用于精确恢复。
最终需真实音频至少两个在线更新及独立状态/后端检查才能称为闭环。

CPU记录根：`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-resident-rollout-cpu-v2/`。
首版41项通过；v2加入真实输入摘要反例后43项通过，实际容器退出0，源码
与固定快照逐项匹配。[独立验证记录](span-resident-rollout-cpu-v2-verified.json)。
