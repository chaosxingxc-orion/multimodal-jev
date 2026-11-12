# 优先推进奖励与常驻训练验收的队列迁移

2026-11-12 UTC。已迁移13个尚未派发任务的等待进程，新进程均以实际
`/proc/PID/cmdline`核验存活，依赖图检查无环。Q/P及lower-LR训练控制器未改动。
这是调度进展，不是新增训练收益或常驻trainer验收。

## 调整依据

实际成功退出的Q1–5，训练容器耗时中位数61.07秒，完整178输入的HF/vLLM
独立suite耗时中位数392.88秒。冻结FSC矩阵为7策略×6视图×3793音频，
每后端159306请求、两后端318612次forward。线性外推约94.98–109.07小时，
中位估算97.67小时。参考耗时包含启动/状态验收，FSC音频和上下文不同；
这不是置信区间或完成时间承诺。

旧依赖把全部奖励优化与常驻验收放在FSC之后。新顺序为：

1. 原冻结D/Q/P及lower-LR D12正常完成。
2. 绑定诊断、beta1zero首步、合成驻留、batch32、beta1zero后续步。
3. 绑定首步、真实音频驻留重放、zero-delta、三臂两步utility。
4. 常驻vLLM A/B/A切换及两更新常驻actor独立验收。
5. 原七策略六视图完整FSC矩阵及学术参考配对报告。

实验的数据、奖励、起点、来源日程、训练预算、模型及prepared源码没有改动。
FSC继续先核验实际lower-LR D12验收，再核验优先链尾的真实训练/suite容器
exit0及完整独立验收；没有绕过数据或后端gate。绑定诊断改为直接等待同一
lower-LR D12，不再依赖与其输入无关的FSC测试完成。

## 迁移与验证

旧进程停止前均检查精确argv、固定config/source及run目录只有progress。
在四卡派发锁内用pidfd停止精确等待进程，等待进程退出，再发布新版本。
旧目录只新增`superseded.json`，明确不是成功终态；旧config/source保持不变。
新目录采用v2，原residency-replay v2升级v3，且保留原prepared任务输出位置，
禁止重复派发。后续仍每600秒观察一次。

14项定向测试通过，覆盖原完整FSC串行验收、优先依赖失败/失踪、600秒等待、
拒绝仅凭成功标记提前派发、两实际容器与验收哈希，以及预算计数。
迁移后逐项比较配置，除调度依赖与源码pins外的workload字段完全相同。

- [实际进程与依赖核验](intent-priority-schedule-v2-verified.json)
- [耗时证据摘要](intent-queue-budget-v1-verified.json)
- 完整迁移脚本、计划、旧进程退出、新launch及workload核验：
  `/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-priority-schedule-migration-v1/`

链中任一必要验收失败仍会阻止后续派发；不得把失败改成成功或靠超时重启。
