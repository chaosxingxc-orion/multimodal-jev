# 意图有限候选RL第二步恢复准备

2026-11-12 UTC。D臂首步四卡训练容器实际退出0，四rank结果均报告一次更新、非零梯度范数0.5216301679611206、基座未改变和同一BF16导出。该观察见`slurp-intent-finite-dp4-v2-observed.json`，独立178输入/optimizer审计尚未完成，不能仅凭rank结果准入续训。

第二步已在新目录冻结源码、config-template及启动参数。保持D、seed317、8条全局请求、lr1e-4、BF16 LoRA与FP32 master/Adam、物理4–7 FULL_SHARD DP4及v2短音频profile；step从0推进到1，按固定日程消费下一全局批次。initial adapter来自首步实际导出，行为概率必须来自首步独立验收suite。

新增`freeze_intent_continuation.py`：先检查确切训练/验收容器都在固定镜像上退出0，再验证配置、源码和全部验收引用文件，绑定四个实际optimizer分片及新行为策略。调用严格lineage核验，解析最终容器挂载后才写config。失败或尚在运行的前序不能冻结，已声明目录不可覆盖；3项控制器测试通过。

当前1179个已知pin的挂载检查通过，未来结果仍待首步独立验收后冻结。未启动第二步。已有CPU四进程恢复等价证据仅作为实现准备，不能替代语音BF16/NPU实际恢复与下一次独立导出验收。
