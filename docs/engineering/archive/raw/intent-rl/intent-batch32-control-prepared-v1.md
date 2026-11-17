# 固定来源预算的batch32实验准备

2026-11-12 UTC。在四Gloo rank、global32、microbatch1/2/8的真实verl数值验证后，已冻结batch32首步训练与独立178输入验收配置。实验根为`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-batch32-control-prepared-v1`，协议SHA256为`f76147b2a26551ad53f05737d392590b7ed7600a177262d8c4d5514ccfadd5f8`。

从原temporal adapter、全新默认Adam出发，D任务、seed317、lr1e-4、BF16执行与FP32master/state、正确性有限支持奖励均保持不变。只改变global_requests8→32，每卡8条，microbatch1。完整预定预算为3步×32=96次来源使用，与原12步×8的扁平来源顺序逐项相等；全89唯一来源、7次重复。固定报告step0–3并以累计32/64/96次来源使用对照原D4/8/12，不声称相同更新次数、行为策略刷新次数或算力。

训练与验收沿用原始冻结src/tools，仅覆盖经审计的五个文件：训练入口、独立验收入口、优化器配置、恢复谱系与batch布局。其他快照文件完全一致；完整差异见`intent-batch32-control-source-audit-v1.json`。新增优化器模块在不显式配置beta时保持原默认行为；训练入口继承已验收的导出chmod0644。数据、音频、模型前向和奖励实现未改变。已有冻结动量、D/Q/P与低学习率队列不受本工作树变化影响。

43项配置、日程、谱系和源码冻结测试通过。准备时实际核对参考资产pins、全部日程和挂载可见性，拒绝其他lr、动量、seed、初始化等混杂。当前只准备首步；后续2–3步待首步验收后独立准备，未新增NPU容器或后台排队任务。该配置尚无真实音频NPU结果，不把CPU等价性当作任务收益。
