# 原生verl混合actor接入与规模化指令预检

2026-11-12。新增`span_hybrid_actor.py`，通过verl已有policy-loss注册接口接入
candidate CE/有限候选PPO混合目标。没有复制或改写原生update_policy：原生循环
负责microbatch、backward、梯度裁剪和Adam step。typed forward只送音频、模板
编码、Span位置和候选token给backbone，监督在外部上下文传入注册的loss。

instruction阶段不需要behavior概率。为兼容原生DataProto保留的old_log_probs等
占位张量不解释为rollout，不进入监督loss。task阶段必须提供完整旧策略概率，
每个候选与当前同策略forward进行原有阈值的一致性检查。

通过线程局部ContextVar绑定当前microbatch概率张量与监督，注册loss只能消费
同一张量一次；finally清理上下文，拒绝嵌套更新和未消费的forward。
每个microbatch先用全局有效来源分母计算loss，再补偿原生actor的accumulation
除法，避免重复缩小梯度。四rank集体检查请求唯一性、同策略、分母与阶段系数。

## 已完成的真实验证

固定镜像`52ba94e1…`，容器`953c8bbf…`于14:57:12–14:57:36 UTC实际exited/0。
4个真实Gloo/DDP rank、micro1/2、各3次更新，instruction→task混合目标，
中间保存并恢复Adam/参数，使用同一个原生actor进行连续更新。
覆盖全mask rank、零权重来源、不同候选宽度及部分监督。

独立参考逐来源计算CE与原PPO数学目标；实际更新调用原生verl loss回调。
最大梯度/参数误差均2.9802322387695312e-8，Adam一阶矩误差3.725290298461914e-9、
二阶矩2.1827872842550278e-11。所有rank拒绝篡改分母、mask、旧概率、阶段系数
和跨rank重复请求。[真实凭证](../../../slurp-hybrid-native-actor-dp4-cpu-v1-verified.json)。

这是合成FP32 CPU原生actor证据；没有将其当作BF16音频NPU或vLLM闭环验收。

## 两个已启动的执行入口

完整processor检查：`slurp-instruction-processor-full-v2`，容器
`a917b8a985ab269ffc87b701c5c55b026231b2ba825b449b22b723023cd45843`。
目标覆盖已冻结两轮日程81,026个非padding录音视图，实际Qwen3ASRProcessor、
完整Span位置/候选映射/上下文长度，4096上限，不截断或删除超长样本来掩盖失败。
v1已在音频处理前实际失败：远端/home与本机不同，审核凭证路径不可见。v2仅更正
为共享存储中逐文件hash一致的证据挂载，保留原失败，未因超时重启。

真实NPU预检：`slurp-instruction-dp4-prefix-v1`，容器
`10dd97c64b6a6e824d68b30704088092a820034265fd5485a42098468254271e`。
启动前持有4–7设备派发锁，检查实际设备占用；仅映射授权四卡。
Qwen3-ASR-0.6B、新LoRA r4/alpha8、语言q/v、BF16执行/导出、FP32 master/Adam，
FSDP FULL_SHARD DP4，global128、micro1、每rank32次累积、lr1e-4、seed317。
连续两次更新采用全量epoch0的首256槽位，记录真实执行dtype、峰值显存与更新秒数。
每个输入在训练前真实processor检查不截断，监督外置；导出初始与两更新BF16 adapter，
保存四rank master/Adam及完整日程游标。源码/配置冻结，父数据及证据精确挂载。

这两步属于全量运行的执行验收前缀，不是新小样本研究任务，不报告其学习集分数
为Stage1收益。当前入口没有实现恢复继续，保存checkpoint不等于已支持连续恢复；
尚需独立重载、完整后端检查及恢复入口后，才能从该状态或另立正式起点跑全量。
大规模训练仍需覆盖每轮40,513录音，两轮634次更新；不得把预检成功冒称完成。

运行/启动细节保存在共享实验根目录对应`launch.json`，后续普通状态检查间隔600秒。
完整processor、NPU实际终态和独立验收未到齐前，不派发整个Stage1或Stage2。
