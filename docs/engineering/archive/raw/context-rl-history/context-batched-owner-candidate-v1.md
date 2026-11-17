# 批量encode与独立owner候选

本轮推进下一版本吞吐优化；原GRPO/RLOO和下游评测不变。上一轮新增scheduler桥接属于
具体实现进展，本轮完成实际批量encode入口和独立owner/worker类，并固定镜像执行CPU回归。

## 完成内容

- `context_batched_request_candidate.py`：先验证整个批次的CPU输入、完整audio区域、marker、
  mask、容量及唯一输入身份，然后一次`encode(prompts, pooling_params=list)`，逐条核对输出。
  输入/输出身份与冻结`FixedContextRollout._score`一致。失败不回退串行、不返回部分批次。
- `context_batched_owner_candidate.py`：校验原`context_switch_rollout.py` SHA256后仅替换两处
  构造参数：显式batch容量和独立worker扩展。派生类在原startup/profile成功后安装分段组件。
  原switch/策略槽/activation/锁生命周期保留；score_many持同一个transaction锁，失败标记failed。
- `context_batched_worker_extension_candidate.py`：复用原SwitchingContextWorker，新增显式
  policy绑定的安装RPC。现有正式worker/profile未改。

这提供了可构造的候选代码，尚未执行实际owner模型初始化，也未接入bulk IPC服务。
`encode`一次调用不保证vLLM只调度一次模型前向；必须在真实运行记录batch规模/调用数。
容量必须满足完整请求及展开后token预算，候选不截断。重复输入目前明确拒绝，上游若
出现相同请求（例如补齐），需在独立服务内按身份去重并恢复顺序，不可绕过这里静默接受。

## 验证

本地新增19项加原请求传输11项：30 passed/2.06秒。证明批次仅一次encode调用、完整批前
校验、逐条与原路径wire identity精确一致、marker/音频内容绑定、返回缺失/错序/坏概率拒绝，
以及源代码派生保留原switch方法并拒绝源漂移。使用engine替身，不是Qwen概率验收。

固定镜像`52ba94e1`独立CPU容器`dc78572d36f890f87e66482941e156046ff36861c03fd9cd84dfeb374b799d27`，
373文件只读包，2CPU/6GB，无设备、无网络、非privileged，缓存写/data与/jev-tmp。
**104 tests通过/27.49秒**，覆盖分段attention、audio、scheduler、请求及原组件回归。
随后真实`ForwardContext`导入成功；`AscendMetadata`导入触发Triton硬件查询，在无NPU容器内
`get_arch`报SystemError。准确docker wait返回1，独立inspect确认exit1/nonOOM。
owner class和worker实际导入阶段未执行，不能称固定镜像接口整体验收通过。

初次collector记录了stdout但docker logs的stderr未纳入该文件；额外独立收集两流并归档完整
失败栈，没有改写原收据或重跑容器。结果与限制见：

- `context-segmented-owner-fixed-image-cpu-v1.json`：容器/镜像/全部文件身份/终态；
- `context-segmented-owner-cpu-v1-failure-analysis.json`：完整日志双流hash与失败位置。

该证据解释了无NPU导入的具体限制，不反推上轮交互探针的未知退出原因，也不表明正式
训练镜像损坏。无需为CPU导入修改镜像或伪造Ascend backend；在真实设备的候选验收中
核对接口/安装/实际请求，再比较串行与批量概率、峰值内存和耗时。

## 当前正式任务

GRPO普通live-v38 local1794717230.8171647，同一e564df04 running/nonOOM，四rank233/634，
64/128/192已提交；下一普通>=1794717830.8171647。TRAIN仍以live-v16为最近18944/40513，
下一普通>=1794717646.5654233。本地七个原driver实际ps存活，364基线和1078/1073/1167
依赖重验无变化。没有新的RL、TEST或跨任务迁移效果结论。
