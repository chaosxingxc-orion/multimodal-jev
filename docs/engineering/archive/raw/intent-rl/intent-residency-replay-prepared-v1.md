# 真实Qwen-ASR卸载恢复重放准备

2026-11-12 UTC。目标是支持后续常驻verl训练，隔离测试真实语音模型
卸载/恢复会不会改变下一步更新。没有修改已冻结D/Q/P队列，也未派发
新NPU任务。共享准备目录：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-residency-replay-prepared-v1`。

## 固定对照

重放已验收D第2步，恢复第1步的FP32 master/Adam，使用原模型、adapter、
音频来源、模板、种子317、batch8、lr1e-4和同一个verl更新算子。唯一新增
操作是在完整候选前向准入之后、更新之前执行FSDP模型和Adam卸载/恢复。
前向先初始化FSDP lazy state；原训练入口默认不执行此回调。

卸载边界调用固定镜像中已审核FSDP搬移函数，逐张量保留优化器原device，
保持BF16执行、FP32主参数与Adam，验证参数对象和优化器引用、值与dtype。
恢复后由原actor再次核对候选概率，执行原第2步并保存四rank状态、导出。
这是一条工程重放，不计为新增学习步或算法提升。

## 需要通过的真实验收

1. 已排队小模型NPU卸载探针通过，既有队列正常收尾且设备所有权明确。
2. 实际四rank训练与独立suite容器退出0，源码和输入哈希保持一致。
3. 四rank卸载/恢复的参数、Adam、dtype、引用与设备均保持不变。
4. 重放后FP32 master、Adam step/两矩逐元素与原第2步相等；同一BF16
   adapter身份；来源和监督相同，不能只看导出权重相等。
5. 独立178输入HF/vLLM gate与实际训练后输出验收通过。

`verify_intent_residency_replay_cpu.py`用于第4项的独立CPU张量对照，并
检查既有完整验收产物。它不访问Docker daemon；调用者仍必须独立观察
准确容器退出。工具尚未对真实重放执行，也尚未加入自动队列。

该实验没有在actor卸载期间运行vLLM，不能证明二者同时占用设备可行。
HCCL、buffer与缓存仍可能驻留，内存读数不能直接当作vLLM容量；下一阶段
需另测受控交接。失败不能自动重复更新或只恢复权重而丢弃Adam。

## 当前证据

固定镜像CPU容器17项状态测试通过，验证精确Adam状态对照、sub-BF16
master变化拒绝、各优化器分量/拓扑/引用/dtype损坏拒绝及卸载恢复控制。
首版测试因CPU Adam默认foreach路径触发torch_npu设备探测失败；第二版
显式使用与真实训练相同的foreach=False/fused=False，通过17项。未添加
NPU设备给CPU测试，也未降低数值比较要求，失败产物保留。

8项本地配置测试通过，拒绝改变arm、step、batch、学习率、初始adapter、
行为gate或Adam设置，要求全部参考rank状态pin。实际计划挂载的无NPU
预检容器退出0，验证全部pin和配置可读；镜像匹配、Devices为空。
证据见`intent-residency-replay-prepared-v1-verified.json`。

准备时重新认证原第2步真实训练/suite退出及完整验收；计划源码和镜像
已冻结。协议SHA256：
`21ab8c4ca25818441d9fdcee00c0f9ec63204a5355347304f23a76e443bff200`。
