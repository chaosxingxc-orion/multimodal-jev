# Audio2Tool 冻结基座引擎容量预检查

目标是在已验收的正式native输入上检查Qwen3-Omni/vLLM能否完成长音频前向。
这不是学界基线成绩，也不替代完整任务生成。Audio2Tool保持external_test，
不根据生成token、任务分数或标注选择提示；四例只由输入长度/音频数量确定。

## 固定执行定义

`probe_audio2tool_engine.py` 复用完整输入选择器、正式renderer与验证音频加载器，
将每例prompt哈希、转换后波形哈希/补零/帧数与已验收processor记录逐项核对。
输入完整30733行和预算文件、原模型audit均固定SHA；实际engine执行前后核对
完整原模型文件哈希。禁止替换模型、缩短输入、改变任务或自动重试。

- 原本地Qwen3-Omni-30B-A3B-Instruct；BF16、TP2、seed17、eager。
- 上下文32768，max_num_batched_tokens32768，max_num_seqs1，audio上限7。
- gpu_memory_utilization0.85，关闭prefix cache；使用当前固定工程镜像与版本。
- 四例：Tier1:3943、Tier7:2030、Tier8:314、Tier7:1118。
- 每例temperature0、n1、max_tokens1。记录生成token ID、finish reason和耗时，
  不计算任务分数、不使用标注；要求实际engine展开token ID与已验收processor一致。

只生成一个token是明确的prefill容量测试，不是验证完整1024-token解码。
即使成功仍须完成正式生成、全量覆盖、失败账本和评分验收。错误必须保留到终态，
不得因长音频失败自动缩短、换样本或更换配置后覆盖原尝试。

## 已完成的CPU输入预检查

源码 `db9db7a` 的 `--preflight-only` 路径实际运行于固定镜像容器
`228aa79e9b7accc09b34488d1d1f04bc6eed72882b39f11d80005b355ec12914`，
2 CPU/20 GiB/900秒上限、无网络/设备，退出0，主体5.48秒。
四例prompt及全部音频收据与实际HF/vLLM验收记录一致。预检查只验证processor资产，
不加载或重新哈希完整权重；完整权重核验分支尚未在此入口实际执行。
结果见[CPU验收](audio2tool-engine-preflight-v1-verified.json)。

相关10项测试通过，包含同长度错序token ID、未完成响应、多个输出、空/多token、
abort、非法token类型、提示/波形收据/采样率变化等反例。测试响应为替身，不能
替代真实NPU执行。对其依赖的完整输入选择器另有实际CPU容器证据。

## NPU执行准备状态

已保存可审查的docker create命令与完整source manifest，直接引用已验收的只读源码
快照，预算NPU6/7、16 CPU、128 GiB、8 GiB共享内存、3600秒加30秒终止宽限。
记录见[准备信息](audio2tool-engine-capacity-v1-prepared.json)，共享目录的
`audio2tool-engine-capacity-v1/preparation.json` 保存完整命令。

当前未创建NPU容器，也未启动自动调度器。现有SLURP→SFC→sampling队列继续优先；
派发时需核验sampling成功验收和精确容器，取得共享NPU6/7锁并检查设备空闲，再固定
本次create intent和容器ID后启动。准备文件不是运行成功证据；未来实际执行与验收
仍需记录容器命令/资源/终态、四例输出以及源/模型/输入不变性。
