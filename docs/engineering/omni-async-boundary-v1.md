# Omni 有序音频与完整采样组的异步接口

`omni_async_rollout.OmniAsyncSampler` 为后续 AsyncLLM 接入提供直接音频边界，
复用 SFC-Jev 原始消息构造和 `omni_rollout_records.capture`。它不是已接入的
verl Ray manager/server/agent-loop，也没有替换当前冻结评测或 K2 采样队列。

## 源码发现与处理

重新读取固定镜像的 `/opt/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
中 generate 方法：默认只把 image/video 写入 TokensPrompt，并只返回 outputs[0]。
它使用 `0 if sampling_params.pop('logprobs', False) else None`，传入整数 0 会
变为 None。vLLM 的原始 SamplingParams 中 0 表示返回采样 token 的 logprob，
所以不能将本项目的参数字典直接交给该 wrapper 后假定仍有完整概率和 K 副本。
这不表示所有 verl 配置都错误：该 wrapper 可能期望布尔开关及每请求 n=1，
但与本项目 K 输出接口不兼容，必须明确适配。

新边界直接向 AsyncLLM 传原始模板文本及有序 `(waveform,16000)` 列表，
由引擎展开一次占位符；结束后逐 token 核对展开 prompt 与 actor 预处理结果。
SamplingParams 明确 n=K、logprobs=0、temperature=1、无分布截断/惩罚，
引擎配置要求 raw_logprobs、generation_config=vllm 和完全相同的 context 上限。
请求不满足完整输出预算时拒绝，不静默缩短 max_tokens。所有副本按引擎 index
归档，缺失/重复副本、文本与 token 不一致或概率缺失均拒绝作为完整组返回。

每次请求使用显式唯一 ID，实例在调用引擎前登记 ID，中断或格式失败后也不能
复用。未完成流会调用 abort，abort 的错误附记在原始异常上。保留 last_output
供调用方保存失败证据，不修复输出。跨进程恢复仍需调用方的持久 dispatch intent，
内存集合不构成持久幂等保证；波形哈希、学习分区和模型版本也由调用方认证。

## 验证边界与下一步

本地模拟异步引擎测试覆盖双音频有序传递、完整 K 输出、部分流、空流、
请求错配、token 错配、概率缺失、取消、重复请求、上下文超限及引擎配置改变。
实际 SamplingParams 另在固定镜像 CPU 测试；没有加载模型或占用 NPU。
固定镜像容器已退出 0，34 项测试全部通过、无跳过，耗时 16.36 秒，见
[测试凭证](omni-async-boundary-v1-result.json)。包含真实 SamplingParams 的测试
确认 n=2、logprobs=0 在新接口保持不变；引擎仍为测试替身。

下一步在独立冻结 TRAIN 协议中比较同一 processor、音频、种子与采样设置的
同步/异步引擎输出身份和概率，保留全部 78×K 决策及失败成本；不能用模拟引擎
测试替代真实生成。原生 verl 接入还需 manager/server 传递音频、agent-loop
后处理的音频特征与位置编码、完整 prompt 分组、policy identity 和导出刷新。

## 完整学习探针执行器

`tools/jev/sample_sfc_omni_async.py` 已实现完整六条学习回合、78 个决策、每题 K2
采样程序。复用同步程序的真实 processor/分区验证与引擎参数，保留两段波形顺序、
原始模型哈希检查及 4096 上下文/1024 输出预算。请求前独占创建并 fsync intent，
每次流结束或失败均保存最后原始输出；失败另保存异常，不跳过或重放。
全部 156 个响应齐全才生成汇总，退出时显式调用 AsyncLLM.shutdown(timeout=30)。
真实生成尚未派发，不修改现有同步采样队列。

三个 runner 测试用替身验证全量 78 次请求/156 个响应、先登记后派发、部分输出
保存、音频校验失败不派发和关闭引擎。本地相关测试为 36 通过、1 跳过；跳过的是
实际 vLLM 包测试，其固定镜像结果已在上一节单独验证，不能算本地通过。

真实 CPU 预检查 v1 已完成六回合/78 决策 processor 与 SamplingParams 校验，
随后构造 AsyncEngineArgs 时退出 1：默认 Ascend general plugin 导入 Triton 算子并
调用 get_arch，而容器没有 NPU 设备。该失败保留在
`omni-async-train78-preflight-v1`。v2 明确设置 `VLLM_PLUGINS=`，只检查实际参数
对象的构造，不挂载 NPU；因此即使通过，也不构成 Ascend 插件或引擎兼容性证据。
两次均使用相同源码快照 `f8bd33c`、固定镜像、2 CPU/12 GiB/600 秒，不加载模型。
v2 已退出 0，全部 78 个决策通过，实际 AsyncEngineArgs 构造完成，总耗时 45.33 秒。
两次精确容器、原始成功/失败终态和预检查结果见
[完整记录](omni-async-train78-preflight-v1-v2.json)。真实 NPU 异步生成与同步概率比较仍未执行。
