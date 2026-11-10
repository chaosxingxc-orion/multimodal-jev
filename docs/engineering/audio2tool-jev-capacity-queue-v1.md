# Audio2Tool-Jev 十二边界容量顺序队列

目标是在完整Jev任务输入定义下，验证Qwen3-Omni/vLLM的NPU prefill容量。
已有CPU输入检查覆盖四种音频布局×三种Jev决策共12例；实际NPU仍未运行。
每例只生成一个token，不代表完整JSON解码、质量分数、全量吞吐或RL提升。

固定入口为已验收`22e0157`的`probe_audio2tool_jev_engine.py`及110份源码，
直接只读挂载`audio2tool-jev-engine-preflight-v1/source`。准备时重新inspect原CPU
容器，核对命令、资源、挂载、环境、成功终态及源码与Git一致。完整create命令
保存于共享目录`audio2tool-jev-engine-capacity-v1/preparation.json`，SHA为
`acbdcce6d4862cc85c84e97d35458b4a1abd69db7ad31f26434f8c4b92fd7712`，
摘要见[准备信息](audio2tool-jev-engine-capacity-v1-prepared.json)。

执行保持原固定镜像、BF16/TP2、32K上下文、音频上限7、max_num_seqs1、关闭prefix
cache、greedy/seed17/max_tokens1。NPU6/7、16 CPU/128 GiB/8 GiB共享内存，
3600秒加30秒终止宽限；模型、数据与源代码只读。HCCL起始端口63800。
完整输入与模型前后验哈希、实际引擎token IDs与12例真实processor记录逐位对应。

`watch_audio2tool_jev_capacity.py`在现有native容量队列之后等待，不改变既有
SLURP→SFC→sampling→native容量顺序。等待绑定原native队列plan及启动收据SHA，
缺少终态时检查实际PID/start identity和非zombie状态；不会把心跳文件当作存活。
native成功后重新验证其前驱sampling验收，并使用最新Docker inspect调用已有
native `require_capacity`核对实际模型、命令、资源、源码及全部四例容量结果。
仅存在acceptance文件不足以放行。

随后获取共用`npu67-sampling-resource.lock`并检查实际设备容器占用；容器检查空闲
后再用`npu-smi info -t proc-mem -i 6/7 -c 0`检查Docker以外的驱动可见进程。
该命令在175的实际NPU6上返回了现有VLLMWorker和python3进程，确认命令可用。
当前仅接受空stdout为空闲；非空或未知格式保留到device-process-observation并
继续等待，不能猜测其为空闲。真正空闲时的驱动输出仍需实际观察。等待总上限
350000秒，实际容器观察上限3900秒。派发前重查依赖与源码，持久写完整intent，
create成功后先保存精确容器ID/intent SHA，再start；create/start失联保留歧义，
不重发。观察阶段的瞬时SSH错误只重新观察同一ID；观察超时不声明容器已经停止。

成功退出后，以协调者实际捕获的Docker inspect调用Jev `require_capacity`，核对
完整12例、原始输入/源码/模型身份、资源及一token响应，保存独立acceptance。
本队列不自动派发1404万次全量生成或训练；全量吞吐和正式预算仍需完成。

队列和既有两类capacity门禁共43项测试通过，涵盖存活依赖、PID替换、失败前驱、
实际inspection传递、验收结果变化、create/start歧义、容器身份变化、同ID观察重试
与禁止重放。完成依赖和容器分支为合成fixture，不冒充实际NPU运行证据。队列启动
身份另行保存；本准备说明不代表队列已启动或容量已通过。

## 队列已启动，等待实际 native 容量前驱

协调者源码固定`17f5442`，实际PID `2932524`、process_start `291613588`，
计划SHA `60062a646a9aa20fb5d11ce05a8762f9d46612edae92cc51cabc7b016951e83a`。
独立核验`/proc`的PID/start/非zombie状态及完整argv，114份队列源码与Git快照
一致；实际progress为`waiting_accepted_native_capacity`。原native协调者
PID `2895373`也核实身份与存活。见[队列启动核验](audio2tool-jev-engine-capacity-v1-queued.json)。
尚无Jev NPU intent/container/结果；实际完成依赖后的分支仍需运行验收。
