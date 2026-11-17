# 分段rollout候选：实际调度元数据桥接

接续[attention与音频组件](context-segmented-attention-candidate-v1.md)，新增
`tools/reports/context_segmented_worker_candidate.py`。这是下一版本的显式安装入口，
未接入当前训练或排队服务，没有新的模型运行或训练准入。

## 固定镜像来源

从原TRAIN processor容器读取固定镜像中的5个实际源码文件，未加载模型或修改运行依赖。
完整源码与hash保存在`.local/decision-objective-research/context-segmented-runtime-source-v2/receipt.json`。
包括vLLM forward_context、GPU model runner、Qwen3-ASR，以及Ascend metadata/runner。
v1按sys.path拼接查找未找到文件，保留空收据；v2用顶层find_spec定位editable安装成功。

源码确认：`ForwardContext.attn_metadata`是按层字典，`no_compile_layers`持有真实Attention
对象。Ascend runner在所有num_computed_tokens为0时标记PrefillNoCache；metadata同时提供
累计query边界actual_seq_lengths_q、完整seq_lens_list和真实token计数。
模型默认单请求profile不能直接提升max_num_seqs，因为语言attention与短音频入口均有限制。

## 实现

通过当前self_attn.attn的对象身份查找唯一层键；只消费该层metadata。要求：

- eager、TP1/PP1、无prefix cache/chunking、BF16、已完成加载和startup profiling；
- PrefillNoCache、pooling、causal、无decode，prefill数量等于请求数量；
- query累计边界等于完整序列长度累计和，真实token数等于hidden行数；
- 每请求positions从0连续增长，无graph padding、DP metadata或ubatch切分。

全部校验后交给上轮分段attention；声学wrapper调用原已对齐的逐录音forward，保留
短音频CNN边界及长音频原行为。先校验所有层再安装，避免后一层不合格留下前半安装。
没有复制训练框架、改变LoRA/head或修改现有profile。安装器必须由未来独立owner在原
startup/profile结束后调用，不能当成现有worker扩展直接派发。

29项本地tests通过（1.88秒），包括21种调度错配、真实对象身份匹配、安装前全层校验、
声学分段委托及6类runtime拒绝。测试metadata、模型和enum为替身；属于基于实际源码的
接口桥接验证，不是实际调度器生成的批次或Qwen/NPU整模型结果。

## 真实元数据导入探针的限制

另通过原CPU容器的独立docker exec进程尝试导入真实ForwardContext/AscendMetadata并构造
CPU实例。SSH在55秒观察超时；随后对同容器docker top核验仅原processor主进程存活，
探针进程已缺失。没有准确退出码/保留stdout，不能记为通过或推断导入根因。
没有重启processor或重新派发探针。观察记录见
`context-segmented-worker-native-metadata-v1-observation.json`。
下一次此类验证必须用独立准确exec/container句柄和持久终态日志，避免交互stdout丢失；
真实NPU验收仍需等待原资源队列，不以本地tests替代。

## 后续

仍需独立owner的批量encode请求身份绑定、容量配置及worker注册，然后有限真实NPU
串行/批量概率、邻居/顺序不变性与吞吐检查。对初次分块输入，单靠position和KV长度无法
判断完整prompt，必须同时保证禁用chunking，并保留现有pooler完整输入/marker检查。
本候选不新增为当前简单任务的继续门槛。

同轮TRAIN原edf4509f已普通观察running/nonOOM，18944/40513；local1794717046.5654233，
下一普通观察>=1794717646.5654233。正式1078/TRAIN1073/finalizer1167依赖重新hash无变化。
GRPO最近仍live-v37的227/634，下一普通>=1794717146.2381775，不提前轮询。
