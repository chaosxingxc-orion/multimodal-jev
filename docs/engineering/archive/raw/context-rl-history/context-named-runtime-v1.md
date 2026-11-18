# 上下文概率头统一执行配置

`context_runtime_profile.py`定义`qwen3-asr-context-bf16-packed-row-v1`，将先前分别
执行的组件绑定为单一显式配置：HF侧固定128×8投影、packed attention与逐请求浅头；
vLLM侧HF顺序、短音频处理、packed attention与固定投影；vLLM仍为TP1单请求完整
prefill/native LoRA。配置安装不是训练准入，也不自动启用缓存、解码或批量vLLM。
联合policy的runtime_profile包含该版本，改变执行方式会改变policy身份。

新增真实音频命名入口探针e42ee9ea已由exact docker wait确认exit0，完整收集复核通过。
该探针在统一配置下进行诊断梯度更新、FP32 master/Adam保存恢复，再经独立vLLM
进程推理。两条真实音频各60概率，HF/vLLM最大差0、阈值翻转0；128参数
FP32 master/Adam与BF16导出逐张量核对一致，恢复概率差0。联合policy为
`26200e05ce48ab413073d7c6eed314d540847a92c53e6f45d7216ba3211a6754`，HF/vLLM
receipt与policy contract均绑定该命名配置。证据`context-engine-named-runtime-npu7-v1.json`。
不得将两条或原256条结果当成后续多卡更新态验收。

## 多卡准备

`context_fsdp.py`保留骨干原有Linear包装，但将完整浅Transformer头作为一个FSDP单元。
MultiheadAttention以functional方式使用out_proj，单独包装它会绕过pre-forward，
因此头内部不再划分Linear单元。ContextDecisionModel通过head(...,return_logits=True)
进入模块forward，保证完整头的FSDP收集/混合精度钩子执行；默认forward仍输出prob。
21项本地head/collate/rowhead/FSDP选择测试通过；修订后的四进程CPU v4
574ba353已exit0且完整收集48case通过，见`context-native-actor-dp-cpu-v4.json`。

`probe_context_native_actor_dp_npu.py`及本地启动器已prepare-only：真实学习256条、
全部60候选、DP4/global128/micro8×accum4、BF16执行、FP32master/Adam、原生verl
instruction loss两步。第三次执行从第一步checkpoint恢复重放第二步，对比每个本地
master/Adam张量；检查冻结参数、执行dtype、各rank完整BF16导出一致性和分项时间。
这不是把研究训练缩为256条；是全量40513两阶段训练之前的固定规模工程前缀。

资源日程已协调：181八卡均被现有vLLM占用且缺少锁定镜像；175新四卡入口与待启动的
full-parity/selection启动器共同使用本地flock，四卡probe持锁准确docker wait直到终态。
原full-audit与driver未停止或重启。probe容器
`0aacafce24afdbbaf59a473616f00b2a406bc345457a536e6a31f64373d76237`
首次在175物理4–7启动，后续到期检查发现torchrun standalone选择了无法解析的
容器hostname，未出现任何rank进度；这是确定的rendezvous错误，非训练计算慢。
已记录stop-request并主动停止，exit137/nonOOM，完整日志归档为
`context-native-actor-dp-npu-v1.json`，不能当成自发OOM或已执行训练。
v2改用static `127.0.0.1:29617`，容器
`c10552d84ad8abed5743f407d10dba5db9c1e5f8b1eb453d28a22e9fcecc5675`
已启动；root session28977准确wait并持同一资源锁，尚无更新成功结论。
普通观察仍launch+600秒，准确等待无轮询。收集器
`.local/decision-objective-research/collect_context_native_actor_dp_npu_v2.py`已准备，
将逐文件hash、学习人口标签、BF16导出和各rank FP32master/Adam复核；不得凭启动即称通过。
实际FSDP、恢复、更新态vLLM概率和稳定吞吐未证明；生产token仍为待冻结的独立事项。

## 更新态导出与输入缓存衔接

`context_fsdp_export.py`将FSDP组装BF16张量映射到普通模型的LoRA/head参数，
只移除完整wrapper路径段，拒绝键冲突、缺失、额外参数、shape/dtype/非有限错误后
才执行加载。保持原冻结marker和基座不变。17项导出/策略/FSDP相关本地测试通过。
`probe_context_engine_audio.py`可消费已认证的assembled_execution，并禁止再叠加
诊断更新；后续256条HF/vLLM启动器已写好，要求DP4终态审核通过才读取权重。

新`cached_context_preparation.py`复用原本地CPU输入缓存，额外绑定marker/audio ID、
上下文预算、BF16特征及编码版本，policy/监督不入缓存。7项缓存测试通过。
真实CPU256条冷/热/整批预取检查bcae8183已启动，exactwait句柄见工具；
限制2CPU，无NPU，不据CPU服务时间声称训练加速。CPU缓存exactwait7032；四卡v2
继续exactwait28977，均不以等待超时推断终态。

最新多卡v3已exit0完整审核通过，详情context-full-training-inputs-v1.md及
context-native-actor-dp-npu-v3.json。此前v1/v2状态是历史失败，不再等待旧句柄。
