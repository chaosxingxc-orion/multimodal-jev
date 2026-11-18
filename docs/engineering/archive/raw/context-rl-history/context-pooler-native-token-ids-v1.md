# Context pooler 与原生 token ID 接口

2026-11-14，NPU探针 `context-engine-updated-audio-npu7-v2` 正常记录失败终态
（exit1、非OOM）。模型已通过spawn注册和权重加载，失败发生在
`gpu_model_runner._dummy_pooler_run_task` 的启动显存探测，而非真实请求输出。
完整现场及失败堆栈保存在 `context-engine-updated-audio-npu7-v2.json`。

固定镜像vLLM源明确用 `torch.int32` 创建dummy token buffer；
`PoolingMetadata.get_prompt_token_ids_cpu()`仅按长度切片，不转换类型。
本项目 `ContextHistoryReadout`要求int64，故拒绝了合法的原生整数输入。
之前测试用int64合成metadata，没有覆盖这个真实接口差异。

修复仅在 `ContextProbabilityPooler`边界接受int32/int64并无损转为int64，
浮点与布尔token仍拒绝。完整prompt身份、marker、policy及prefill检查保留；
启动profiling仍须显式开启，不能凭缺少请求身份推断为dummy请求。

新增测试使用固定镜像真实 `PoolingMetadata`、`PoolingCursor`与`PoolingParams`，
覆盖int32启动和真实请求元数据、与直接head精确比较、原buffer不变、无效类型拒绝。
容器85889178已exit0，13项测试通过，源码pins核对通过；
证据 `context-pooler-int32-cpu-v1.json`。测试hidden为合成数据，不是实际引擎验收。

v3实际NPU探针5d9a7322已在175物理7卡启动，复用v1已更新的同一联合策略及
HF参考，逐文件hash确认复制不变；仅重新执行vLLM阶段，不重复optimizer更新。
启动时卡7空闲，完整selection链使用卡5/6，与本probe分离。
该记录写入时仍在确切docker wait，不能宣称已有推理概率或训推一致性结果。

后续终态：v3已exit0/nonOOM，完整收集且独立核对输入、联合策略、原生LoRA、
128个FP32 master/BF16导出及Adam状态。两实际音频各输出60个概率，
HF/vLLM最大差分别0.00853425与0.01041567；0.5阈值翻转分别0/1个。
实际engine执行已打通，数值一致性未准入，不能作为正式RL rollout路径。
证据 `context-engine-updated-audio-npu7-v3.json`。

独立FP64概率复算见 `context-native-probability-gap-v1.json`：二元logprob最大差
0.02057/0.02285。60位探索分布的最大动作log-ratio为0.4444/0.5018，最小为
-0.5240/-0.5500；这些是遍历各位取极值得到的理论动作边界，并不代表高概率动作
或实际采样分布。说明单候选误差小不能直接保证联合PPO ratio一致。
下一定位应在同一真实输入/联合策略下比较backbone hidden与相同head读出，
区分语言/音频backbone执行、原生LoRA映射和head内核差异；不通过放宽阈值准入。
