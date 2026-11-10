# Omni NPU 真实响应概率验证准备

当前仍执行冻结基线、离线教师优化、native/Jev比较。没有派发权重训练。
CPU统一FP32专家候选在六条原始输入上的批间logits完全一致，但五条相对
原基座超过0.1 logprob阈值，最大0.892723，因此不进入默认actor。
见[完整诊断](omni-compact-verl-v1.md)。该CPU结果不能推导NPU相同失败，
也不能将该0.1诊断阈值当作正式on-policy重要性比率容差。

## 已实现：不读取标签的完整采样构批入口

`src/multimodal_jev/omni_probability_batch.py`新增`make_probability_batch`，
使用真实processor的prepared输入和完整K响应记录构造verl DataProto。
要求所有请求的所有决策均有完整、唯一的replicate；输入顺序保持调用方顺序，
不依据文本格式、结束原因、奖励或概率挑选样本。保留原始EOS与length截止
响应，不补写token，不调用奖励模块、不读取targets、不生成合成old logprob。

检查策略/指令/request/prompt/token哈希、token解码和原始概率长度，明确要求
`waveform_boundary='vllm_hop'`。prompt左填充、response右填充，完整audio
features与mask在最后一维补齐，保留每条的单/双音频顺序。超出上下文预算
直接拒绝，不截断。仅在response mask有效位置保存FP32 old logprob，原记录
保持不变；来源文件中JSON浮点数不由FP32表示替换。metadata标记diagnostic_only，
无奖励或优势字段，不能拿它代替reward-aware训练构批器。

该入口是受信运行器内部接口。调用者仍须独立认证采样容器/模型/原始结果、
完整learning分区和实际processor输出；声称`vllm_hop`的字符串不构成特征
等价证明。原排队采样的raw HF准备仅用于token检查，其vLLM音频输入未改；
后续actor使用已验证的hop补齐，并重新核对实际特征。

宿主测试34通过、1跳过（无verl）。固定镜像真实运行35通过、0跳过，
12.408秒，容器`43eec590d816c65a350756e4f2647a32946b992b37541e3414fb79abc4d9632d`，
源码`09cd5fa`，2 CPU/8 GiB、无网络或设备，退出0无OOM。外部核对123份
源码、实际容器配置及JUnit，见[独立测试核验](omni-probability-batch-tests-v1-verified.json)。
测试覆盖实际DataProto重排，合成单/双音频与长短响应，EOS/old logprob保留，
身份、覆盖、非有限值、FP32溢出及非法mask拒绝。这不是实际Omni前向证据。

## 目标NPU的内存约束

本地checkpoint全部safetensors头与index逐key匹配，Thinker共31,719,205,488
参数，BF16权重63,438,410,976字节，即59.0816 GiB（含视觉分支）。
原始64 GiB 910B3单卡剩余空间不能视为可用激活预算。当前HF实现先计算完整
序列lm_head，再由verl截取响应；B=2、T=4096时仅BF16 logits就需2.3203 GiB，
1024响应位置转FP32还需最多1.1602 GiB，尚未计音频/decoder激活、工作区与
log-softmax额外分配。静态清单见[内存核算](omni-thinker-npu-memory-inventory-v1.json)。
此次只读取safetensors头，未加载权重、未重新哈希全部shard或测量NPU峰值。

因此首次完整验证不能默认沿用小ASR的单卡`.to(npu:0)`配置。须先核查固定
镜像的分片加载和真实verl设备行为，再登记明确的加载方案、容量边界和实际
峰值内存采集。静态跨卡device_map、HF tensor parallel、FSDP各有不同语义，
不能将两卡分层推理冒称实际verl分布式actor或训练支持。

## 尚待执行

1. 等待原TRAIN78×K2采样及其独立验收；当前监督进程已重新确认PID/start/argv
   一致且活跃，仍等待原SFC native完整TEST。不得重采样或用greedy替代。
2. 真实记录进入新构批器，实际processor重新验证hop边界、全部156条响应、
   全部有效token、EOS/length终止、mask、原始old logprob和策略身份；不读取gold。
3. 核查固定镜像中NPU分片与算子支持，登记资源、启动配置及容量预检。NPU4–7
   保持既有冻结评测队列顺序，不能提前抢占；当前没有新NPU容器或新队列派发。
4. 用同权重、同完整音频+prompt+真实响应执行无梯度teacher-forcing，比较逐token
   logprob、log-ratio和重要性比率，并按响应/决策/音频数报告尾部误差。运行前
   固定容差与异常规则；现阶段尚未新增正式NPU概率验收门槛。
5. 完整actor微批与rollout一致性、参数更新/导出/恢复分别验证，不能由上述
   构批单元测试或CPU数值稳定性替代。权重训练继续受当前执行优先级约束。

teacher-forcing仅对已经采样的响应计算概率：仍由同一个多模态模型直接消费
音频与任务，不使用转写文本或独立文本模型。
