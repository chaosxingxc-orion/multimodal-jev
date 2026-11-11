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

随后完成[分片结构核查](omni-fsdp-layout-v1.md)：原完整配置未命中所检查的
AutoModel映射，默认wrap列表含不存在的旧解码层类而失败。显式Thinker与
真实层类在meta模型中形成108个单元，最大BF16 1.4310 GiB，已与checkpoint
头部计数独立核对。这只是结构准备，未验证NPU算子、FSDP初始化或完整前向。

完整采样响应的数值比较入口已实现，见[概率核验协议与CLI](omni-probability-audit-v1.md)。
它要求显式预登记协议，检查原JSON概率、全部有效token及所有运行两两差异；
整段比率仅作描述，不以正负误差抵消代替逐token验收。目前仅合成测试，
尚未登记正式NPU阈值或生成真实actor概率通过证据。

概率输出端现已通过 `omni_probability_capture.recompute` 衔接：预检完整原始
身份后只调用一次 actor，按有效 mask 导出原精度概率并绑定原记录哈希。
固定镜像实际 DataProto 重排及合成输出全流程测试 114 通过、0 跳过；
[独立核验](omni-probability-capture-tests-v1-verified.json)确认源码、容器和
JUnit。下一步仍是队列中的实际NPU层探针及完整真实采样概率验收，不能以
合成actor测试代替真实数值结论。

完整模型的后续加载还须遵循[位置编码精度约束](omni-buffer-precision-v1.md)：
显式BF16权重加载不能用全模型dtype cast舍入原始FP32 buffer。九份真实历史
prefix证据与未舍入RoPE精确一致；FSDP再转FP32不能修复已发生的数值变化。

[完整CPU加载与meta准备](omni-fsdp-loading-v1.md)已执行并独立核验，全部
checkpoint值和四个buffer精确一致。下一缺口是两rank实际FSDP初始化、
buffer同步后验收及NPU前向，不能将CPU预检结果作为分布式概率通过证据。

固定verl的[Ascend概率分派核查](verl-npu-logprob-source-v1.md)进一步确认：
原生路径调用NPU cross entropy，CPU集成使用的显式FP32适配不能代替其
数值验证。真实运行须记录实际分派/输入输出dtype，并用同份logits的FP32
参考区分概率归一化误差与模型前向差异；当前仅完成源码及哨兵分派验证。

## 完整采样完成后的真实输入预检入口

`tools/jev/preflight_omni_sampled_probability.py` 接收采样 root、独立验收文件及其
外部固定 SHA。先使用既有学习分区和 processor 资产检查，再通过
`omni_accepted_sampling.load_accepted_records` 重新核对验收与容器/源码标识、
配置、identity/result、全部156条记录及每次请求的 intent/raw/result。
该读取接口依赖协调者固定的独立验收，不能自行证明 Docker 状态或源码真实性；
原采样快照的独立验收器仍须先完成。

随后用实际本地 processor 以 `vllm_hop` 准备全部音频，构建真实 verl DataProto，
检查完整响应 mask、token 和 old logprob；保存原始/补齐音频长度、特征张量哈希、
构批张量哈希及两rank微批1/2的完整分片计划。输出目录必须全新。
不加载权重、不读取 targets、不启动模型前向或通信，不补造未完成采样。

```bash
PYTHONPATH=src python tools/jev/preflight_omni_sampled_probability.py \
  --root "$SAMPLING_ROOT" \
  --acceptance "$SAMPLING_ACCEPTANCE" \
  --acceptance-hash "$SAMPLING_ACCEPTANCE_SHA256" \
  --output "$PROBABILITY_PREFLIGHT_OUTPUT"
```

此命令须在固定镜像运行，当前尚未执行真实数据路径；既有采样尚待完成。
新增读取接口与原始验收器的40项合成测试通过，覆盖验收、配置、模型策略身份、
原始概率、合并记录、输入名单和终态变化的拒绝。尚未验证新CLI的真实
processor/DataProto运行。预检成功也不等于vLLM特征比较、NPU数值通过或
正式容差登记；真实概率运行器仍须接入这些步骤及完整FSDP加载与前向。

随后在固定镜像、2 CPU/8 GiB、无网络/设备容器内完成40项测试，零跳过，
并成功执行新CLI的 `--help` 导入与参数入口。外部核验390份源码与Git
`9304490`、实际命令/挂载/资源、JUnit及退出0无OOM，见
[固定镜像记录](omni-accepted-sampling-tests-v1-verified.json)。本次仍是合成
采样产物测试和CLI入口检查，未调用真实processor或构造实际采样DataProto。

### 补入实际 vLLM processor 比较（尚未运行真实数据）

上述入口现已接入实际 vLLM processor：在认证完整采样后，从原始音频和
相同提示重新准备 vLLM 输入，与本次 `prepare(vllm_hop)` 结果逐决策比较。
检查完整 token、每段音频的有效帧数与顺序，以及 HF 有效帧拼接后转为
BF16 的特征；沿用既有 TRAIN78 特征比较的 `atol=rtol=1e-5`，不将其
作为概率容差。结果保存为 `processor-parity.json`，其哈希和两侧实际
processor 源码哈希写入预检结果。不会加载模型或派发 NPU 工作。

比较函数位于 `omni_processor_parity.py`。9 项宿主测试通过，覆盖双音频
不同帧长、忽略 padding、BF16 舍入，以及 token、帧顺序、特征顺序、截断、
dtype、非有限值和非法 mask/boundary 的拒绝；CLI 编译及差异检查通过。
这批测试使用合成张量。此前固定镜像的40项测试不覆盖此次新增入口；
本修订尚未执行固定镜像测试或真实采样预检，不新增实际特征通过结论。

随后在固定镜像验证本修订：源码 `451e506`，49 项测试通过、零跳过，
CLI `--help` 成功；独立核对392份源码、实际命令/挂载及2 CPU/8 GiB、
无网络/设备配置，容器退出0、无OOM或重启。见
[固定镜像核验](omni-processor-parity-tests-v1-verified.json)。
此次仍未执行真实processor/采样数据分支，实际预检须等待完整采样验收。
