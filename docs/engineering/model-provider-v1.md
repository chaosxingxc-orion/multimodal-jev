# 用配置选择 Jev 语音模型

范围：当前单阶段 Joint TPO 的初始策略、TRAIN 输入、actor、固定 anchor、逐 macro 刷新的 reference、Noul 和 Choice 端点评测，以及固定策略的 vLLM selection 入口。以前封包的源码、历史对照和已排队任务继续使用原先的实现。

**切换已适配的模型时，实验方只改配置。** 将 [模型模板](../../configs/portable/models/jev-audio-model-v1.json) 的字段放进实验 runtime 的父配置；权重路径、内容 ID、Python 模块名、模块文件 SHA256 和 options 从本地宏读取。可以把整个绑定对象作为同一个 JSON 宏传给初始策略和缓存配置。包内的 actor、anchor、reference 和端点评测均从训练父配置取实现，不各自猜模型名。旧配置没有这个字段时保留原 Qwen3-ASR；显式传入空值、错误哈希、残缺实现或冲突的策略 profile 会报错。

例如，使用仓库自带实现时，模块是 `multimodal_jev.qwen3_asr_provider`，options 为 `{}`，对选定 checkout 的 `src/multimodal_jev/qwen3_asr_provider.py` 运行 `sha256sum` 得到模块摘要。换成合作方模块时使用其真实导入名、摘要和选项；模块及其依赖必须随源码快照交付，或已经装入固定摘要的镜像。插件是团队审核的可执行 Python 代码。权重、处理器、初始策略和目标数据分别按原协议绑定，模块摘要不能替代它们。

**第一次接入新架构时**，实现一个声明 `ABI = "jev-audio-model-v1"` 的模块，导出 `create(options)`。不需要新增训练器、reward、调度器或评分器。接口见 [解析器](../../src/multimodal_jev/model_provider.py)，现有适配示例见 [Qwen3-ASR](../../src/multimodal_jev/qwen3_asr_provider.py)。

| 适配器方法 | 实现方负责的边界 |
|---|---|
| `runtime_profile`、`initialize_policy(config, output)` | 分配 marker、从冻结 base 初始化 LoRA 和一层 Jev head，按共享策略格式输出可重载的权重、处理器和初始化收据。外部模型用自己的 `provider:<name>-vN` profile。策略内记录架构实际的 LoRA 模块名、shape、rank 和缩放。 |
| `processor(path)`、`encode(processor, request, audio, **encoding)` | 直接处理音频和任务定义，返回共享 CPU 输入协议及升序 marker 位置。把模型特有的音频掩码映射到公共字段。编码不接收监督标签。 |
| `collation(base, processor, model)`、`choice_compiler(processor, collate, limit)` | 给出 marker 和音频 token ID、组批参数，以及带 apply/render 接口的变长候选编译器。保留录音、问题顺序和 NONE 的位置。 |
| `initialize_actor_runtime()`、`adapter_transition(config, fresh)`、`load_actor(config, device, output, execution, transition)` | 初始化该模型的 veRL 环境并装载实际策略；按现有接口返回 base、processor、Jev 模型、FSDP 包装、优化器、执行信息和 runtime handle。固定 BF16，训练的只有 LoRA 和 Jev head。 |
| `load_reference(config, bundle, output)`、`forward(model, items, *, device, dtype, policy_id, **collation)` | 装载同一策略，返回 model、device、base、handle；forward 从无标签输入返回有序的原始 logits，shape 为 [录音数, 判决数]。actor 上保留梯度；reference、评测关闭梯度。公共代码计算 TPO 目标、sigmoid 和 Choice 分布。 |
| 可选 `load_vllm(config, bundle)` | 固定策略 vLLM 评测使用此方法，返回支持 score/close 的常驻引擎。适配器负责该引擎的架构注册、LoRA 映射、marker 和 head 的完整加载。缺此方法，该入口拒绝运行；不同架构不能借用 Qwen 的 vLLM 权重映射。Joint TPO 当前 TP1 reference 是 HF 前向，两种引擎须分别验收。 |

模型输入缓存键包括 provider 代码和 options 的摘要、处理器资产、音频和编码规约。显式绑定即建立新的缓存命名空间。原始音频、标签和划分仍共享。对新架构用同一绑定跑现有 CPU 预检与 TRAIN 缓存生成；单阶段封包的 `input_identity_publication` 指向该缓存的 MANIFEST/inputs.jsonl，SHA256 取新 manifest。只从输入收据构造训练请求身份，不需要该模型以往的预测。测试集缓存另行生成、验证，再作为评测包输入。未携带绑定的旧缓存不得用于显式绑定的实验。

**准入标准**：固定权重、LoRA、head、marker、processor 和输入；先验证同一录音的 singleton 与批处理、变长候选、初始及更新后的 reference，以及保存重载后的概率与决策。使用实验冻结的精度阈值和原有有限验收，另测实际 NPU/引擎。CPU 接口测试不能证明新架构可在 181 多卡训练，也不能证明 vLLM 数值或吞吐。纯文本模型即使支持相同 Python 接口，也不符合端到端音频任务。
