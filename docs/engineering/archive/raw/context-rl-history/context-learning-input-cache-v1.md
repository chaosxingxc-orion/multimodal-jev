# 全量学习输入预构建

2026-11-14，目标是把固定学习人口的冷准备成本移到NPU训练启动之前，同时让后续
epoch和实验复用相同的纯输入缓存。仅缓存输入，不缓存策略、监督、采样或奖励。

## 真实依据与代码变更

全日程前两步的后台输入服务63.74/57.23秒，第二步模型更新15.61秒但前台仍等待
27.17秒，见`context-full-schedule-prefix-npu-v1.json`。预取不能隐藏大于计算耗时
的全部输入服务；本地镜像与缓存预构建仍需完成实际吞吐验收。

审查发现早期`probe_cached_context_inputs_cpu.py`的processor身份取值与训练入口
不同（bundle不返回files，走了整个MANIFEST文件hash的fallback）。这不否定旧CPU探针
逐张量一致性，但旧探针缓存不能宣称可被训练直接命中。
现在统一为`cached_context_preparation.processor_cache_identity`：验证processor资产hash，
再用训练原有`content_hash(processor文件hash映射)`算法计算身份；与权重更新无关。
不更改已运行远程快照，原训练的processor身份值保持不变。

`LocalAudioMirror(...,read_only=True)`支持只读绑定的本地副本，只允许本地文件系统，
逐条读取核验内容，禁止发布或向共享源回退。请求中的原音频路径不变，因此缓存键、
标签绑定和输入身份不因物理副本路径变化而改变。

`tools/jev/prepare_context_learning_cache.py`按完整40513学习人口预构建：

- 固定release、完整mirror manifest、processor资产和独立processor预检ledger；
- 4个CPU worker、torch单线程、最多8个待处理项，无NPU设备；
- 消费全部60个候选定义、同一marker和4096预算，禁止截断；
- 每条核对预检的request、token张量、60marker位置、音频处理receipt及特征形状；
- BF16特征和其他输入写入本地cache，输出逐条key、文件hash和张量hash；
- 全人口完成才发布MANIFEST，不把部分输入标为完整缓存。

注意旧预检特征为FP32，本次缓存为BF16，故这里只对旧特征形状比较；不能将它描述为
全部BF16特征已有独立逐张量重算。完整缓存终态还需独立检查文件绑定/张量hash，并在
实际NPU路径验证命中和吞吐。

## 当前执行链

镜像容器b4d97c6d保留原handle、准确wait70996。自动驱动PID281383：
`continue_context_learning_inputs_v1.py`准确wait同一容器→独立镜像collector→
`launch_context_learning_cache_v1.py`启动固定镜像CPU预构建并准确wait。
任一失败停止保留句柄；完整缓存终态仍需独立收集，不自动派发正式训练。
状态`.local/decision-objective-research/context-learning-inputs-v1-status.json`，日志同名前缀
`-driver.log`。本轮记录时仍等待镜像，缓存未启动。

15项本地缓存/镜像/批次测试通过，覆盖物理源移除后的本地复用、损坏拒绝、只读镜像
拒绝写入/共享挂载、原请求身份保留、缓存身份不随模型权重更新变化。CPU组合与NPU
稳定性能仍未完成，不把测试通过当吞吐收益。

## Token源码检查

固定镜像transformers Qwen3-ASR源码hash与timestamp引用已保存到
`.local/decision-objective-research/context-timestamp-source-review-v1.json`。
配置定义timestamp_token_id=151705；processor.decode_forced_alignment按这个ID选择
时间戳位置，普通Qwen3ASRModel骨干前向未引用该字段。当前Jev marker为151706，
151705为诊断占位guard，原embedding行不改。该检查不等同完整生产token准入，
还需冻结guard的非输入边界及请求版本（旧policy通用版本与实际exclusive_event版本不同）。
不得仅把test_only改成admitted而宣称验收完成。

旧fullaudit98629d12在远程12:30:35 UTC仍running/nonOOM，累计130分钟；原driver保留，
selection尚未产生。本轮未重启该任务。

## 后续收集入口

`collect_context_learning_cache_v1.py`已准备：终态后以只读容器逐cache文件复算hash、
key、tensor hashes、dtype/marker维度和完整输入绑定，流式回传收据；本地重新加载完整
官方clean learning人口并构造请求核对。不是再次processor重算，不冒称波形到特征
独立复算。验收输出`docs/engineering/context-learning-cache-v1.json`只在全部通过后写入。

镜像到期观察observed_unix1794659820确认原b4d97c6d仍running/nonOOM，已完成7936/40513，
记录耗时634.23秒，无terminal；驱动继续准确等待。缓存尚未启动，没有新完成声明。
