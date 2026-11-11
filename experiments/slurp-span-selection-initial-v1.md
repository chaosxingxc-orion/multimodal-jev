# 固定初始SFT的独立来源selection诊断

模型固定Qwen3-ASR-0.6B、初始SFT adapter
`7341430053a9b4f36f5d5b8fe80616889af1857c71187d22b4895b062f557140`，
BF16未合并LoRA，HF eager，一次音频+问题模板前向，原始音频不做增强。
物理NPU4，固定现有镜像，无模型下载、无优化器、无生成式文本解码。

在读取本轮selection预测前固定全部2304请求，覆盖44独立来源、三个问题arm、
九个规则对及其显式否定。没有为更好分数或更短序列筛选样本。参考语义审核
屏蔽的两个决策及其依赖规则已经在数据发布中固定，不能按预测再删行。

报告Final accuracy、逐规则balanced accuracy再宏平均、全Span accuracy、
预测事实+确定性执行器、三问相对单问的配对胜负、互补规则同时正确比例。
每arm768请求，Final正负各384；不伪称2304个独立音频，不当作官方TEST分布。
selection与声明learning来源组/音频哈希隔离，未证明说话人/语义近重复隔离，
审核没有人工听音。这里不读取官方TEST，不以selection反馈优化权重或提示。

入口要求显式`selection-full-v1`和固定协议；默认历史入口仍只返回原216个
learning探针。原始manifest保留禁止训练/未自动评测准入标记，单独protocol
仅授权固定名单的零更新推理，不能用于奖励或任意策略筛选。

首轮只有共同初始SFT策略，不根据四分支learning分数选胜者。后续如比较全部
训练策略，应在查看它们的selection分数前固定完整名单和比较方式。
该HF推理完成不自动证明selection输入的vLLM一致性；若用于跨后端比较，
仍需相同输入/策略下完整概率验收。当前运行证据见
`docs/engineering/span-selection-initial-sft-v1-launch.json`。
