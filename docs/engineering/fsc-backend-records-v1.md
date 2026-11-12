# FSC后端概率到迁移评分的接口

2026-11-12 UTC。已准备固定RL起点的真实FSC后端检查包：`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-real-span-step0-policy-prepared-v1`。复用已通过SLURP全候选检查的HF/vLLM执行工具、固定镜像和`qwen3-asr-bf16-hf-order-short-audio-v2`配置；12条TRAIN音频×direct/structured×native/permutation/rebind，共72请求。模型/adapter、CPU输入、源码与容器挂载均固定并核对哈希。准备时重新核验两个前序容器实际退出0。该包尚未派发，D12连续实验仍优先，不读取目标标签，不计算迁移指标。

新增`fsc_backend_records.normalize_gate`，将实际HF/vLLM输出归一为`fsc_transfer_metrics.score_policy`的输入。独立核对六视图全量名单、配置重编译、音频文件身份、处理回执、CPU特征摘要、Span位置、候选token顺序与语义映射、策略身份及请求摘要。重新计算全部候选的归一化和后端概率误差；阈值沿用logprob误差0.02与所有独立Span联合比值[0.98,1.02]，不信任产物中的parity通过标记。另报告argmax变化，容差内并不保证argmax相同。

31项接口、配对评分和配置测试通过。测试覆盖缺行/重复、策略错配、候选重新排序、非法概率、音频或特征错配、请求摘要错误，以及各Span误差均小于0.02但联合比值越界的情况。测试夹具是合成概率和输入身份；不是实际NPU结果。

该模块只验证产物内容，不负责官方划分准入、实际容器终态、文件哈希认证或源域checkpoint选择。外层正式评分工具仍须执行这些检查。已有准备包的源码先于此新评分接口冻结；不原位修改，它只负责无标签后端推理，后续可用独立版本的评分程序读取其产物。

本轮04:45:27 UTC核验：step2资源等待控制器PID3547966及D12控制器PID3548079的`/proc`命令行与启动记录一致，均存活。资源等待最新记录仍报告容器占用，step2 suite未启动。未重启训练或越过资源检查；下一次交互式状态轮询不早于04:55:27 UTC。

另将真实CPU产物的72条输入身份与合成均匀后端概率连接，完整通过接口校验，确认实际产物字段可消费，记录于`fsc-backend-records-schema-v1.json`。这项检查没有加载模型或读取目标标签，不能作为后端一致性结果。
