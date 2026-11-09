# 2026-11-10 开发集基线运行记录

北京时间约 05:26 的观察记录，不是实时状态。共享实验根目录为
`/mnt/cephfs/xingchao/experiments/multimodal-jev/`。
退出状态以对应 Docker ID、terminal.json 和完整产物校验为准。

| 项目 | 已验证进展 | 证据／协议 |
| --- | --- | --- |
| FSC 数据 | 30,043 个 WAV，官方 train/valid/test 为 23,132/3,118/3,793 | `/mnt/cephfs/xingchao/datasets/fsc` |
| 音频 SFT 初始化 | 47 次更新完成；合并模型重载误差通过预设阈值 | [收据](../docs/engineering/fsc-sft-init-v1-receipt.json) |
| 原始音频策略 | FSC 186、SLURP 168 个开发决策完成；全部输出格式无效，严格准确率为 0 | [预训练基线收据](../docs/engineering/pretrained-jev-baseline-v1-receipt.json) |
| ASR 转写 | FSC WER 14.56%，SLURP WER 17.90%；仅固定开发子集 | [ASR 收据](../docs/engineering/asr-baseline-v1-receipt.json) |
| SFT 开发评测 | FSC 62/186、SLURP 145/168；Docker 均 running | `post-sft-v1/evaluation`、`slurp-transfer-eval-sft-v1/run` |
| 完整 FSC 冻结基线 | 特征阶段 16,823/26,250；Docker running | `fsc-frozen-audio-full-v1/run` |
| 文本决策对照 | 四组已启动并实际产出预测；尚未完成 | [协议](../experiments/text-evidence-baselines.md)、[运行收据](../docs/engineering/text-baseline-v1-launch-receipt.json) |
| SFT 起点 RL | 真实资产预检通过；尚未执行 NPU engine rollout／完整更新 | [预检收据](../docs/engineering/rl-sft-preflight-v2-receipt.json) |

文本对照使用固定 Qwen3-0.6B，不进行模型训练。FSC 每组 186 个决策、
SLURP 每组 168 个决策，共 708 个计划输出；重复视图不计为独立录音。
两种证据分别为缓存的 Qwen3-ASR 转写和原始参考转写；后者不是语音模型成绩。
四个目录均为 `text-{fsc,slurp}-{asr-text,reference-text}-dev-v1/run`。

自动汇总在 `text-evidence-comparison-v1/run/{fsc,slurp}` 等待上述精确 Docker ID
成功退出后运行。固定参考减 ASR 的比较方向、2,000 次说话人聚类 bootstrap、
随机种子 17；校验相同模型权重、代码中的提示构造和解码参数。
模型、提示或解码差异会使汇总失败，不会被当作纯转写证据差异。
SFT 对比的独立汇总位于 `sft-development-comparison-v1/run/{fsc,slurp}`。

截至本次观察，两台 NPU 的全部设备仍由既有 vLLM 任务占用，每卡约
61.7–61.8 GB / 65.5 GB。上述新增文本任务仅使用 CPU，每组 4 核、12 GiB 上限。
AudioJev 官方复现还需独立 CUDA 环境与完整 MMAR 推理。
当前证据不支持 RL 提升、正式跨任务泛化或学界复现完成的结论。

## SLURP 完成更新

SFT 评测容器现已 exit 0，168/168 个输出完成，统一 scorer 重算和
2,000 次说话人 bootstrap 完成，原始产物哈希再次验证通过。
格式有效数从 0/168 提升至 168/168；实际非空输出 25 个，全部错误。
三个 Choice 视图各 0/24，非空槽位共 0/9；非空目标总体 0/81。
71 个正确结果全部来自空槽位，低于全 null 控制的 87/168。
置换和重绑定成对正确率均为 0/24。因此这次仅见格式学习，未见有效语义迁移。
生成耗时合计 2,505.23 秒（四核 CPU；不含加载）；结果与分层区间见
[完成收据](../docs/engineering/slurp-sft-development-v1-result.json)。
FSC 和文本对照仍单独运行，不能把 SLURP 完成推定为全部实验完成。

## 真实 vLLM 与 actor 数值校验

官方 ARM CPU vLLM 0.30 已完成波形采样（6 个 TRAIN 视图、12 个响应）。
NGA 镜像内的 actor 校验随后失败：最大 log-prob 误差 0.16856，大于
固定门槛 0.02，因此没有执行更新。完整 vLLM 波形预处理与 HF 相差一个
特征帧；显式传入 HF 特征的处理器检查已达到 token 完全一致、特征误差 0。
第一次特征引擎运行又被 enable_mm_embeds 入口检查拒绝，现已修复并以
vllm-cpu-rollout-sft-v3 独立启动；通过处理器检查不能代替 actor 数值校验。
见 [协议与失败证据](../experiments/vllm-cpu-integration.md)。

本次观察中 FSC SFT 开发评测到 135/186，四组文本评测仍在实际运行。
波形概率独立审计已完成的首个响应在相同合并权重的 HF 单条推理下误差
为 0.10925，说明误差也存在于不加载 LoRA 的诊断路径；完整审计尚未完成，
还不能归因于某个单一因素。没有调整 0.02 门槛，也没有 RL 提升结论。
