# multimodal-jev

[GitHub](https://github.com/chaosxingxc-orion/multimodal-jev)

围绕**语音多模态 Jev 的设计与实现**开展研究与工程工作。主要参考模型为 **Qwen 系列最新开源 ASR 模型**，
参考数据集和任务为现存的 **SLU（Spoken Language Understanding）** 与 **Agentic Speech** 任务。
参考 [nga](https://github.com/chaosxingxc-orion/nga)
的类型化决策读出、语义一致性评估方法，以及 [omni](https://github.com/chaosxingxc-orion/omni)
的多模态证据处理与可复现实验组织方式。

当前主线使用本地 **Qwen3-ASR-0.6B-hf**，直接消费音频和完整问题模板，
一次前向填充多个 typed Span，不进行自回归文本解码。优先使用清理并审核的 SLURP 学习数据，
已确认错误标签不进入训练、奖励或模型选择。Omni 冻结基线保留为历史证据。
详细结论见 [研究报告](docs/research/README.md)，本地资产见 [本地资源](docs/research/06-local-assets.md)。
研究范围与初步工作项见 [课题说明](proposal/00-research-scope.md)。

当前工程优先级：**BF16 LoRA 训练与重载 → actor/vLLM 概率一致性 → verl 单步更新与新版本复核**。
BF16 candidate-CE 已完成 20 次均衡更新；48 条学习录音在独立 HF 重载后候选概率误差为零。
跨后端检查已在一条真实录音通过，100 个分层激活点逐位一致；48 条顺序请求也全部通过，最大候选 log-prob 误差 1.19×10⁻⁷。
一次真实语音 verl GRPO/PPO LoRA 更新已完整通过：独立HF重载误差0，更新后两条vLLM概率检查通过。
新adapter的48条全量复核进行中。
分布式训练和跨任务泛化尚未验收。
运行配置和准入门槛见 [BF16 训推一致性](docs/engineering/verl-vllm-lora-span-alignment-v1.md)，
生命周期入口见 [单步训练验收](experiments/span-actor-update-v1.md)。
早期[冻结基线与 TFRL 协议](proposal/02-frozen-baselines-tfrl-jev.md)及下文旧实验用于追溯，不代表当前训练安排。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/multimodal_jev/` | 可安装的 Python 包，后续存放可复用实现 |
| `configs/` | 可版本化配置；不包含凭据和机器私有设置 |
| `docs/` | 设计说明、参考来源与工程决策 |
| `proposal/` | 研究问题、假设和阶段计划 |
| `experiments/` | 实验协议、数据清单和实验入口 |
| `tests/` | 后续实现对应的自动化测试 |
| `tools/` | 数据准备、运行和结果检查工具 |
| `wiki/` | 随代码版本管理的实验记录与索引 |

## 本地开发

需要 Python 3.11 或更新版本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -c "import multimodal_jev; print(multimodal_jev.__version__)"
```

当前包没有运行时第三方依赖；`dev` 提供 pytest，`assets` 提供下载工具所需 requests。
使用 `.venv/bin/python -m pip install -e '.[dev,assets]'` 安装工具依赖，`.venv/bin/python -m pytest` 验证共享资源工具的校验与不覆盖行为。
加速器、深度学习框架和多模态模型依赖固定在独立实验镜像中，见运行时说明。

## 研究与实验约定

- 在运行前记录假设、数据划分、指标、阈值和失败判据；测试集不参与调优。
- 一致性目标须配套相关事实变化的区分测试，避免常量预测等退化解。
- 区分类型化读出、生成式标签评分和外围规则处理，不将设计目标写成已验证能力。
- 模态缺失、冲突证据及未知状态的处理应在实验协议中明确。
- 模型、数据、特征、checkpoint 和完整日志存放在 Git 之外；仓库保留清单、版本和结果摘要。
- 长任务应可脱离会话运行、可恢复，保存启动命令、进度、成本和带退出码的完成标记。

来源与初始化边界见 [参考说明](docs/references.md)，协作规则见 [AGENTS.md](AGENTS.md)。
本地 `wiki/` 是仓库内文档，尚未配置独立 GitHub Wiki 同步。

长期算法问题、受控对照与泛化判据见 [Omni RL 设计](proposal/03-omni-sfc-rl-design-v1.md)；
镜像兼容性见 [运行时说明](docs/engineering/runtime.md)，数据与奖励入口见 [数据契约](docs/engineering/data-contract.md)。

工程证据入口包括[完整CPU数值修复候选](docs/engineering/omni-compact-keys-diagnostic-v1.md)、
[真实verl冻结actor比较](docs/engineering/omni-compact-verl-v1.md)和
[Audio2Tool原生/Jev解码成本测量](docs/engineering/audio2tool-decode-profile-v1.md)。
六条CPU候选响应logits与参考一致不证明NPU概率一致、训练收益或任务泛化。

## 历史实验记录

以下保留早期ASR、SFT与转写对照记录供追溯，所述“已启动”“等待执行”均为当时
状态，不是当前任务安排。串联文本模型和参考转写替换不再派发；旧SFT/RL配置不能
直接用于当前Omni actor。原始设计见[历史RL协议](proposal/01-rl-research-protocol.md)。

实现进度：已加入 [冻结声学头开发实验](experiments/fsc-frozen-audio-pilot.md)、
[音频策略与 verl 接入审计](docs/engineering/audio-rl-integration.md)，以及成对正确性奖励。
这些是基线/RL 研究组件；正式学界复现、跨任务泛化和 RL 提升仍需实验验证。

跨任务数据已扩展到 [SLURP 域留出协议](experiments/slurp-transfer-v1.md)：
59,014 条真实录音、413,098 条派生决策，覆盖 scenario Choice 和四类开放槽位；
已完成全量数据校验和固定开发子集的原始模型、SFT 迁移评测。
SFT 输出格式全部有效，但非空目标正确数为 0/81，尚未显示语义迁移收益；
见 [SLURP 结果](docs/engineering/slurp-sft-development-v1-result.json)。

[vLLM rollout 与 verl 更新入口](docs/engineering/rollout-training.md) 已实现；
实际 NPU engine rollout 和完整 RL 更新尚待验证。原始 ASR 模型的实际采样暴露了
结构化输出失败，因此先进行 SFT 初始化，并运行完整 FSC 冻结基线作为对照。

SFT 初始化现已完成 47 次更新并通过合并/重载数值校验，见
[完成收据](docs/engineering/fsc-sft-init-v1-receipt.json)。FSC/SLURP 开发集评测已启动，
完整 FSC 冻结基线仍在运行；尚无 SFT 泛化或 RL 提升结论。
[SFT 起点的两轮 RL 配置](configs/experiments/rl-schedule-sft-smoke.json)
已通过真实资产预检，仍待分配空闲 NPU 执行 vLLM 采样与 verl 更新。

[转写文本对照](experiments/text-evidence-baselines.md)已在 FSC、SLURP 固定开发子集启动：
使用同一个 Qwen3-0.6B，比较 ASR 转写与参考转写对决策的影响。
四组均已产出预测，完成后自动生成成对说话人 bootstrap 比较；当前无完整文本对照结果。
实验状态与证据索引见 [运行记录](wiki/2026-11-10-development-baselines.md)。
