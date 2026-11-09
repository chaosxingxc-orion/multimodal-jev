# multimodal-jev

[GitHub](https://github.com/chaosxingxc-orion/multimodal-jev)

围绕**语音多模态 Jev 的设计与实现**开展研究与工程工作。主要参考模型为 **Qwen 系列最新开源 ASR 模型**，
参考数据集和任务为现存的 **SLU（Spoken Language Understanding）** 与 **Agentic Speech** 任务。
参考 [nga](https://github.com/chaosxingxc-orion/nga)
的类型化决策读出、语义一致性评估方法，以及 [omni](https://github.com/chaosxingxc-orion/omni)
的多模态证据处理与可复现实验组织方式。

当前阶段：已完成资源调研与共享下载，正在构建 NGA/verl/vLLM 兼容镜像、Qwen3-ASR 推理探针、FSC Jev 数据契约和奖励接口；
尚未完成学界基线复现、NPU 训练或 RL 效果验证。详细结论见 [研究报告](docs/research/README.md)，下载与整理状态见 [本地资源](docs/research/06-local-assets.md)。
研究范围与初步工作项见 [课题说明](proposal/00-research-scope.md)。

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
加速器、深度学习框架和多模态模型依赖在确定基线后单独固定版本。

## 研究与实验约定

- 在运行前记录假设、数据划分、指标、阈值和失败判据；测试集不参与调优。
- 一致性目标须配套相关事实变化的区分测试，避免常量预测等退化解。
- 区分类型化读出、生成式标签评分和外围规则处理，不将设计目标写成已验证能力。
- 模态缺失、冲突证据及未知状态的处理应在实验协议中明确。
- 模型、数据、特征、checkpoint 和完整日志存放在 Git 之外；仓库保留清单、版本和结果摘要。
- 长任务应可脱离会话运行、可恢复，保存启动命令、进度、成本和带退出码的完成标记。

来源与初始化边界见 [参考说明](docs/references.md)，协作规则见 [AGENTS.md](AGENTS.md)。
本地 `wiki/` 是仓库内文档，尚未配置独立 GitHub Wiki 同步。

三阶段目标、前瞻实验设计及当前限制见 [RL 研究协议](proposal/01-rl-research-protocol.md)；
镜像兼容性见 [运行时说明](docs/engineering/runtime.md)，数据与奖励入口见 [数据契约](docs/engineering/data-contract.md)。

实现进度：已加入 [冻结声学头开发实验](experiments/fsc-frozen-audio-pilot.md)、
[音频策略与 verl 接入审计](docs/engineering/audio-rl-integration.md)，以及成对正确性奖励。
这些是基线/RL 研究组件；正式学界复现、跨任务泛化和 RL 提升仍需实验验证。

跨任务数据已扩展到 [SLURP 域留出协议](experiments/slurp-transfer-v1.md)：
59,014 条真实录音、413,098 条派生决策，覆盖 scenario Choice 和四类开放槽位；
已完成全量数据校验，尚未进行该发布上的模型评估。

[vLLM rollout 与 verl 更新入口](docs/engineering/rollout-training.md) 已实现；
实际 NPU engine rollout 和完整 RL 更新尚待验证。原始 ASR 模型的实际采样暴露了
结构化输出失败，已据此启动 SFT 初始化和完整 FSC 冻结基线。
