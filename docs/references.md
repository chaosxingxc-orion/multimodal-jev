# 参考工程与初始化边界

本次初始化参考同级 workspace 中的 `nga/` 和 `omni/`，只借鉴工程组织和研究约定。
没有复制源代码、历史实验结果、模型、数据、运行状态或主机配置，也没有将参考仓库作为运行依赖。

## NGA

- 远程：<https://github.com/chaosxingxc-orion/nga>
- 参考文件：`README.md`、`AGENTS.md`、`.gitignore`。
- 参考时本地 HEAD：`61f244ff9ac0409b6a2a59e4563d4e4e6afdef72`。
- 借鉴：类型化决策研究、一致性与区分性配对评估、实验预注册、计划/协议/记录分层。

## Omni

- 远程：<https://github.com/chaosxingxc-orion/omni>
- workspace 下的 `omni/` 是工作区集合，顶层没有 Git 仓库和 README。
- 顶层参考文件：`omni/AGENTS.md`；工程说明参考 `omni/repro/publication/omni/README.md` 和 `.gitignore`。
- 该 publication checkout 的参考时本地 HEAD：`88c56350d8f14265af61f1b9c6cc795821100614`。
- 借鉴：多模态证据、外置模型与数据、来源记录、可恢复运行和明确的终态证据。

以上 HEAD 记录本地参考位置，不宣称对应远程最新状态；顶层 `omni/AGENTS.md` 不由该 HEAD 固定。
两个项目的既有实验结论均不构成本项目的验证结果。

## 已明确的课题方向

用户明确本课题为语音多模态 Jev 的设计与实现；主要模型参考为 Qwen 系列最新开源 ASR，
任务与数据参考现存 SLU 和 Agentic Speech。Omni 工程中的 Qwen3-Omni 基线仅提供工程参考，
不自动成为本课题的 ASR 模型选择。具体范围见 [课题说明](../proposal/00-research-scope.md)。

## 后续待确定

1. 首个 SLU / Agentic Speech 任务、语音与上下文输入契约和类型化输出语义。
2. 官方核验的最新 Qwen 开源 ASR 具体版本及读出方式，训练与推理依赖版本。
3. 数据许可、来源、划分、模态对齐和缺失证据协议。
4. 计算环境、评测指标及首个可复现实验。

新工程尚未选择发布许可证；后续代码复用或发布时明确许可范围。
