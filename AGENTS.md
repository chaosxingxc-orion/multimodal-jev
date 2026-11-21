# Repository Guidelines

默认中文交流与报告，代码标识符使用英文。此文件只维护跨实验都必须遵守的工程规约。

## 维护入口

- 开展新实验，或调整模型、数据版本、目标、预算、设备布局和评测口径时，先在 [experiments/](experiments/README.md) 登记或修订协议与版本化配置入口。
- 领取实验、通过验收、遇到阻塞、结束训练或完成评测时，更新 [Wiki 台账](wiki/experiment-ledger.md) 同一行。进展、失败原因、成本和结论写入 [Wiki 报告](wiki/reports/README.md)，经核验后同步主线。
- 实验包按 **读取一行台账 → 构建或复用数据 → 训练 → 评测 → 回填台账** 交付。修复和续跑沿用研究 ID，另记包版本和运行句柄。[五步规约](wiki/experiment-package.md)

## 工程与执行

- 复用共享基座、固定脚本与版本化配置。模型、并行、目标和评测通过组件接口接入。机器、路径、设备和镜像从本地宏绑定。运行器禁止硬编码实验参数或动态改写训练源码。[组件规约](docs/engineering/component-base-v1.md) · [本地宏](docs/engineering/local-config-macros-v1.md)
- 先冻结协议和配置，通过有限验收后再正式派发。运行中、排队中或待审核的包及依赖保持不可变。修复另发版本，跨机复核内容身份，再绑定资源和发布句柄。
- 用固定脚本做采样、收集、评分和独立审核。优先复用内容身份和审核证据均匹配的产物。按完整分片续跑，避免重复派发。并发与预取由配置限定，I/O 用有界线程池，Python 密集计算用有界进程池。终态事件只调用获准的确定性程序。
- 派发前登记依赖、用卡和可并行工作，CPU 准备与审核放在租约外。核对准确句柄后执行和释放。长任务保存心跳、断点和终态，实时状态用现场进程或容器核验。CPU 验证、NPU 执行、正常退出和独立审核分别记录。[操作规约](docs/engineering/context-prob-v1-baseline.md#日常任务执行规约)
- 可复用实现放在 `src/multimodal_jev/`，工具放在 `tools/`。使用 Python 3.11+ 和项目内 `.venv`，按变更验证。复用外部代码或模型前核对来源、版本和许可。推送前检查差异与远程更新。

## 研究与证据

- Jev 直接消费音频与任务定义，输出候选判决。主实验禁止用转写替代音频。执行参数由 Harness 或 System2 处理，禁止计入 Jev 的监督和验收。
- 只用审核通过的监督。训练、难例挖掘、校准和教师反馈只用获准的源域 TRAIN。selection 只用于选择，TEST 和迁移集独立封存。保留原始评测人口和标签版本。更改人口时单独报告。[SLURP 准入](docs/engineering/slurp-quality-hold-v1.md)
- 预登记假设、对照、检查点、主指标和退化判据。评测独立复算，逐数据集报告分母、对照、变化、缺失项及结论。跨任务需要独立监督及新判决证据。只凭完整证据确认完成。

## 存储与协作

- 热状态放本地持久盘并保留同步语义。受影响的共享盘可直接读取已发布资产，运行目录、热写及文件或目录同步走本地盘。提交只包含可复现所需的代码、配置及证据索引。LoRA 不重复保存主模型。清理前列清单，保护活跃依赖和最小复现包。[存储规约](docs/engineering/runtime-storage-policy.md) · [LoRA 规约](docs/engineering/lora-checkpoint-storage-policy.md)
- 用户明确委托模型子代理时，显式指定 `gpt-6.2-sol`。Codex 使用 `fork_turns="none"` 并交接目标、证据和边界。接口不支持时说明限制。[协作指南](docs/collaboration/simple-task-rl-engineering-guide-v1.md)
