# Stage2 难例视图启动交接：v0.0.125

> 本页为历史交接。181 的正式训练和 selection 结果见 [0.6B Stage2 交接](stage2-preflight-host181-20261119.md)；69 的 1.7B 两阶段启动与断点续跑见 [接续规约](context-stage2-endpoint-successor-v1.md)。

状态：已冻结工程候选，已完成宿主回归和 175 固定镜像 CPU 封包验证；未对本轮终点执行 NPU 验收，尚未产生正式 Stage2 包、更新或指标。09:26 UTC（2026-11-19）按实际 container ID inspect，Stage1 rc7 为 running、OOMKilled=false，尚无正式 collection。旧 rc5 reference 和旧 634 步全量训练不满足本轮依赖。

## 资源与实验约定

仅 clean SLURP TRAIN 的单音频原生意图；模型直接听音频。Stage1 终点完整策略初始化 Stage2，重置 Adam，按已准入的 LoRA reuse 配置继续；不隐式启用未准入的堆叠 adapter。全 TRAIN 的每条录音都有固定 reference；音频镜像、特征和 reference 都复用。用 TRAIN 的错误样本、间隔不足的边界样本及简单回放定义训练人口，selection/test/FSC/MINDS 均不参与采样。

使用 DP2 训练、两个单卡推理实例，计四卡；每训练卡有效 batch=64、microbatch=8，global=128，服务内部 batch=32。两个 epoch，真实曝光为 2N，更新数为 2×ceil(N/128)，N 为该次冻结视图的实际人口。回放率和按类保底在 stage2-dataset.json 配置。每个 epoch 内真实 ID 恰好出现一次；零权重填充不算曝光。若 N<128，跨 epoch 的诊断两批都可能有 padding。

本轮奖励候选是 [概率间隔目标配置](../../configs/experiments/stage2-probability-margin-candidate-v1.json)：排序/存在性/任务级 RL 的初始权重分别为 0.25/0.25/0.5，配合正确候选相对最强错误候选的概率间隔目标 M=0.1。部署时绑定编辑文件的实际 SHA 和完整 options；换 reward、系数、父包、数据集或 actor 组件，旧有限收据不能直接沿用。CPU 验证没有证明该目标提升准确率或跨域迁移。

## 操作顺序

统一使用 [run_simple_experiment](../../tools/reports/run_simple_experiment.py)，运行位置为冻结源根；配置模板、源快照、后端配置指纹及本地宏规则见[统一入口指南](simple-experiment-admission-v1.md)。每个 window 的输出、driver、tag、remote_root 均独立绑定，不覆盖冻结包。先 plan，再按阶段 run 或 detach；plan 不占卡。

1. 等 rc7 正式终点独立 collect/audit。建立与该终点、同版 stable-prefix 编译器及同一推理 runtime 身份一致的 Stage2 父包和全 TRAIN reference；验证完整候选顺序、策略、输入 cache 身份。历史父包只用作对照，不能填入本轮宏。
2. **stage2_dataset**：以该父包的实际 config.json 为 STAGE2_INPUT_CONFIG，用标准 release 产生视图、绑定配置及其收据。视图只做一次确定性生产与独立复算。
3. **stage2_prepare**：使用 [stage2-view-package.json](../../configs/portable/operations/stage2-view-package.json)。STAGE2_VIEW_PARENT 为 {root,manifest_sha256}，DATASET_RECEIPT、BOUND_CONFIG、OBJECTIVE_DESIGN 三个同前缀宏均为 {path,sha256}。WINDOW 依次可取 prefix、recovery、tail、formal。前缀和尾批的 RECOVERY、ADMISSION 取 JSON null。准备操作只写本地封包。包内只复制 ID、调度、诊断、消费契约和小型 actor 组件，不另打包共享模型、音频或 reference。
4. **stage2_window**：使用 [stage2-view-window.json](../../configs/portable/operations/stage2-view-window.json)。STAGE2_VIEW_WINDOW_OPERATIONS 对象包含 schema=jev-stage2-view-window-v1、window、dispatch、output、monitor_interval_seconds=900、image_cpu={path,sha256}、objective_loss_atol、deployment、training_package。后者含 output、manifest_sha256、reference_job.collection；deployment 采用原四卡 host、lease、shared_root、driver_root、package、manifest_sha256、operations_root、tag，必要时包含 release_evidence。每次显式提交一个窗口并独立审核，失败保留 exact handle，不自动重试。
5. **prefix** 两步：验证完整候选、逐 rank 真实分母、一次完整行为 macro、固定 reference、BF16 训推、LoRA/head 发布与服务激活；独立重算可编辑目标的有限轨迹。**recovery** 从该 prefix 第一步恢复、执行第二步。prepare 的 RECOVERY 对象填 audit、transport（均为 path/sha256）及 run；window 的 prefix 对象填 audit、run。必须复核另一容器的 Adam、参数、batch、rollout、奖励轨迹。**tail** 用本轮第一轮尾批和第二轮首批检查真实分母。旧包的有限收据不计入本轮。
6. **formal** 准备时可先用 ADMISSION=null 预览。真正可执行的正式包需另取新目录，以 prefix/recovery/tail 三份审计收据的 path/sha256 作为 ADMISSION；正式运行时同样提供 admission。缺失、容器重复、血缘不一致或恢复证据不完整均拒绝。效率用正式训练计时，不新增独立 benchmark。
7. 每个窗口成功时在 output 写 audit.json 和 terminal.json；评测仍按登记的源域曲线和迁移协议独立执行。本轮没有后台排队或自动启动正式 Stage2。

调用示例（所有路径由本地发布环境给出）：

```bash
cd "$EXPERIMENT_SOURCE_ROOT"
PYTHONPATH=src:.:tools/jev "$EXPERIMENT_PYTHON" -B -m tools.reports.run_simple_experiment \
  --config configs/portable/operations/simple-experiment.json --local "$LOCAL_MACROS" --action plan
```

## 已核实的证据与剩余验证

候选收据：`/srv/multimodal-jev-xingchao/operations/stage2-view-base-4mrfTP/candidate.json`。绑定源码 manifest SHA256 为 `2dddb8e90dd09ceb97a0656b8776d12d51b00015c212baff54054367e5f15882`。同目录 host.xml 107 项、controller.xml 40 项通过；175 固定镜像 CPU report 位于 image-2/report.json，6 项通过、未挂载 NPU。镜像测试覆盖真实发布格式和封包逻辑，输入为小型测试数据，并未加载 Qwen、vLLM、执行 HCCL 或重放真实 optimizer。示例五条入选样本、二十条完整 reference，两个 epoch 共十次真实曝光、其余为 padding。冻结快照与 CPU 测试包共同收录的 518 个项目文件内容一致。

真实样本数、设备数值、实际尾批、吞吐、训练时长及收益均须待本轮终点和真实设备结果确认。固定镜像 CPU 只做一次；不能据此预填正式审核收据。此前“旧全量后端不得接收视图”的禁令继续适用；新实验阶段名为 stage2_window，不用 stage2_full。
