# Stage2 完整评测与存储整改接续

2026-11-19，用户授权按既定计划推进，并将资源清理委托给 Sol6.1 子代理。
训练 v0.0.93 已完成 634 更新，退出码为 0；独立训练审核尚未结束。
本说明不将成功退出视为通过审核，也不提前宣称 RL 提升。

## 执行顺序

1. 保留 v0.0.93 冻结运行，继续 CPU 收集与独立审核。
2. 使用已封闭容器的显式 manifest 子集提取十个预登记点的 LoRA/head 与加载资产，
   约 254 MiB。内容哈希、容器身份、policy manifest、native adapter 均核验；
   此准入仅允许推理，不替代训练审核。
3. v0.0.94 候选执行完整 selection：先 634，再 64/128/192/256/320/384/448/512/576。
   每点 9226 录音；四个 TP1 实例，各服务 batch32，macro128；物理卡 4–7。
   先执行独立 HF panel，再执行四实例宏批评测；CPU 输出审核与下一个 NPU 点并行。
   普通观察每 900 秒，固定脚本运行，不自动唤起模型或重启失败任务。
4. 汇总完整曲线；step0 只能在策略和输入身份一致时复用 Stage1 endpoint。
   训练审核与全部 selection 审核通过后，按原选模规约固定策略，再执行
   SLURP TEST、FSC native31、MINDS native14；目标数据不参与训练或选模。

冻结模型、任务、奖励、阈值与预登记检查点保持一致。本轮不添加复杂任务、
不额外启动效率对照。正式任务内已有时延统计用于工程效率报告。

## 工程入口与证据

- 显式收集：`tools/reports/collect_manifest_subset.py`，
  `configs/portable/operations/manifest-subset.json`。
- 评测封包：`tools/reports/prepare_committed_selection.py`。
- 评测执行：`tools/reports/context_selection_runner.py`；
  控制器 `tools/reports/run_committed_selection.py`。
- 可共享入口：`configs/portable/operations/committed-selection.json`，
  本地路径、设备与证据由 `${COMMITTED_SELECTION}` 宏绑定。
- 调用：在封存源码根目录，通过 `python -B -m tools.reports.run_committed_selection
  --config <portable-template> --local <local-macros> --action detach`。
- 当前本地宏：`/srv/multimodal-jev-xingchao/operations/stage2-selection-curve-v0.0.94-macros.json`。
- 输出：同目录下 `context-stage2-selection-curve-v0.0.94`，各点保留准确容器句柄、
  终态、传输清单与独立审核。源训练未通过审核前，结果标记
  `training_audit_pending`，不发布为最终结果。

本地 54 项测试通过；真实第634步封包重建及部署参数验证通过。
固定镜像 CPU 与 NPU 实际状态以后续容器和独立报告为准，不能用本段代替现场核验。

## 存储整改与清理边界

体积原因见 `stage2-storage-inventory-20261119.md`：主要为每步重复 execution、
processor 和 adapter，以及双rank重复轨迹；没有重复保存完整基座模型。
清理子任务使用固定脚本、显式清单、归档核验和依赖保护，独立记录删除字节数。

当前 v0.0.93 待审核产物、十个检查点、v0.0.94 评测依赖、Stage1 和共享数据模型
全部保留。历史无引用且已验证归档的临时产物可按清单删除。不能仅为减小体积
删除本轮中间检查点，破坏已预登记的完整效果曲线。

后续运行版：共享 processor/特征/reference 按身份引用，临时训推权重待消费者
确认后回收，持久评测点与必要恢复状态独立保留，轨迹只存一份；显式清单收集。
这些改动须在新候选上验证，不原地改变当前运行包及其审核契约。

## 本轮启动证据

固定镜像 CPU 独立收集：54 tests，0 failures/errors/skipped，容器退出码 0。
报告位于 `operations/context-stage2-selection-cpu-v0.0.94/report.json`。
冻结源码 manifest：`f700bd6ceb23553e81ac37e94ac6670560f4445015fd5d179072d74239693d44`。

634点评测于 2026-11-19 01:50:59 UTC 启动，容器
`aa7b0533bfbe2328fc91ac93389a87b109c92d32797b8dcda37b8c37cacac102`。
启动核验确认 HF panel 进程实际占用物理4卡；四服务阶段按脚本在该阶段成功后启动，
尚不声明完整四服务评测完成或任何效果指标。控制器 PID2901942，start_ticks362976604。
这些是启动时证据，后续状态必须重新核验。
