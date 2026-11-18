# Stage2 reference 重复审核优化

## 边界

开发候选 `v0.0.50`，入口 `tools.reports.context_shared_reference_audit`。
本候选只优化 CPU reference 输出证据审核，不运行模型、不调度 NPU、不改变训练配置。
原 reference 审核器保持逐字节不变，候选通过显式函数适配器接入。
现有 v0.0.49 调度器没有热替换；其 Stage2 审核仍使用原实现。

## 方法

`ReferenceAuditEngine` 只复用全量 reference dataset 与 macro journal 的叶节点审核证明。
每次调用都重新读取 dataset/journal 所有文件并计算 SHA256，记录目录清单，拒绝符号链接。
键包含完整 population、独立输入 binding、policy/runtime identity、macro/rollout 拓扑，以及
`src/multimodal_jev`、`tools/reports`、`tools/jev` 中 Python 源码内容。
空的额外 macro 目录也必须使缓存失效。不能用文件时间戳、大小或旧 PASS 代替内容验证。

相同键通过共享 Future 合并计算；最多一个叶节点重放在运行，默认保留两个结果。
失败传播，不自动重试；结果深复制防止调用者污染共享证明。计算完成以及交付前再次校验
完整内容键。证据必须冻结；这些校验不是并发写文件时的原子快照保证。

外层 endpoint/commit、数据发布来源、独立输入 receipts、40513 条输入与缓存记录的
映射在每次 `replay` 时仍执行。完整容器/transport/parity 审核继续由调用方负责；
`results_auditor()` 只替换原 `audit_results` 的输出重放入口，其余检查保留。
不能把叶节点命中称为整条 Stage2 准入完成。

## 配置与执行

共享模板 `configs/portable/operations/context-shared-reference-audit.json` 从本地宏
`SHARED_REFERENCE_AUDIT_CONFIG` 读取配置，不绑定个人路径。字段：

- `output`：新的独立结果目录，已存在则拒绝覆盖。
- `io_workers`：1–32 个 I/O 线程，本次4。
- `reference_audit_config`、`expected_output_audit`：原证据路径与 SHA256。
- `pins`：审核代码路径与 SHA256。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.local/verification-python:src:tools/jev \
/usr/bin/python -B -m tools.reports.context_shared_reference_audit \
  --config configs/portable/operations/context-shared-reference-audit.json \
  --local .local/config/context-shared-reference-audit-v0.0.50.json
```

程序连续重建两次完整输出证明，逐字段比较原独立审核结果，并要求一次计算、一次复用。
输出配置、provenance、两份证明、耗时报告与带退出码的终态。历史证明中的 provenance
保留以便精确差分；本次执行 provenance 单独存储，不冒称历史执行身份。

## 验证

19项 CPU 测试通过，覆盖原 journal 重建、并发去重、返回对象隔离、同size/mtime篡改、
日志缺失、额外macro目录、policy/binding变化、概率篡改、符号链接、代码变化、审核中
证据变化以及适配器没有改动原模块。测试不代表真实数据上的加速幅度。

真实证据：`/srv/multimodal-jev-xingchao/operations/context-shared-reference-audit-v0.0.50/`。
耗时和全量差分结论以该目录 `report.json`、`terminal.json` 为准。

本次真实验收已 exit0：40513条、317个macro，两次完整输出证明均与原始证明逐字段一致。
首次输出审核129.125秒，第二次88.903秒，耗时减少31.15%；其中叶节点含完整内容验证的
耗时44.356秒降至4.737秒，减少89.32%。共一次原始journal计算、一次复用，模型调用0。
这是同一进程顺序执行的单次冷/热复用测量，未做多次配对统计；不是完整Stage2链或训练加速。
外层校验仍约84秒，应继续分项测量数据来源重建、独立输入编译和记录绑定，不能将这一差值
直接归因于某一个函数。

## 发布条件

只有完整 Stage2 runtime/transport/parity 链接入后的差分验收通过，才可由新版本调度器使用。
冻结任务不热替换，运行或排队依赖不原位修改。本候选全量输出证明等价也不自动授权跳过
上述验收；未宣称训练速度或 NPU 利用率已改善。
