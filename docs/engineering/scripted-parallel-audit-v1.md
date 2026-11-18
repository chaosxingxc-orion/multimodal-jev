# 确定性并行审核规约与当前实现

2026-11-18：所有实验审核由脚本执行，不以 ChatGPT、远程教师或子代理作准入裁决。
模型用于用户要求的开发和研究解释；通过/失败、评分、下一阶段派发必须可脱离模型运行。
规则已写入 AGENTS.md「日常任务执行规约」。

## 现状

当前 v0.0.45 的审核是 Python 程序。准备、准入、曲线重建等入口会递归调用共同来源审核，
多次重读全量数据、重建候选 target、校验 checkpoint/optimizer/导出及文件哈希。
这些重复调用是工程计算，不能称为“模型独立审核”。

v0.0.45 使用线程并发准备两个评测点及验收分支，但 Python 密集循环受 GIL 限制。
其中 `cpu_workers=4` 目前只是配置校验项，并没有驱动内部进程池；不能把它描述为四核审核已实现。
冻结运行代码保持原样，新组件经有限验收后才能由新版本接入。

## 新组件

- `src/multimodal_jev/parallel_evidence.py`：配置 `io_workers`，有界线程并行读取、SHA256；每次重新读取全部文件内容，不按 mtime/size 跳过；拒绝符号链接及文件清单变动。
- `tools/reports/context_parallel_batch_audit.py`：配置 `cpu_workers`，spawn 进程池执行原始 `check_batch`；每进程 torch 线程数1，限制嵌套并发。
- 工作进程严格复用原审核函数，核验原始音频标识、候选定义、target、顺序、权重、padding及rank槽位排列；按原索引返回，不因完成顺序改变结果。
- 每个batch的原始字节必须匹配原transport的SHA256。全量覆盖不得抽样替代；串行对照面板只用于实现等价性测量。
- 线程/进程数从本地宏读取，默认候选4/4，范围1–32；不绑定个人机器路径。
- 输出 `report.json`、逐batch证据、配置与终态；失败不重试，不派发训练。无模型调用。

共享入口 `configs/portable/operations/context-parallel-batch-audit.json`，本地绑定
`.local/config/context-parallel-batch-audit-v0.0.46.json`，报告目录由 `output` 指定。
本轮证据目录 `/srv/multimodal-jev-xingchao/operations/context-parallel-batch-audit-v0.0.46`。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.local/verification-python:src:tools/jev \
/usr/bin/python -B -m tools.reports.context_parallel_batch_audit \
  --config configs/portable/operations/context-parallel-batch-audit.json \
  --local .local/config/context-parallel-batch-audit-v0.0.46.json
```

这是独立CPU验收候选，不是新NPU运行包，也不是完整训练准入审核替代品。
7项测试通过：1/2进程结果一致、排序保持、字节篡改、target、rank、weight、缺失文件，
以及同size/mtime内容变更与符号链接拒绝。真实运行检查全量318batch、81,026真实曝光，
另取前8batch做串并行差分；以真实报告为准，不用测试速度宣称生产加速。

## 后续接入要求

重复审核应收敛为显式依赖图：一次验证共同前驱，独立子任务并行，依赖成功后汇总。
缓存复用必须绑定审核器代码、配置、完整输入内容清单与输出摘要；进程退出0或旧PASS不是替代证据。
当前新组件尚未实现跨阶段审核结果缓存，也未接入冻结v0.0.45调度器。
先记录真实耗时和精确等价性，再发布新控制器接入；不能未经测量只增大线程数或删减检查。

## 真实候选验收（2026-11-18）

4进程/4线程：全部318batch、81,026真实曝光通过；文件清单校验0.582秒，
全量batch校验40.196秒，包含来源读取及差分面板的总时间55.100秒。
前8batch串行3.759秒、并行4.630秒，结果完全相同；小面板启动开销使并行更慢。
模型调用0；这只覆盖batch/文件校验，不代表完整checkpoint、曲线审核或NPU调度已加速。
全量单进程对照使用独立输出 `context-parallel-batch-audit-v0.0.46-serial`，以报告为准。

全量单进程对照已完成：batch校验148.460秒，对比4进程40.196秒，单次顺序测量为
**3.69倍吞吐、耗时减少72.9%**。两份全量逐batch结果（含输入SHA256、语义摘要、
索引和真实样本数）逐项完全一致。两次总耗时165.063秒/55.100秒。
此对照不是不同负载下的重复统计试验；不能据此宣称整条审核链或训练加速3.69倍。

## 完整来源链接入（2026-11-18 02:40 UTC）

开发态v0.0.47将并行batch checker接入完整Stage1来源审核：所有transport、checkpoint、
FP32 master/Adam、BF16导出及policy校验继续执行。真实318更新、81,026曝光的完整来源计划
与原计划逐字段完全一致，总耗时90.920秒。证据：
`/srv/multimodal-jev-xingchao/operations/context-parallel-source-audit-v0.0.47/report.json`。

v0.0.48增加显式共享审核引擎：每次使用前重新读取batch文件并计算哈希，键包含完整population、
task、日程、policy、width/world及项目审核代码内容。相同键的并发请求共享一个Future；
4个CPU进程最多只有一组batch重放在运行，I/O线程4个。返回证据深复制，失败传播，不以旧PASS或mtime复用。
其他checkpoint、optimizer、transport审核仍逐次执行；这是受限batch证明复用，不是整个证据链的任意缓存。

真实链验收：首次完整来源91.007秒，第二次45.734秒；共同batch证明重用0.65–0.82秒。
实际终点selection包完整重建125.131秒通过。共1次batch计算、3次复用，
两份完整来源证明与原始计划完全一致，模型调用0。上述时间非配对多次统计基准，不宣称整体训练同倍加速。
证据：`/srv/multimodal-jev-xingchao/operations/context-shared-source-audit-v0.0.48/`。

已由独立控制器 `tools.reports.context_audit_optimized_pipeline` v0.0.49 接入Stage1 selection：
准备、启动校验、结果审核和完整曲线重建共用该引擎；原冻结文件与NPU运行源码未改动。
Stage2 reference/GRPO审核仍用原实现，尚未消除其重复重建，不能称全链优化完成。
相关23项CPU测试通过（最终控制器调整后6项相关测试再次通过）。

运行入口：`configs/portable/operations/context-audit-optimized-pipeline.json`；
本地宏 `.local/config/context-audit-optimized-pipeline-v0.0.49.json`。
02:40 UTC新driver PID2368231，start ticks354628961；原2358767已退出。
切换脚本在持有设备租约、检查全部线程子进程、无旧NPU attempt且远端四卡无进程后执行。
新运行根 `/srv/multimodal-jev-xingchao/operations/context-async-pipeline-v0.0.49`；
切换证据 `retirement-complete.json`。原包留存供追溯，禁止重新detach旧v0.0.45。
新控制器仍为开发候选；启动不等于完成NPU评测，数值门禁和正式训练门禁继续有效。

## Stage2 reference 独立候选

v0.0.50 已完成全量40513条/317macro输出证明的内容绑定复用验收，两次证明与原始审核
逐字段一致。完整输出审核129.125秒→88.903秒（单次顺序测量减少31.15%），19项CPU测试通过。
只新增候选模块，v0.0.49的791个源码pins无变化；未热替换正在运行的Stage2审核链。
设计、命令、边界与后续发布条件见[reference审核优化](shared-reference-audit-v1.md)。

v0.0.51 已进一步定位并消除重复独立输入重建：原独立输入78.769秒，内容绑定复用5.042秒；
完整输出审核首次138.588秒、重复23.700秒。全量40513条/317macro两份证明与原证明完全一致。
相较v0.0.50的重复88.903秒，本次减少73.34%（跨运行单次测量）。首次计算没有加速，
现有冻结队列未热替换，完整Stage2生命周期接入待验收。见[输入复用与分项计时](reference-input-reuse-v1.md)。

2026-11-18 03:24 UTC：用户要求已验证修复直接接管，不等旧前置审核结束。v0.0.54完成
隔离来源进程全量等价验收；v0.0.53新控制器已停止旧v0.0.49本地driver，接管正在运行的
原终点318容器，未停止或重启NPU任务。Stage1完整曲线优先，Stage2后续流程保留。
现场身份、20项测试及证据入口见[进程隔离与接管](stage1-priority-process-handoff-v1.md)。
