# 双训练卡与双推理实例的两步工程验收

v6 已正常结束，完整收集与独立审计通过。原自动收集失败记录保留；恢复收集没有
重跑训练或修改参数。证据见同名 JSON，以及
`/srv/multimodal-jev-xingchao/operations/macro-training-v6/audit.json`。

固定镜像、Qwen3-ASR-0.6B、GRPO joint 排序/存在性目标、global128/micro8；
175 卡4/5训练DP2，卡6/7各一个TP1推理实例。每个rollout由一个实例完整处理，
macro128完成后训练消费，更新后等待两实例确认。主干/LoRA/head执行BF16，master/Adam
和native verl loss为FP32；排序小张量使用CPU FP64协议。

审计覆盖全量TRAIN身份、实际两批目标、组奖励/优势、完整macro轨迹、逐microbatch
训推对齐、双实例策略激活、LoRA/head更新、FP32 master到BF16执行导出、原生adapter
逐张量复核及checkpoint范围。两步均更新112个LoRA张量，head分别更新13/12个张量。
checkpoint仅保存LoRA/head及优化器恢复信息，不保存主模型。

两步完整耗时82.0453秒、80.0685秒，平均81.0569秒，1.5791 query/s。
平均actor39.6409秒、准备36.7939秒（其中rollout31.1306秒）、导出/保存/同步4.6216秒。
rollout内部reference15.7328秒、behavior与credit14.4380秒；嵌套时间不能重复相加。
两批reference均冷缓存，共256 miss；actor峰值分配39,804,589,568字节，约37.07GiB。
未测启动/评测总成本和推理显存；不能用两步或不同batch的reward变化宣称任务提升。

收集修复：发布目录继承mkdtemp的0700权限，safetensors文件为0600；普通用户无法读取。
旧rglob枚举还会跳过不可读目录。现收集器遇到不可读目录立即报错，实际目录/两个
adapter文件权限修复并逐文件验证哈希不变，原不完整归档保留。源端三个policy发布入口
补齐目录0755/文件0644，保留fsync和原子发布，28项CPU测试通过（含严格umask回归）。
本地审计缺少safetensors，通过独立目录安装固定镜像同版0.8.0解决；未修改训练镜像。

新进程恢复包已从实际审计通过的第一步checkpoint生成，保留原冻结Python源码，
只重放第二步并与连续训练的group、execution、双rank参数和Adam逐张量比较。
该验证现已通过：见 `context-macro-replay-v1-observed.json`，新进程第二步与连续训练
逐张量一致。冻结恢复包的旧发布权限由固定脚本修复，保留前后哈希收据。
后续100step效率对照见 `context-macro-efficiency-v1.md`。
