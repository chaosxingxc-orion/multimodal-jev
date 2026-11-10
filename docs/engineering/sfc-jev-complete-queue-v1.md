# SFC-Jev 完整分片队列

`tools/jev/run_sfc_jev_shard_queue.py` 将已登记的完整 parent 人口计划逐片派发、
逐片验收，全部成功后调用 `merge_sfc_jev_shards.py` 汇总原始输出并评分。
本实现尚未启动真实队列，不能作为模型效果或吞吐证据。

## 范围与前置条件

- 原始完整计划：learning 69 片 / 69,210 决策，selection 18 片 / 17,469 决策，
  test_id 39 片 / 38,697 决策，test_ood 12 片 / 11,346 决策。
  每次队列绑定其中一个完整人口；不截取前几片或按结果筛选样本。
- 当前仅支持冻结 parent 指令。教师候选需完成语义审查、实际 processor
  上下文检查和独立运行协议后再接入，不把 parent 结果当成提示优化结果。
- 准备与执行均验证 TRAIN78 成功验收及其确切 Docker 状态、全量 CPU 上下文审计、
  原始数据/划分/计划、模型与运行时绑定；准备不是绕过前置条件的入口。
- 真实 TRAIN78 尚未完成。完整队列派发前需用其实际耗时估算完整人口成本，
  排定与现有 native SFC 队列的资源顺序。

## 固定源码与调用

准备入口要求显式 Git commit。正在执行的仓库 `src/`、`tools/` 全部 Python 文件
必须与该提交一致，包括拒绝额外未跟踪 Python 文件。导出独立只读源码目录，
不链接可变工作树。每次派发前检查文件集合和内容哈希，执行时还检查 runner
与关键导入模块确实来自该快照，防止只校验磁盘快照却执行工作树代码。

以下为调用格式，尖括号参数必须替换为正式登记值：

```bash
PYTHONPATH=src .venv/bin/python -B tools/jev/run_sfc_jev_shard_queue.py \
  --prepare --root <new-queue-root> \
  --template <first-shard-config.json> --template-hash <sha256> \
  --official-scorer <metrics_multilevel.py> --scorer-hash <sha256> \
  --source-commit <commit>

PYTHONPATH=<new-queue-root>/source/src .venv/bin/python -B \
  <new-queue-root>/source/tools/jev/run_sfc_jev_shard_queue.py \
  --root <new-queue-root> --queue-plan-hash <sha256>
```

主进程环境必须包含固定的官方 scorer 依赖。长任务还须通过现有脱离交互会话的
监督方式启动，并记录监督进程 PID/启动身份与完整命令。上述示例本身不启动后台任务。

## 执行与失败语义

- 每片使用 NPU6/7、BF16 TP2、4096 上下文、1024 输出预算以及已验收的冻结运行时。
  每片容器最多 86,400 秒，观察器最多 86,700 秒；执行前检查卡是否被占用。
- 原始模型、数据、TRAIN 验收、CPU 审计、划分和源码只读挂载；队列结果目录可写。
  CPU 验收文件从本机工作区按原始字节复制到共享队列目录并保持原哈希，
  避免把 175 服务器不可见的本机文档路径传入容器。
- 持久化排他派发意图，保存确切容器 ID、启动参数、观察状态、原始结果、验收及终态。
  任何失败都停止后续分片，不自动重跑；已完成分片保留在增量 roster 中。
- 观察超时仅代表协调器观察失败，不能据此宣称容器已停止。必须重新观察相同 Docker ID，
  或人工核对派发日志中的身份，不能启动替代容器。
- 全部原始分片通过工程验收后，才读取目标并执行严格 Jev 评分和原始 SFC scorer。
  TRAIN 仅评分登记的完整 learning/selection 人口，TEST 保持原始完整划分。
  不修复原始输出，不平均分片百分比，不以任务正确率作为工程验收门槛。

## 验证边界

新增测试使用真实 Git 临时仓库验证快照来源，使用真实合成数据重建完整人口计划；
NPU 派发和前置 Docker 验收在单元测试中明确替身化。
覆盖完整顺序、分片准备/派发/验收失败、源码和输入漂移、导入来源、挂载及禁止重放。
关联的分片 runner、merge 和 native 队列测试一并运行；这些不等同真实 NPU 全量验收。
