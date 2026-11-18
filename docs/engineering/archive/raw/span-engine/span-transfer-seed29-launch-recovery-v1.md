# Seed29 完整源码快照重派与评估衔接

2026-11-12 00:26 UTC 核验。当前基座保持 Qwen3-ASR-0.6B，BF16 LoRA，
固定原 transfer protocol、seed29、structured、auxiliary weight 0.2、12 次更新。
没有切换模型、调奖励、改变数据或放宽概率 gate。

此前 conditional seed29 v1/v2 从旧运行目录复制入口，遗漏写死物理卡 5 的检查。
v2 在 version-0000 的 vLLM 验收入口退出 1，未产生任何 update 目录。
同源 SFT seed29 v1 在初始 HF 验收期间主动停止，未产生 update 目录。
这些根目录保留为失败/停止证据，不纳入有效训练轨迹。

新增 `tools/jev/launch_span_transfer_training.py` 从当前工程完整复制 src、tools、
temporal task 配置；逐文件验证快照等于当前源码并编译所有 Python 文件。
启动前按原协议验证全部 12 个均衡批次，绑定独占根目录和显式物理卡。
快照 manifest 记录文件哈希及 parent commit；未提交文件也按实际字节固定，
因此不能仅凭 parent commit 重建本次运行。

实际启动回执：

| 分支 | 根目录末级 | 卡 | 容器 ID 前缀 |
|---|---|---|---|
| conditional seed29 | span-transfer-conditional-s29-a2-v3 | 4 | de477c4681d3 |
| SFT seed29 | span-transfer-sft-s29-a2-v2 | 7 | a45661d0ba90 |

00:23 左右远端 docker ps 确认两个容器运行；这仅证明已启动，不证明已完成
首个更新或 gate。seed17 的 SFT/GRPO 及其队列保持不变。

新增 detached 评估队列：

- `span-transfer-conditional-s29-a2-evaluation-queue-v3`，PID 3494534，
  目标 `span-transfer-conditional-s29-a2-evaluation-v1`。
- `span-transfer-sft-s29-a2-evaluation-queue-v1`，PID 3494535，
  目标 `span-transfer-sft-s29-a2-evaluation-v1`。

00:26 的进程表确认两个 supervisor 存活，各自 progress 为
`waiting_existing_training`。队列绑定完整训练容器 ID，并等待该容器退出 0；
随后核对源码哈希，独立审计完整轨迹，再在原卡派发固定初始/末次策略的全量评估。
队列派发成功与评估完成分开记录。不得原位修改其固定的 src/tools/task 文件。

回归验证命令：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_span_transfer_training_launcher.py \
  tests/test_span_transfer_evaluation_queue.py \
  tests/test_span_transfer_runtime.py \
  tests/test_span_runtime_topology.py
```

37 passed。新增测试覆盖当前完整快照、显式卡 4/7、独占目录、无效协议拒绝；
既有测试覆盖异常训练/容器替换/源码改变时拒绝评估，以及批次与拓扑约束。
这些 CPU 回归不替代本次 NPU 验收或训练结果。下一步需完成三方法固定评估、
多 seed 对照，再判断任务提升；四卡真实恢复与跨数据集泛化仍未完成。
