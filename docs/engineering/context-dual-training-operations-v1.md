# 有限双信号训练的运行入口

`tools.reports.context_dual_training_operations` 为已经封存的 batch1/batch4
两步初始对照包提供启动和收集入口。它复用原设备锁、准确容器等待和训练 collector，
没有新增训练器或自动重试队列。所有机器路径由本地宏提供。

```bash
PYTHONPATH=src:.:tools/jev python -m tools.reports.context_dual_training_operations \
  --config configs/portable/operations/context-dual-training-operations.json \
  --local .local/config/dual-training-batch4-operations-v1.json
```

默认命令只本地校验。两个真实对照包均已通过此入口的完整文件清单、runtime
manifest、包内 pins 和 Docker command 检查。共享模板保持
`finite_training_dispatch_allowed=false`；`--launch` 也不能绕过此条件。

启动流程要求：

1. 获得原训练/评测排他锁，拒绝已存在的 attempt/handle。
2. 重算已完成真实音频 probe 的独立 audit，确认 logits/probability 和策略/镜像身份。
3. 验证固定镜像存在、物理 4–7 卡均空闲，以及本地可写盘的容量和 inode 余量。
4. 检查全部外部 pins，显式只读挂载共享基座/数据、Stage1 LoRA/head 和音频镜像；
   现有 encoded cache 使用原可写子目录，运行输出与输入根目录分离。
5. 封包运输校验，创建唯一容器，先持久保存准确 ID，再 start。启动响应超时继续
   wait 同一 ID，不重建、不再次 start。
6. 无论任务成功与否，正常获得终态后由原 collector 保留结果；随后核查实际
   image、command、device、mount、env、退出码及 OOM。成功运输仍需独立训练审计。

容器使用 24 CPU、128 GiB、8 GiB shm，挂载物理 4–7 卡及管理设备；service 仍使用
已验证的进程可见性规则限制到物理 7 卡。启动器不使用 `--volumes-from`，所有输入
挂载可从配置和 pins 检查，不继承历史容器的隐式可写挂载。

用户允许在确定效率收益后提前切换，因此新增
`configs/portable/operations/context-dual-probe-idle-window.json`：真实音频 probe
可在获得原排他锁且设备空闲的窗口执行，不再要求整条旧研究队列完成。
原默认 completed_research_chain 模式保留。两种入口都不会停止任何已有任务；
它们也不能在占用设备上用“空闲窗口”标记绕过实际检查。

当前入口支持 start0→stop2，以及绑定已审计父前缀的 start1→stop2 恢复封包和独立
双信用训练审计；完整组合路径已通过 CPU 合成归档检查，见
`context-dual-training-audit-v1.md`。尚无真实 NPU prefix 或效率结果，不能据此切换旧训练。与旧训练调度器的
受控交接仍需实现；现有运行的配置及后台进程未改动。

本地 31 项 operations/probe 检查通过，包含占用锁、错误包、非法路径、超时后准确
handle 收集等。另有 11 项吞吐准备/比较测试。固定镜像 42 项全部通过、零跳过；
容器 `914f287ab193ee3bdde48c611ab441972ba5f962ea7b96e1e0eb34a8f0a7f78f` 已准确
wait 和独立收集，exit0、非 OOM，410 文件快照。证据为
`context-dual-training-operations-fixed-image-cpu-v1.json`。远程操作均为测试替身，
不属于 NPU 实测。
