# 原研究队列结束后的实际音频验证

已启动独立的一次性等待进程，补上效率验证尚未入队的缺口。
原 `research-successors-v1` 队列、冻结实验包和运行中训练均未修改。

入口：`tools.reports.queue_context_dual_probe`；共享模板：
`configs/portable/operations/context-dual-probe-queue.json`；本机绑定：
`.local/config/dual-probe-queue-v1.json`。有限推理使用独立的
`context-dual-probe-after-research.json` 模板和本地 dispatch 宏，原准备配置仍保持 false。

驱动 PID **1770867**，start_ticks **338720209**；进程身份收据位于
`/srv/multimodal-jev-xingchao/operations/dual-probe-after-research-v1-driver.json`。
状态目录为同级 `dual-probe-after-research-v1`。启动后实际 pidfd 检查确认存活，
状态为 `waiting_exact_research_process`。

它等待 PID 1330344、start_ticks 333752347 的准确进程退出，不进行周期性模型调用
或远程 NPU 轮询。退出后重新检查 1242 个依赖/配置 hash，并复用原 probe 的
成功队列输出审计、排他锁、实际设备空闲检查、固定镜像、只读输入挂载、准确容器
等待和独立实际音频审计。原队列失败、依赖变化、锁或设备被占用均停止并保留现场；
不会自动重试，也不会停止任何已有服务。

范围仅为已封存的 serial/batch 实际音频 logits、概率与 G16 一致性验证。
成功后仍须开展 DP4 两步匹配更新、独立审计、新进程恢复和代表性持续吞吐测量。
这些后续步骤尚未入此队列；本队列不允许正式训练或提前切换旧训练。
因此“已入队”不表示已经完成 NPU 验证，更不表示存在训练提速收益。

验证：`test_context_dual_probe_queue.py`、`test_context_dual_probe_operations.py`、
`test_recovered_research_successors.py` 共 **38 passed**；新队列测试使用模拟
前驱和 launch，验证顺序、重复启动拒绝、失败不重试、准入拒绝及审计失败。
真实 1095 文件音频包通过本地 validate；本轮没有派发 NPU 容器。
