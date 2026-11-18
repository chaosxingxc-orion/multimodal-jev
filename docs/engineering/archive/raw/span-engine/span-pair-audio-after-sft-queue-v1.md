# 卡5成对奖励探针自动衔接

2026-11-12 01:03 UTC，detached supervisor PID3500897 已确认存活，正在等待
既有 SFT 完整评估容器
`c099602ada42e2a6af24450a0e39649ce78a590a0ee207bd0fedbd3879ccde1b`。
队列根 `span-pair-audio-after-sft-queue-v1`，目标为已预检的
`span-pair-audio-pilot-prepared-v2`；当前仍未启动该 NPU 探针。

`queue_span_pair_pilot.py` 绑定原容器ID，以docker wait等待成功退出；SSH错误、
非零退出、源码/证据变化均终止队列，不重启评估。进入派发前持共享卡5启动锁，
检查所有运行Docker容器是否映射卡5或具有privileged权限，再检查host进程占用。

远端SSH用户UID9009，普通fuser可能看不到其他用户描述符。因此另用固定镜像
短命CPU容器，在host PID namespace、root和SYS_PTRACE权限下只读stat主机进程
fd与卡5设备号。该检查不映射NPU设备、不打开设备、不结束任何进程；发现占用、
权限拒绝或检查失败均拒绝派发。此额外主机检查路径已做故障单元测试，尚待队列
实际触发验证，不能把模拟通过当作主机路径已实测。

卡可用后重核CPU预检结果及其全部文件哈希，重建launch plan并要求与排队版本
逐字段相同，再docker create、记录完整container ID及launch.json、最后start。
start前再次检查卡占用。超时或start失败不删除重建容器，保留原ID供明确恢复。
共享锁只协调使用同一锁的任务，不构成全服务器调度器；外部任务仍可能有竞态，
所以不声称获得全局资源预留保证。

排队过程中不修改既有SFT/GRPO/conditional轨迹，不影响CPU三方法比较队列。
两成对奖励分支各一次更新，仍必须通过独立HF重载和224请求完整vLLM验收。
队列退出0只代表容器已派发，不能替代NPU终态、READY、轨迹独立审核或科学收益。

新增队列测试和控制器测试合计22项通过，包含其他容器/主机进程占卡、权限不足、
SSH失败、证据变化、create/start失败，以及先写回执再start的顺序检查。
运行证据见同名launch.json；旧流程与完整准备见
[准备记录](span-pair-audio-pilot-prepared-v2.md)。
