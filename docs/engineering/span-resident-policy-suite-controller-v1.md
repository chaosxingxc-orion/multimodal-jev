# 多策略常驻推理验收接入自动队列

2026-11-12 UTC。已准备的span-resident-policy-suite-prepared-v1进入自动队列，
安排在三臂utility完整12步曲线之后、完整FSC矩阵之前。suite本身源码、配置
和两策略（initial/D12）不变，每策略178输入；先完成独立HF进程，再由一个
常驻vLLM引擎切换LoRA。该任务不执行训练或actor/vLLM同卡共驻。

控制器要求全部utility曲线与正确性状态重放实际验收成功，然后等待资源，
持创建锁一次派发。suite实际退出后，宿主重新验证人口、策略身份、BF16、
全候选概率及联合比率，再保存完整验收凭证。FSC前驱检查重新认证utility
完整结果与resident实际退出/证据，不凭控制器terminal或passed字段放行。

跨冻结目录的验收脚本路径可能不同，宿主比较先核对两侧脚本哈希完全相同、
旧凭证中的脚本pin正确，再仅替换该脚本路径键。其余实验文件逐项相等，
旧凭证所有文件重新检查。更改脚本或实验输出均拒绝。

初次控制器v1因旧源码包缺少verify_span_resident_policy_suite.py，在导入阶段
退出，未创建run目录或NPU任务。保留stderr与failed-startup.json，确认进程
不存在后用完整模块的新v2目录启动。随后修正下游脚本路径比较，迁移仅等待的
FSC v5→v6；不改已冻结目录。所有失败、迁移脚本和精确pidfd退出证据保留。

实际入口：resident控制器v2 PID3682839；FSC矩阵v6 PID3683328；外部报告v6
PID3683329。12:19 UTC均确认完整argv存活、配置pins一致。resident等待utility
控制器PID3678326。Q/P控制器PID3677691仍存活，已接受Q6，Q7独立suite尚未创建。
共享梯度CPU容器0076de75…实际运行中，没有因耗时长而重启。

原控制器版本16项本地及固定镜像CPU测试通过；修正路径比较后同组固定镜像
测试另行验证，见[证据](span-resident-policy-suite-controller-v1-verified.json)。
这些是调度契约与实际监督进程证据，不代表resident NPU suite成功或RL效果提升。
真实闭环仍需要suite执行、行为消费以及后续在线actor更新验收。

共享记录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-resident-policy-suite-schedule-v1/`。
