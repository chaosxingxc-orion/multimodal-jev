# 正式SLURP全量GRPO启动入口

## 2026-11-16 14:58UTC：固定后继已启动等待

`run_context_formal_successor.py`及portable `context-formal-successor.json`已接入：
精确等待现有parity驱动退出→确认三数据集成功→逐个重新审核完整容器/transport/
实际HF与macro结果→重新审核100步选择并封正式包→独立正式启动准入→共享生命周期
运行634步→完整收集审核→生成实际11个source selection点的清单。
该脚本不生成评测成绩、不自动重试或恢复，不对远程进行周期轮询。

本地配置`.local/config/formal-successor-v1.json`锁定117个源码/配置依赖。后台进程
PID1916785，start_ticks341776302，cmdline hash
`fa2556df1f31c08753d80c7fbf172e82680a8b03011be3c3228fad2c2f51644b`。
准确/proc身份已核验live，`operations/formal-successor-v1/waiting.json`绑定现有
parity进程PID1900510/start341333918，不占用NPU。启动session64838终态0，
父脚本退出与后台训练终态有别；现在仅进入等待，尚未正式封包或派发。

后继顺序/失败停止/重复detach与formal启动组合28项3.45秒通过（16061）。覆盖前驱
失败、等待后pin改变、任一parity失败、prepare/launch/collect/plan失败及已有attempt。
单元测试的进程、模型和远端动作是替身；实际后台进程等待身份已另行核验。
四条驱动73/81/95/117pins全部不变，不要再启动相同后继或修改它们依赖。

`tools/reports/launch_context_macro_formal.py`补齐正式训练的实际启动入口。它复用
现有DP2训练与双TP1推理的容器生命周期、准确句柄、设备租约、空闲检查、本地存储
和失败收尾规则，不实现第二套训练驱动，不修改正在运行的效率/尾批/精度队列。

正式范围为SLURP clean TRAIN 40513条、两轮634步、81026真实样本曝光，从共同
Stage1重新起步。global batch=64×2个DP副本，microbatch由已完成且匹配的效率
对照确定；当前服务容量对照固定micro8。算法继续双信号GRPO，执行BF16，master/
Adam为FP32。每步只导出LoRA/head并同步推理服务，恢复状态每64步及终点保留。

## 准入过程

1. 校验已封正式包，重新审核完整100步效率对照并重算预声明选择结果，不使用部分
   步数、OOM前缀或手工填写的赢家。原FORMAL证据必须与重建结果一致。
2. 将正式训练与被选择的父配置逐项比较，只容许预算、明确formal标志、稀疏保存、
   输出路径与依赖身份变化。奖励、优化器、loss、数据/调度、起点、精度等保持原值。
3. 原1098个运行源码文件不变，仅增加formal与resume入口；基座引用、adapter、
   源策略及日程等继承资产逐字节相同。服务配置保留所选容量和全部执行设置，内部/
   外部依赖pins完整，拒绝修改配置后只重新计算manifest。
4. 重新读取正式全量日程，核实81026曝光、634步及65真实+63零权重尾批。
5. 重新审核实际尾批诊断的容器、transport、输出、梯度/导出和激活证据，要求与
   正式任务、源策略、loss和基础运行源码一致。尾批可使用另一个已完成效率组的
   服务容量；其算术验证不能替代所选容量的100步运行验证。
6. 成功后走原共享生命周期，拿同一设备租约，检查175卡4–7实际空闲，保留
   preflight、双service、client的准确句柄。启动证明写入原attempt.json的provenance。

正式封包中的`formal_launch_admitted=false`表示封包本身不授予执行资格，仍保持
不可变。外部启动入口每次重建实际准入证明；不原位修改包以翻转这个标志。

## 配置和调用

portable入口：`configs/portable/operations/launch-context-macro-formal.json`，
读取本地宏`FORMAL_TRAINING_LAUNCH`，对象包含：

| 字段 | 用途 |
|---|---|
| `operations` | 正式prepare产生的完整operations对象 |
| `preparation` | 原prepare配置，含comparison_kind/config/local/project等 |
| `tail_operations` | 实际成功尾批尝试的完整operations对象 |
| `formal_training_dispatch_allowed` | 明确是否执行，admit不派发 |

```bash
PYTHONPATH=src:tools/jev python -m tools.reports.launch_context_macro_formal \
  --config configs/portable/operations/launch-context-macro-formal.json \
  --local /path/to/local-macros.json --action admit
```

`--action run`执行同样的准入后实际启动；长任务须通过固定后台驱动运行，保留日志/
身份/终态。既有attempt拒绝重新启动，失败也不自动重试。收集和源曲线继续使用
`context_macro_formal_operations`与`context_macro_source_evaluation_plan`等既有入口。

## 已验证内容和待完成项

实际封包静态核对通过：service32父包1098个Python文件在尾批包中全部一致；尾批
仅新增formal/tail两项入口。实际训练配置归一化后与尾批、计划formal配置一致。
证据 `/srv/multimodal-jev-xingchao/operations/formal-launch-static-binding-v1/report.json`，
执行60561终态0。这不是尾批NPU验收，也没有正式派发。

首轮测试因复用的旧夹具缺少lease字段失败，补齐夹具后13项通过；增加完整依赖pins
校验后，与正式封包/服务容量选择/命令隔离组合57项5.35秒通过。测试实际封包、
配置重定位、源文件/资产比较和命令构造；容器、效率和尾批运行证据是替身。
补充服务batch候选胜出/退化保留基线的启动选择测试后，最终组合59项5.60秒
通过（22543终态0）。三条既有驱动准确进程身份均live，73/81/95个pins未变化。

稀疏checkpoint的实际NPU恢复仍未验证，启动证明明确
`sparse_npu_recovery_verified=false`及`automatic_resume_allowed=false`。
fresh训练允许从已验证的共同Stage1开始；如发生中断，必须单独验证恢复，不能
以本启动入口证明连续执行/恢复等价。当前100步对照、尾批真实运行及正式封包尚未
完成，因此没有正式启动配置或派发。目标迁移精度队列仍按原序列执行，不抢设备。
