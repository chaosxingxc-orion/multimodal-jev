# 正式训练后的完整源域评测队列

`run_context_source_curve_successor.py`通过固定脚本衔接正式训练及源域selection，
不调用模型分析状态，不周期轮询，不自动重试、不恢复训练、不读取迁移目标数据。

执行顺序如下：

1. 等待正式训练后继的准确PID/start_ticks/cmdline退出，要求634步收集审核和源评测
   清单绑定成功。等待使用pidfd；存在锁或旧终态文件不作为进程身份的替代。
2. 重新审核正式训练容器/transport/完整学习输出，重新构建实际源清单；与前驱
   实际launch-config中的operations完全匹配，防止使用另一次训练的清单。
3. 依次执行0、64、128、192、256、320、384、448、512、576、634这11个点。
   每个点重新绑定真实LoRA/head权重、完整9226条SLURP selection输入及缓存证据，
   复用已有封包、双TP1服务、macro128/rollout32、共享设备租约、准确容器生命周期。
   同一个点只派发一次；失败保留准确句柄及已有结果，不跳过失败点继续选模型。
4. 每点完成后收集并独立重新评分；全部完成后调用原source curve collector，
   再次重建完整源曲线，按固定set F1主指标、NLL等次指标规则冻结三个策略角色。
   目标TEST和跨数据集成绩不参与选择。

脚本只串接既有工程接口，没有新增训练/评分算法，也不会以部分曲线选胜。
`source_curve_completed_roles_frozen`终态才表示完整源曲线完成；它不表示FSC/MINDS
迁移或SLURP TEST已经评测。实际角色来源JSON随后供全量目标部署/评分入口消费。

## 配置

portable模板为`configs/portable/operations/context-source-curve-successor.json`，
读取本地宏`SOURCE_CURVE_SUCCESSOR`。配置包含前驱driver/config/terminal、source_plan、
本地输出/封包/operations/curve目录、远程根目录、标签前缀、固定输入证据及pins。
所有机器路径只在本地宏提供；源输入字段白名单不允许混入另一个parent/policy/step。

每个实际评测点的operations由脚本写入独立本地宏文件，模板为
`context-source-point-operations-object.json`，宏`SOURCE_POINT_OPERATIONS`。
完整曲线配置保存在队列目录`curve-config.json`，可供原collector独立复算及迁移准入。
所有输出目录必须是新的、互不包含的本地持久目录。重复运行拒绝覆盖。

## 当前执行证据

2026-11-16 15:03UTC：本地配置`.local/config/source-curve-successor-v1.json`已生成并
校验，142个pins；后台启动58961终态0，实际等待进程PID1918252，start341804511，
cmdline hash `bf94ddf6cc9a69f7c115aab80f3ff6dec2182a6f94ce314128c550db919fa5fb`。
准确/proc身份已验证live；`operations/source-curve-successor-v1/waiting.json`绑定
formal-successor1916785/start341776302。现在仅等待，没有生成实际源曲线分数。

全部五条驱动准确进程身份仍live，73/81/95/117/142pins未变化。不要重复派发或修改
运行队列依赖。实际源评测输出将写到`operations/source-curve-points-v1`，最终曲线
写到`operations/source-curve-v1`，当前这些目录尚未由模型运行生成。

顺序/失败/已有输出/detach测试与原source collector/point封包组合44项2.83秒通过，
执行80432终态0。测试真实配置与文件写入、11点遍历、顺序和停止逻辑，容器/训练/
模型证据为替身；不将合成结果作为NPU效果或迁移收益。
