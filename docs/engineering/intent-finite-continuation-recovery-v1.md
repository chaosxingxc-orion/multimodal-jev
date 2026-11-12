# 意图有限支持RL：导出权限修复与连续更新队列

2026-11-12 UTC。首步训练实际退出0后，原`intent-finite-update-queue-v3`在冻结独立验收时失败：safetensors导出为root:0600，本地控制器不能读取。该失败发生在验收容器创建之前，不能解释为训练失败。随后`intent-finite-continuation-queue-v1`识别前序失败并退出，未派发第二步；两个失败终态均保留。

只将首步`run/adapter/adapter_model.safetensors`设为0644；固定镜像、无设备挂载的短容器执行权限修复，前后SHA-256均为`3ff0f7fc7651eaa178efc6aad678280aebb78ecb817b48cb57432547f23ed23f`。没有修改模型内容、重跑首步或覆盖失败队列。证据见`intent-finite-export-permission-repair-v1.json`。

原先尚未声明的`slurp-intent-finite-dp4-suite-prepared-v2`首次冻结并派发。实际验收容器为`b86cf2c15ff0b4983fbd889ea6ddfa956be70d4bf4ef9c98b6a04dd64d7cb34e`；检查178个D/Q输入的HF/vLLM概率，以及首步优化器/导出的一致性。启动回执见`slurp-intent-finite-dp4-suite-v2-launch.json`，尚不代表验收通过。

训练源码新增导出后的显式0644权限。已准备的v1源码不改动，新建`slurp-intent-finite-resume-step2-prepared-v2`与`slurp-intent-finite-resume-step2-suite-v2`。D、seed317、批次8、lr1e-4、BF16执行/导出、FP32 master/Adam、物理卡4–7 FULL_SHARD均不变；新增权限语句不改变损失或张量计算。第二步恢复的真实参数和Adam逐张量审计仍由独立suite执行。

`intent-finite-continuation-queue-v2`明确跟踪上述验收容器，不再等待已失败的旧控制器；进程PID3539536启动身份已核对。首次观察不早于04:15:22 UTC，后续间隔600秒。前序验收实际退出0且证据完整后，才冻结/派发第二步，再执行第二步独立验收。已声明目录不会重入，创建结果不确定也不会自动重试。14项控制测试通过，覆盖前序失败、PID复用、派发回执替换、容器未结束、冻结重入与确切手动验收交接。

新增学习集成对比较工具`compare_intent_learning_update.py`，从完成的独立suite读取同一89条来源的D/Q全概率，区分本步更新8条与其余81条，并报告正确概率、NLL、预测类别数和对错转换。其余81条仍是学习发布，不是开发集。7项比较/评分测试通过，包括聚合准确率不变但更新组改善、其他来源退化的区分性案例。真实更新后比较待验收完成，未预先声明效果。
