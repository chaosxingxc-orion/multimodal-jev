# 关闭一阶动量的D对照准备

2026-11-12 UTC。目标是检验既有D学习诊断中的Adam历史动量因素；不改变奖励、数据覆盖或候选配置。参考step3/4的重建梯度与实际位移方向相反，step5则为下降方向，因此该对照不能预先认定动量是全部失败原因。

配置从原始temporal adapter和全新Adam开始，D任务、seed317、global_requests8、lr1e-4、beta2=0.999、eps1e-8、weight_decay0、BF16执行和FP32master/state均保持原值，仅beta1由0.9改为0。固定原D12日程：12次更新、96次来源使用、89个独立学习来源、7条使用两次。协议SHA256为`f5d6f5f007ca541d8a8ac78f92b84f88486628b81d2872b891f9439794b51ea1`。全部0–12检查点均需报告，不能挑选学习指标最优点，也不将本轮称为独立泛化验证。

首步训练与独立验收产物位于`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-d12-beta1zero-control-prepared-v1`。准备器完成原始资产pins、真实89来源日程和容器挂载可见性检查，相关测试42项通过，包括排除lr、seed、起点、batch和beta2混杂，以及防止当前工作树无关算法进入动量快照。此阶段仅准备首步；步骤2–12尚未生成，没有新容器或后台控制器启动，现有资源队列不变。

为控制源码混杂，快照沿用原D首步训练/验收的src与tools，仅覆盖4个已审阅文件：`intent_optimizer.py`、`intent_resume.py`、`train_intent_finite_npu.py`、`verify_intent_finite_npu.py`。逐文件比较确认其他文件完全一致；完整diff保存在`intent-momentum-control-source-audit-v1.json`。除优化器构造与验证外，训练入口含此前step2已采用的safetensors导出chmod0644，作用是允许宿主读取导出文件，不改变模型数值。该审计覆盖本地src/tools，不声称遍历镜像外部依赖。

前置固定镜像CPU测试已经验证默认行为精确不变、两种beta的精确恢复与下一步一致性、错误beta在状态写入前被拒绝；见`intent-optimizer-control-v1.md`。实际四卡音频路径尚未运行该对照，既有NPU成功不能自动替代新配置验收。后续先运行首步并验收完整178输入HF/vLLM与optimizer契约，再展开独立连续曲线；禁止修改原D/Q/P曲线中的Adam。
