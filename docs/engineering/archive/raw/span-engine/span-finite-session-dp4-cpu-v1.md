# 可复用Span训练会话及四进程验证

新增`src/multimodal_jev/span_training_session.py`。会话显式区分accepted、updating、awaiting_publication、awaiting_acceptance、failed；各rank先协商阶段、策略和游标再进入更新。一次optimizer更新后必须发布新导出身份并调用独立验证器，成功后才能更新actor策略身份并接受下一批。异常可能发生在参数已修改之后，因此标记failed，不自动重试或伪装回滚。模型/Adam精确恢复、文件身份与后端验证仍由调用方负责，会话不把一个passed字段当作独立验收。

只在accepted边界导出游标，包含policy、completed_updates、schedule身份和checkpoint身份；恢复要求这些身份匹配。接受导出身份未变化的更新，以保留BF16舍入无变化等真实情况，不要求每步必须换一个哈希。调用方必须所有rank共同进入方法；此组件不解决rank进程崩溃后的集群容错，也尚未接入Ray或NPU资源驻留调度。

7项状态单测通过：未验收不能再更新/保存游标、优化器异常不得重试、错误策略/步数/失败gate不能晋升、rank状态分歧在actor调用前拒绝、相同导出身份允许保留。另在固定镜像的无NPU四Gloo/DDP进程中，用真实verl actor运行8步，与独立全局权重参考比较；resumed分支在第4步后重建模型/Adam/actor并恢复会话游标。

容器`e2a163f3e25800c843e4e4eff21b842996d6fb63b140691695c61aeef2c8471f`实际退出0，源码hash、内部终态和四rank结果已核验。持续驻留与中途恢复的梯度、参数、Adam矩逐元素精确相等，独立参考最大误差2.98e-8；56个旧策略batch和64个未验收二次更新被拒绝。两分支均保留不等请求权重和一整个无有效监督rank。验收回调实际比较参考张量，未只返回虚构成功。

证据见`span-finite-session-dp4-cpu-v1-verified.json`，共享根`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-finite-session-dp4-cpu-v1`。这是可复用状态组件接入合成FP32 CPU actor的验证，尚未验证真实音频/NPU/BF16/FSDP驻留、vLLM热更新或完整Ray trainer；既有D/Q/P冻结快照未改变。
