# 上下文概率策略的原生 LoRA 加载准备

联合快照必须同时绑定基座、LoRA矩阵与执行缩放、决策marker行、共享Transformer head、
processor及输入协议。新增 `context_policy_artifacts.py` 先认证整个目录和全部129个BF16
执行张量，再产生原生PEFT配置及112个未合并LoRA矩阵。实际Qwen快照覆盖56个q/v目标。

源码核验基于固定镜像52ba94e1：Qwen3-ASR的HF mapper将
`model.language_model.*`映射为`language_model.model.*`；q/v由vLLM打包到qkv。
导出保留rank4、alpha8、scaling2，不合并基座，不把head丢进LM词表读出。
候选只接收普通q/v LoRA；异构rank/alpha、DoRA、其他缩放模式未验收即拒绝。

本地30项相关测试通过，包括完整性、错策略/基座、额外文件、processor损坏、链接拒绝、
缩放不支持拒绝及未合并矩阵代数。真实既有Qwen快照独立读取129张量通过。
固定镜像v1的19项测试通过，原生parser测试在pin_memory触发无NPU设备初始化时失败；
证据 `context-native-lora-cpu-v1.json`。v2只在CPU测试关闭native模块PIN_MEMORY，
其余解析、mapper与张量检查不替换。该设置不是NPU生产配置修改。

尚未实现或验收：vLLM engine联合加载、processor绑定、marker/head更新、原生qkv打包
执行、无在途请求的原子策略切换、实际音频HF/vLLM概率一致性、FP32 master/Adam恢复。
CPU parser通过不能替代上述验收，也不证明RL或任务收益。

固定镜像v2 fb6691c2已exit0/nonOOM，20项测试全部通过：真实LoRAModel解析得到56个
映射后的目标，112个BF16矩阵逐一精确相等，rank/scaling与联合契约一致。证据
`context-native-lora-cpu-v2.json`；原RC2 1360文件hash未变。


新增固定策略引擎入口vllm_context_model.py/context_resident_rollout.py及真实音频两阶段
诊断probe_context_engine_audio.py。启动独立head、基座load_weights后恢复marker、
显式结束profiling、请求绑定joint policy与唯一LoRA，完整history只输出prob。
固定镜像7b125587的33tests已通过；包含实际adapter序列化/原生parser，模型构造器和
engine encode尚未执行。真实probe已prepare-only，未占卡。HF/vLLM分进程释放显存，
共同BF16输入，保留原始误差不复用旧parity。源快照LoRA B为零，因此随后还需非零
adapter与实际更新后的概率对照，不能据初始策略通过宣称训练策略一致。


更新态探针补充：同一实际音频计算诊断线性泛函梯度，BF16 head+LoRA配FP32 master/Adam，
真实更新后要求LoRA B和head均有可见BF16变化。保存有序参数身份/源policy/执行参数/master/
Adam，扰动后重载并逐张量核对master/Adam，再重新export联合策略并做HF/vLLM对照。
这不是任务reward或训练收益，不使用selection/TEST标签。更新态NPU入口已prepare-only，
未运行。合成骨干上的两步不中断/恢复继续对照本地11项相关tests已过。


终态收集器collect_context_engine_audio.py已准备：绑定准确container/image/退出码，
校验完整文件集、pins、joint bundle和derived adapter，独立重算输入身份与两音频60维
概率误差；更新态另检查128个可训练张量、FP32 master到BF16执行精确映射及Adam步数。
无新NPU终态，不能报告其通过。为独立读取safetensors，在本地隔离
`.local/verification-python`安装0.6.2，真实已有快照112矩阵精确roundtrip完成，
wheel hash见context-local-adapter-verification-v1.json；训练镜像与系统环境未改变。


首个真实7卡更新探针5618ee6e已终态exit1/nonOOM。HF阶段成功：128可训练张量
FP32 master→BF16 export逐一精确，56个LoRA B及11个head张量发生变化；诊断
prob最大变化0.025915，保存/扰动/恢复后的prob误差0。独立证据
`context-updated-audio-npu7-hf-v1.json`，非任务学习收益。vLLM失败于spawn子进程
未注册Qwen3ASRForContextDecisions，尚未进入模型构造器/权重加载/推理。

修复使用独立context_worker_extension模块在worker扩展导入时注册架构。固定版本
worker_base源码确认该导入发生在worker构造及init_device之前。新鲜Python子进程
测试v1已验证架构注册，但解析API缺model_config参数导致测试失败；v2修正实际
API调用复验中。NPU v2只复制逐hash一致的前次updated policy/adapter/HF参考并
执行vLLM阶段，避免重新更新改变比较对象。原failed容器/快照不修改。

终态collector用只读volumes-from辅助容器读取root-owned snapshot，避免主机用户
权限导致采集失败；不改原文件权限与字节。完整失败证据已保存。
