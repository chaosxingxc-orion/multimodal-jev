# 原始基座函数的零LoRA增量对照

2026-11-12 UTC。当前D/Q/P使用旧temporal任务adapter作为共同初始化，
其step0分数不能代表原始Qwen3-ASR基座。新增固定推理对照，用同一112个
BF16 LoRA张量的结构，保留A、将全部56个B置零；配置不变，不做训练，
不改现有三臂初始化或加入RL/SFT比较。

plain LoRA的增量为`(alpha/r)*B*A*x`；零B在有限计算下给出零增量。
实际源adapter检查bias=none、dropout=0、无DoRA/RSLoRA、无额外保存模块、
无rank/alpha pattern、rank4/alpha8，仅语言层q/v投影。112张量均BF16
且有限，56个实际因子对在固定合成输入上增量精确零，序列化后逐元素一致。
此证据是局部算子验证，尚不是完整音频模型和vLLM路径的实测等价性。

固定镜像CPU构建容器退出0，无NPU设备。源adapter身份：
`7341430053a9b4f36f5d5b8fe80616889af1857c71187d22b4895b062f557140`；
新对照身份：
`856158fe698107d20806203255ddf6efbe5c323bbe2a0f28a5003fe541ab1ad2`。
导出文件初始0600，已仅将权限改为0644供控制器验证；原内容身份复核不变。
证据见`intent-zero-delta-adapter-cpu-v1-verified.json`。

推理gate复用已验收89条学习音频的D/Q共178输入、BF16、TP1和相同runtime
profile，全候选HF/vLLM概率比较及固定学习评分。全部输入保持，不挑选更
短或已答对样本。成功后需与temporal adapter逐来源比较，区分初始化影响
与奖励设计；在结果出来前不能断言旧adapter造成塌缩，也不自动改RL起点。

控制器PID3631238实际argv匹配，等待真实模型卸载重放控制器PID3628148，
保持所有既有队列顺序，没有立即创建NPU任务。三项队列测试验证600秒等待、
失败/缺失前序停止，以及前序成功文件不能替代真实重放验收。
启动记录见`intent-zero-delta-controller-v1-launch.json`。

共享根为`intent-zero-delta-adapter-cpu-v1`、`intent-zero-delta-gate-v1`和
`intent-zero-delta-controller-v1`，均位于
`/mnt/cephfs/xingchao/experiments/multimodal-jev`。

同轮另对真实模型重放的冻结验收包装器完成10项固定镜像CPU集成测试：
普通验收通过时，错误master/Adam/rank/来源/导出或卸载证据不能得到重放
成功记录；实际执行__main__的普通阶段/重放阶段异常均留下失败终态。
数值gate在该测试中mock，张量比较实际执行；不能替代NPU验收。证据见
`intent-residency-acceptance-integration-cpu-v1-verified.json`。
