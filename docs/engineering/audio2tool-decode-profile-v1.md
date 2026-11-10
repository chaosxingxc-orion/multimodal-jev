# Audio2Tool 原生/Jev 完整解码边界耗时测量

现有NPU容量入口每例只生成一个token，不能据此预算正式JSON生成。新入口
`profile_audio2tool_decode.py`沿用已锁定的四个native输入边界与十二个Jev边界，
分别使用正式定义的任务prompt，固定greedy、seed17、max_tokens1024及自然EOS。
不查标注、不根据输出修改提示，不替代完整30733行或14044981项Jev生成。

按固定顺序一次执行：排序后的4个native边界，再按已验收context记录顺序执行12个
Jev边界。没有另加warmup或自动重复。CPU预检查也实际加载认证波形、执行完整HF
processor并核对输入token哈希、全部音频帧和有限feature；分别尊重旧native与
Jev证据的不同JSON哈希格式。实际NPU结果再要求vLLM输入ID与本次HF逐位相同。

每例先持久保存attempt，再调用engine，保存原始输出、实际输出token、停止原因、
截断标志和耗时。非法JSON保持原样；不以JSON合法性选择或重跑案例。独立token
解码不一致时保留原response并记录失败。实际异常后停止，不补写未尝试样本。
输入处理耗时包含认证加载、重采样、prompt和HF处理；engine耗时单独计时。

需要特别区分prefix cache与多模态processor cache。保持既有ENGINE的prefix cache
关闭，但不假定vLLM默认多模态cache也关闭；实际运行会记录可读取的
mm_processor_cache_gb/type及共享内存单对象上限。读取不到配置时明确记unknown。
同一音频在不同任务请求间可能受到既有缓存影响，不能把这一有序单次测量直接解释
成稳态吞吐。输出自然提前EOS也不等价于完成1024-token压力测试。

汇总只报告各家族边界的输入/输出token、输入处理和engine总耗时、最短/最长请求
耗时及截断数。population_throughput_estimate明确为空；16例按最大输入长度与
音频数量选定，并非全量分布代表样本。后续预算需结合全量输入分布、实际解码长度
与缓存证据；不把假设吞吐或此边界均值冒充全量预测。

实际NPU入口要求外部协调者先捕获成功native及Jev容量容器的最新inspect并固定
证据文件SHA，内部再调用两种独立capacity验收。模型文件执行前后核验，保持原
本地模型与固定镜像。预登记CPU预检查为2 CPU/20 GiB/1800秒加10秒、无网络设备、
模型只读且不加载权重。后续NPU profile预算NPU6/7、16 CPU/128 GiB/8 GiB shm、
7200秒加30秒，需在既有两种容量队列成功后取得共享设备锁并记录精确容器身份；
当前尚未派发NPU profile，不能报告实际解码速度。

15项helper测试通过，覆盖两种历史token哈希、非法JSON/截断原样保留、禁止重复
调用、输入错序/缺帧在engine前拒绝、实际runtime错误、解码不一致保留输出、
preflight不产生推理声明，以及缺项/错序/错误耗时/失败记录禁止汇总。这些引擎
返回值是合成fixture；真实CPU输入预检查和NPU结果另存。
