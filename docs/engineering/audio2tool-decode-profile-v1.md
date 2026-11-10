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

## 真实CPU输入预检查已验收

源码`2dee707`的精确容器`fdf7a9a736b0be8b40951908b886f1121c98898d199c7d9527fb8e421511aee1`
退出0，实际执行50.51秒。协调者核验命令、资源、挂载、环境、完整源码与Git、
全部16份attempt/record与原始processor的prompt/audio/长度身份。4个native和12个
Jev都完成真实HF处理与输入token检查；未加载权重、未运行vLLM引擎。
见[CPU独立验收](audio2tool-decode-preflight-v1-verified.json)。
native旧证据采用带空格JSON的token哈希，新receipt为canonical JSON，二者不能
直接比较哈希值；固定运行源码从同一实际ID数组核对旧哈希并生成新收据。独立审计
明确保留这一证据边界，未声称仅凭两种不同哈希能复算输入ID。
NPU profile尚未派发，实际解码耗时及全量执行预算仍未取得。

## 完整解码保存结果的验收

新增`audio2tool_decode_acceptance.audit_records`，面向后续同一已验收源码的NPU
profile。它逐项核对16个有序案例与真实CPU预检查的prompt、音频收据、输入token
身份，要求attempt/单条结果/汇总结果一致，拒绝缺项、额外案例和源身份改变。
每条原始输出通过外部传入的独立tokenizer解码再次比对，检查1024上限、自然停止
和截断标志；非法JSON与达到上限的结果均原样验收，不按内容成功与否筛选。

重新计算家族计时汇总，并检查载入/输入准备/推理计时不超过总体耗时，保留真实
多模态cache配置观测，仍禁止由边界均值输出全量吞吐预测。该函数仅验收保存的
记录：调用方还必须核验实际Docker终态、固定源码/镜像/模型、两种容量依赖和
tokenizer来源，不能把一套可编辑文件或fixture作为NPU执行证据。

新增16项合成artifact测试，与既有15项profile测试合计31项，覆盖缺项、额外项、
输入/attempt错配、输出token或截断不一致、独立解码差异、伪造汇总/计时、cache
缺失、预检查冒充执行、缺容量及误读目标。合法截断和非法JSON原样保留也有正例。
当前尚无实际NPU完整解码结果，这些测试不代表真实耗时或任务得分。
