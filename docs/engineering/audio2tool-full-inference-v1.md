# Audio2Tool 全量冻结推理入口与逐条记录

`infer_audio2tool_native.py` 实现完整30733条external_test发布记录的顺序推理，
直接消费真实音频和固定native任务定义，greedy、seed17、1024输出token。
复用已固定32768上下文/7音频/TP2/BF16引擎参数；不存在按任务结果选择提示、
跳过超长样本、转写替代音频或替换模型的分支。

## 输入与调用

完整输入和预算固定SHA，检查各tier分母、顺序、唯一release身份与全部上下文预算。
两组重复输入保留为四条独立release记录，不缓存或复用预测。因此正式生成请求数是
30733，唯一输入数30731；论文及不同分母的比较仍需显式说明。

每次真实调用先用共用接口核验源WAV身份和转换后波形，构造固定prompt，再运行HF
processor（truncation=False），核对实际帧数、有效特征有限性与原预算token数。
vLLM返回后要求prompt token IDs与本次实际HF展开逐位相同；这里只对四个边界案例
另有BF16特征逐元素相等证据，不把全量token一致冒充全量特征一致。

## 完成与失败账本

`audio2tool_inference.Ledger` 用排他创建和fsync写入attempt及terminal record，
正式调用前已落盘输入身份、prompt/音频收据和input IDs哈希。每条release独立存储。
中断后旧目录不能重开运行，同一attempt或terminal record不能覆盖；无自动重放。

成功输出保留原始文本，格式错误、空解码文本和length cutoff不自动修复或过滤。
length必须与1024输出token对应，stop允许提前EOS。引擎异常保存真实失败attempt并
立即终止，尚未尝试的行不填充为伪失败。输入校验失败时进度文件保留当前行，应用
终态记录异常；它不产生已调用engine的假账本。

只有全部唯一attempt/terminal记录存在、身份匹配且无运行异常，才导出完整
`ledger/predictions.jsonl`，每行恰为id/raw_text/error，直接对接现有投影scorer。
格式错误仍是有效运行结果，留给scorer计入失败分母；基础设施中断不能作为完整基线。
完整模型文件在实际engine前后校验，模型、版本、源码、输入、容量依赖写入identity。

## 派发前提与验证范围

真实生成路径要求已登记NPU容量容器的成功Docker终态，核对image、精确外部指定ID、
命令、设备、资源、环境、挂载、固定source manifest和四条实际单token结果。
CPU preflight、运行中/失败/OOM、不同模型/设备/命令、缺失边界结果均拒绝。
Docker inspection由外部协调者实际捕获并固定SHA；文件本身不是独立信任根。
后续派发需重新inspect原容量容器，不能只信任旧文本。

32项相关测试通过，覆盖同长度错序token、未完成/多输出/超长/错误终止、重复调用、
中断/异常不能完成、输入/记录身份修改，以及上述容量依赖反例。容量信封测试明确
使用合成容器与结果，并mock固定plan哈希及prior processor依赖；不算NPU运行证据。

实际本地 `.venv` 执行完整清单预检查通过：30733条、30731唯一输入、最大17253
input tokens。见[全量清单结果](audio2tool-full-population-preflight-v1.json)。
它未解码全量音频或加载模型，真实HF+engine循环、权重前后核验及完整账本导出仍待
真实容器验证。当前NPU容量未完成、全量推理未派发、无Audio2Tool模型成绩。
正式评分还须增加全量生成容器的独立验收，不能仅凭predictions文件存在调用scorer。
