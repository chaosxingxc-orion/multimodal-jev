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

## 独立验收与评分入口已实现，待真实运行

`score_audio2tool_native.py` 接受事先固定SHA的完整launch plan与精确actor容器ID，
通过SSH实际inspect，不接受只把离线容器JSON当成当前状态。验证固定镜像、命令、
NPU6/7、16 CPU/128 GiB/8 GiB shm、完整源码清单、冻结参数/版本/模型文件身份及
成功应用终态；当前实现完整生成预算为86400秒加30秒终止宽限。尚未生成或派发
这一全量launch plan，容量入口的一小时预算保持不变。

验收还会重新inspect同一个容量前驱容器，并同时核对actor当时记录的容量inspection。
全量ledger按原输入顺序逐条核验：attempt/record完整集合、输入身份、源WAV哈希/大小、
独立header中的采样数与重采样/右补零算术、输出token数量/终止方式/时长及原始文本。
用实际固定HF tokenizer/template重新构造每条完整prompt及音频展开token，核对其
哈希与输入token数。不会重新解码全量波形；波形内容真实性依赖已核验runtime源码中
对同一字节缓冲区的哈希/解码和成功执行，不能把收据本身说成独立波形重算。

`ledger/predictions.jsonl`必须逐条等于terminal记录的id/raw_text/error投影。
缺行、多行、改变原始输出或错误地修复JSON均拒绝验收。验收保存完整账本文件哈希
映射的摘要、调用耗时总和、截断数和verifier源码身份，再次inspect actor后才打开
固定SHA的scorer-only targets，调用现有`evaluate_projected`及固定官方metric函数。
全量主调用/名称与参数可用子集仍分开报告，未知附加参数保持未知。

46项相关测试通过，涵盖账本/推理/capacity门禁及新增独立验收反例，包括实际
预测投影修改、同时篡改attempt与record的重采样参数、相同长度token错序、缺失/额外
文件、错误length终止和容器状态/设备/挂载差异。标签访问顺序测试将accept替换为
失败函数，确认失败时连target哈希都未读取；它不是一次真实完整容器验收。
当前无Audio2Tool全量模型输出，故该独立验收/评分链尚未端到端执行，测试不计作
学术复现或模型效果证据。

## 输出文本与token绑定

独立验收进一步使用固定tokenizer、skip_special_tokens=True，要求每条记录保存的
output token IDs解码后严格等于raw_text。即使同时修改terminal record和预测投影，
也不能用修复后的JSON冒充原始生成文本。该规则与既有sampling验收的解码约定一致；
未做文本修复或放宽空白比较。若实际vLLM文本与独立解码不一致，将保留失败证据再
定位，不能宣称已完成基线。新增协同修改文本的反例后相关47项测试通过。

## 全量评分侧提示重建已在固定镜像验证

`prompt_expander` 从独立验收入口抽为共用函数，用固定renderer、完整工具registry、
真实HF模板/tokenizer与已验收音频token预算重建每条输入。缓存按模式/角色布局和
音频token长度组织，最多保存64个展开结果，返回列表副本，避免跨样本污染。

`check_audio2tool_acceptance_prompts.py` 在源码 `1a0d6c9` 的固定镜像容器
`7d5f9c1fa9701d0bd5fd9742f22d3f21504973b96508e78c68c05fef7f64fc54`
执行，2 CPU/20 GiB/1800秒上限，无网络/设备，退出0，主体152.62秒。
全部30733条、30731唯一输入、6种布局均匹配固定完整预算，最大17253 input tokens。
四个边界样本的prompt哈希和逐token序列哈希同时匹配此前真实HF/vLLM处理记录；
两组重复输入的context身份相同，但仍保留各自release记录。

已独立核对实际容器命令/资源/终态、完整源码清单、30733行身份/顺序/token计数和
重复输入签名，见[重建验收](audio2tool-acceptance-prompts-v1-verified.json)。
产物为输入context哈希清单，不含模型预测或目标，不是伪造的生成账本。
这是全量真实tokenizer重建加四例既有实际processor对照，未重新计算全部音频特征，
未运行生成或完整actor验收。后续新增输出token解码校验未在此次输入重建中执行，
仍需真实输出验证；运行中原容器的不可变源码快照保持原样。
