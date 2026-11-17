# Audio2Tool 全发布记录端到端输入

统一转换器 `prepare_audio2tool_inputs.py` 从已核验的音频索引、发布记录映射、
Tier7前缀及结构投影构造输入；不重新打开含答案的发布JSON。来源投影的生成过程
曾进行标注文件结构审计，不能把整个历史流程称为从未接触标注。

产物 `audio2tool-native-inputs-v1/inputs.jsonl` 共30733条，均为external_test，
保留tier+release_id唯一身份和source_query_idx作为来源。30731个唯一输入，
Tier4的468/469及1534/1535保持独立发布记录；未生成或复用任何预测。
actor总音频引用36417，另有4段决策后agent引用保存在provenance；重复引用仍计入。
全量独立核验单音频映射、Tier7有序前缀/provenance和输入字段集合通过。

## 固定本地调用协议

`audio2tool_input.messages`仅接受明确字段集合的input、按顺序加载的波形及固定
SHA256的独立registry原始字节。它生成完整152工具签名/描述/defaults/constraints，
不接受每样本functions或自行传入的system提示。waveform哈希及重采样仍须由调用方
核验，renderer本身不能证明传入数组就是清单音频。

- Tier1–6：单段原始音频，每个说话人变体独立记录。
- Tier7：截至最后user的完整对话音频，角色/轮次保序，不混入转写。
- Tier8：原始mixed.wav，公共指令要求只关注前景说话人，不拆出干净前景或读取
  background_query/background_tool等答案字段。

统一输出为JSON调用数组，每项仅有name和arguments；保留调用顺序与重复调用，
参数类型与默认值依据完整工具定义。它是明确的项目native任务接口，**不是论文
历史prompt复现，也尚不是Jev决策拆分**。原文工具调用字符串如何与此JSON接入
固定官方scorer及严格类型scorer仍需实现和核验，不自动宣布模型成绩可横向等同。

公共任务文本及三种模式已由源码固定，不依据TEST模型输出选择提示。22项相关测试
验证重复项保留、混合音频原样传递、Tier7边界及转写/目标字段拒绝。结果见
[完整产物核验](../../../audio2tool-native-inputs-v1-verified.json)。

下一步需核验NPU engine容量，再接入评分与完整推理账本。此前32768容量结果使用
较早提示，不能直接作为新prompt已验收的证据；下节是新提示的独立检查。
当前无推理派发、权重更新或教师请求。

## 新固定提示的实际processor检查

`audit_audio2tool_native_context.py`直接调用上述正式renderer，使用完整发布记录及
原音频头部计算帧数和token展开；6种实际文字布局逐一检查特殊token展开可加性。
完整30733条请求身份/顺序与输入清单一致，没有去掉重复记录或超长输入。

| 上下文（预留1024输出） | 超限记录 |
| --- | ---: |
| 4096 | 30733 |
| 8192 | 920 |
| 16384 | 11（全为Tier7） |
| 32768 | 0 |

最大输入17253 token，32768预留输出后余14491。按预测token数从每个模式中确定性
选取最大输入，共3条真实processor前向；不根据标签或模型输出挑选：

| 模式 | release ID | 输入token |
| --- | --- | ---: |
| single_audio | Tier1:3943 | 15322 |
| dialogue_last_user | Tier7:2030 | 17253 |
| foreground_mixture | Tier8:314 | 7035 |

每条全部音频字节SHA与清单匹配，24k→16k polyphase重采样长度核验，逐段补齐到
160-sample边界，真实HF processor使用truncation=False。实际有效帧全部覆盖，
完整input_ids与预算展开序列逐位相同，特征全部有限。未读取标注文件。

源码 `4867129`，固定镜像容器
`fc2a323084806f50314cc221d6eb11655183cf0e2209605ac42954e607e97432`
退出0，2 CPU/20 GiB/900秒上限，无网络/设备；主体13.61秒。终态、源码清单、
全量请求预算及按tier超限算术独立核验，见[结果](../../../audio2tool-native-context-v1-verified.json)。
这是全量预算加三例真实波形处理，未重处理全体波形、未验证本提示的vLLM engine/
NPU显存或生成质量，也没有完成官方scorer接入与论文复现。

## 正式推理共用的音频加载接口

`audio2tool_audio.load_audio` 现在对每段音频验证路径留在数据集目录内、字节数、
SHA256与原24kHz/mono/PCM16 WAV格式。哈希与解码使用同一个字节缓冲区，避免
先验证再重开文件导致验证内容与消费内容不一致。完整解码后使用现有polyphase
重采样，核对预期样本数，再右侧补零到160样本边界；每段保存源身份、转换、补零数、
actor波形哈希与预期帧数。Tier7依次消费包括agent在内的完整已登记决策前缀。

`prepare_request` 绑定固定registry和原renderer，将同一批已验证波形交给vLLM请求，
检查全部音频占位数量。它尚不代表HF/vLLM特征一致、engine容量或生成验收。
31项相关测试通过，覆盖实际WAV重采样/仅右侧补零、角色顺序、文件变化、越界软链接、
registry替换及请求音频对象绑定；测试中的processor为接口替身，不是实际HF验证。

`check_audio2tool_audio_loader.py` 在宿主CPU上对固定全量清单和预算哈希做核验，并按
输入token数确定性选择三种模式各自最大样本，真实加载原始音频。三例帧数与此前
实际HF验收结果全部一致，包括五段最长对话。结果及numpy/scipy/soundfile版本见
[音频加载验证](../../../audio2tool-audio-loader-v1-verified.json)。本次没有加载模型或调用
HF/vLLM engine；宿主验证不冒充固定镜像验证，也未遍历解码全部36419个音频。

## 正式输入的实际 HF/vLLM processor 一致性

在上述共用音频接口与正式renderer上新增 `probe_audio2tool_native_processor.py`，
固定完整30733输入/预算哈希及既有processor资产身份；选择三种模式token最大输入，
另加音频数量最多的七段输入（七段中按token数取最大）。不读取标注、转写或生成答案。
选择器测试覆盖遗漏/重复/乱序、split、模式、帧数与上下文越界反例；其validator使用
替身隔离选择逻辑，实际容器始终调用完整输入校验器。

| 检查项 | release ID | 音频数 | 输入token |
| --- | --- | ---: | ---: |
| 单音频最大 | Tier1:3943 | 1 | 15322 |
| 对话token最大 | Tier7:2030 | 5 | 17253 |
| 混合语音最大 | Tier8:314 | 1 | 7035 |
| 音频数最大 | Tier7:1118 | 7 | 8330 |

四例均完整保留帧数；实际vLLM输出与HF有效特征转BF16后逐元素完全一致，最大误差0，
input_ids逐位相同，并匹配已固定预算。没有将FP32量化差异用作放宽容差的理由。
镜像固定为工程当前sha256:52ba94e…，源码 `983f5c9`，容器
`17889805612e4b2ccca53fac6422f75f4e75c49732c1c1efec2c2b589b193fcb`
退出0；2 CPU/20 GiB/900秒上限，无网络/设备/权重加载。已独立核对源码清单、
实际Docker命令/资源/终态、单项记录、预算和补零算术，见
[验收结果](../../../audio2tool-native-processor-v1-verified.json)。

这是固定四例实际processor验证，不是全量特征对比、NPU engine容量或模型推理。
后续engine预检查应保留全部四例及32768上下文、7音频上限，并登记算力/时间预算；
当前4–7卡的既有基线与队列继续优先，未插队派发新NPU任务。
