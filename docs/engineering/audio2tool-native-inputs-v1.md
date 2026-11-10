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
[完整产物核验](audio2tool-native-inputs-v1-verified.json)。

下一步需用实际processor重新计算该固定新prompt的全量预算并核验音频处理、NPU
engine容量，再接入评分与完整推理账本。此前32768容量结果使用较早提示，不能
直接作为新prompt已验收的证据。当前无推理派发、权重更新或教师请求。
