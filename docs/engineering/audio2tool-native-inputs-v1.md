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
全量请求预算及按tier超限算术独立核验，见[结果](audio2tool-native-context-v1-verified.json)。
这是全量预算加三例真实波形处理，未重处理全体波形、未验证本提示的vLLM engine/
NPU显存或生成质量，也没有完成官方scorer接入与论文复现。
