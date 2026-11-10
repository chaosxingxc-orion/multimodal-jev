# Audio2Tool-Jev：完整工具决策与缺失监督

本协议把完整Audio2Tool native音频请求映射为可逐项消费的Jev决策，保持全部发布
记录及external_test用途，不把外部测试集改名为train。请求、任务库与gold分文件存储。
它是项目语音Jev扩展，不是Audio2Tool论文的历史评测格式，也不是nga原生Tacit执行。

## 参考与边界

参考nga工程 `docker/engineering-base/source/native/anyjev/question.py` 的choice/score/
noul定义，及 `tools/explicit_noul_protocol_v1.py` 的独立true/false说明。
读取时nga HEAD为 `61f244ff9ac0409b6a2a59e4563d4e4e6afdef72`，文件SHA分别为
`72e61e3b846e9efa76cbe5b1f5c8fdc5ffa5e5f9eb86de17bcbae5ffdd4ac368`、
`ca3ae496cd1bd8a4b2d4bfcda140e3fb09c2adb94e39fce8b3e85dbace9d9e0c`。
实现独立编写，没有复制nga代码。其choice字母读出上限为26，因此这里不把152工具
硬塞成原生字母choice。多工具可同时为真，采用独立Noul；没有原始ordinal标签，
不制造score监督。参数列表沿用本工程SFC-Jev的typed_call_list思路。

## 公共任务库与actor输入

任务库只从固定SHA的完整152工具CSV生成。每工具有三个决策：

- `.present`：最终用户请求是否至少需要一次该工具调用；JSON boolean。
- `.absent`：最终用户请求是否不需要该工具调用；JSON boolean。
- `.calls`：该工具全部调用的参数对象列表，保留重复、工具内顺序及嵌套类型；
  无需调用时为 `[]`。对象字段语义来自完整原始signature/defaults/constraints。

另有全局 `tools.sequence`，输出工具名数组，保留跨工具顺序与重复，以便把各工具
参数列表重新组合成native调用序列。其标签采用原始primary后接additional的发布
顺序；该顺序尚未经逐条语音执行时序验证，严格sequence分数仅作为本地标注指标。
这一结构和typed_call_list都是项目扩展，不冒充nga三类基本读出。

共304个Noul、152个参数列表、1个工具序列，即每条457项；任务库对全部样本相同。
请求input只含原native context与task_bank_sha256，不含gold、正确工具筛选或
unavailable_decisions。Tier7完整决策前缀与Tier8原始mixed.wav沿用已验收native输入。
原native请求整体哈希保留来源绑定，包括未传给actor的来源元数据。

`audio2tool_jev_runtime.messages` 验证完整registry重新生成的任务库，选择一个登记
决策，将native system输出契约替换为当前JSON类型，同时保留全部工具说明、完整
音频顺序、角色提示和前景指令。它接受波形但不自行验证其字节来源；正式runner仍须
调用已实现音频加载器。尚未接入实际HF/vLLM生成、上下文容量或NPU实验。

## 缺失标注不是unknown答案

gold保留完整native投影及其来源冲突信息，另显式列出缺少监督的decision IDs。
`answer` 返回 `{status: available, value: ...}` 或
`{status: unavailable, reason: source_additional_arguments_missing, value: null}`。
后者是scorer侧标注状态，不能传给actor、作为拒答类别或作为负例。

例如同一工具主调用参数已知、附加调用参数缺失：present仍为true、absent仍为false，
完整calls列表的监督不可用。已知主参数继续保存在native_target中，没有抹去。
其他不出现的工具仍可有已知空列表；出现的合法 `{"x": null}` 参数也是已知值，
与缺失整个附加调用arguments不同。每条调用的原始字符串与证据等级仍留在原投影。

参数完整的记录必须通过重新组合检查：每工具present/absent与calls一致，工具序列
的次数与所有列表长度相符，逐次取出参数后与原始完整调用做类型严格、有序比较。
bool/int/float/string、嵌套对象、重复调用均保留。参数不完整的记录不伪造完整目标。

## 转换与评测约束

`prepare_audio2tool_jev.py` 固定原输入、投影目标、registry和官方metrics源码SHA；
将公共bank与全部请求、受限权限gold写到共享datasets/audio2tool-jev的内容寻址版本。
每条源记录保持独立身份；两个重复发布输入对不会去重。文件只保存公共bank引用和
compact gold，457项答案由确定性adapter按需展开，避免重复存储千万份工具定义。

总决策规模为30733×457=14044981，不能用少数可用条目冒充全量Jev评测。未来应分别
报告完整Noul/序列分母、参数可评分/不可评分分母、原生任务分数与Noul/call一致性。
152工具带来大量真负例，不能仅凭总体accuracy或“全部不调用”控制宣称有效提升。
奖励或权重只能在已有学习集设计，不能依据Audio2Tool测试分布调整训练规则。

转换程序源码为 `47dd43d`，宿主 `.venv` 单进程，1800秒上限、无模型调用；独立
driver保留进程身份、源码清单、日志与退出码，失败stage不会被静默删除或重跑。
输入协议、类型/重复顺序还原、缺失参数与null区分、矛盾决策拒绝及音频渲染相关
17项测试通过。测试的registry部分使用明确fixture，不替代真实全量转换验收。
完整转换结果另行验收；没有启动1404万项推理、教师请求或权重训练。

## 全量发布已完成并独立验收

共享数据目录：
`/mnt/cephfs/xingchao/datasets/audio2tool-jev/a3a34fdd0da98e6f6c53e9201751993e37202f931a1240a7475bf90593c50235`。
转换driver退出0，主体365.98秒；其精确源码/命令/进程启动身份与终态保存在
`experiments/multimodal-jev/audio2tool-jev-export-v1`。这次是宿主Python数据处理，
没有Docker/NPU或权重加载，不冒充镜像推理验证。

| 项目 | 完整数量 |
| --- | ---: |
| 发布记录 / 唯一输入 | 30733 / 30731 |
| 每条决策 / 总决策 | 457 / 14044981 |
| 可评分Noul | 9342832 |
| 可评分工具参数列表 | 4664793 |
| 参数列表监督不可用 | 6623 |
| 可评分工具序列 | 30733 |
| 参数完整 / 不完整记录 | 25969 / 4764 |

6623是不可用的工具参数列表决策数，4764是包含至少一个缺失参数调用的源记录数，
两者不能混用。所有不完整记录仍保留，所有可评分判断仍有完整分母。

`verify_audio2tool_jev.py` 流式读取全部四份源/派生文件，独立逐行比较身份、顺序、
原始音频context、完整native_target、源请求哈希和派生gold绑定；重新计算不可用
工具列表、所有tier分母、决策覆盖、30731唯一输入和完整序列，未调用转换函数来
生成预期请求或gold。公共bank单独从固定registry核验。验收结果及SHA见
[全量发布验收](audio2tool-jev-v1-verified.json)。gold文件权限为0600。

请求文件约20.8 MB、compact gold约26.9 MB、公共bank约271 KB；原WAV只引用共享
native数据，不复制音频。后续单决策adapter另提交于 `a32e963`，没有修改该数据版本。
尚需真实processor上下文验证、冻结Jev推理与native配对评分，当前只完成数据发布，
不能据此宣布Jev收益或跨任务泛化成功。

## 完整标注上的恒不调用退化控制

固定控制为每工具present=false、absent=true、calls=[]，全局sequence=[]。
这是无需模型的解析基线，不写伪预测文件，不给actor/教师反馈，不估计训练权重。
完整30733条标注含37723个“样本×需要工具”组合和4633693个不需要工具组合。
重复调用同一工具仍只计一个存在性正例，调用次数由参数列表和sequence另行约束。

| 指标 | 恒不调用控制 |
| --- | ---: |
| 汇总Noul accuracy（含正反问） | 99.1925% |
| 可评分参数列表exact | 99.3333% |
| 必需工具召回 / F1 | 0 / 0 |
| presence balanced accuracy | 0.5 |
| 完整工具序列exact | 0 |
| 原生主工具accuracy | 0 |
| 完整参数25969条子集exact | 0 |

这里工具参数列表exact分母为4664793，保留6623个不可用列表的独立计数；正反Noul
都预测正确的大量负工具能抬高汇总分数。将正问和反问混在一起后，即使原始boolean
标签数量平衡，也不能消除“所有工具都不需要”的语义退化。须单独报告present侧
必需工具召回/精确率、正反一致性、native完整任务指标与覆盖，不用总体accuracy
替代业务效果。未来RL权重仍只能从学习集确定，不能依据本external_test频次调参。
全量解析结果见[退化控制](audio2tool-jev-absence-control-v1.json)，两个反例测试覆盖
重复缺参调用的存在性计数及未知注册工具拒绝；它们不是模型成绩。
