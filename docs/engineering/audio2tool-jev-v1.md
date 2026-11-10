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
调用已实现音频加载器。实际HF/vLLM processor上下文检查已完成（见后文）；
尚未接入Jev模型生成或NPU容量实验。

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
现已补充真实processor上下文验证及严格评分实现；冻结Jev推理与native实测配对
仍未完成，不能据此宣布Jev收益或跨任务泛化成功。

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

## 全量上下文预算与真实processor验收

固定镜像中的CPU审计容器
`cb441ebf6e471cbf5c40763894696f7877f776a80c0ce80a97a8886c38bd0b33`
已退出0，主体312.65秒，2 CPU / 20 GiB，无网络、NPU设备和模型权重加载。
固定源码为`130c34a97f7d3346a70ebaece19372b375f534d6`。
6种音频/角色布局×457决策得到2742个真实tokenizer profile，结合全部30733条
原始音频token预算覆盖14044981项输入。每种布局/决策均检查特殊音频token展开。

最大输入为18168 tokens，预留1024输出后的窗口溢出数如下。该输出预留只证明
输入预算适配，不保证1024足以生成任意完整调用列表，正式推理仍须记录截断。

| 上下文窗口 | 超限决策数 |
| --- | ---: |
| 4096 | 14044981 |
| 8192 | 665900 |
| 16384 | 14624 |
| 32768 | 0 |

真实音频处理覆盖三种输入模式×三种决策，以及七段音频边界×三种决策，共12例。
HF/vLLM展开后的token IDs逐项一致，BF16音频特征逐项一致，保留完整音频帧。
独立验收重新检查精确容器终态、镜像/命令/挂载/资源、全部源码与Git commit一致性、
2742个profile及完整输入预算算术、12个独立结果文件与原音频来源绑定。
结果见[上下文验收](audio2tool-jev-context-v1-verified.json)，验收脚本为
`tools/jev/verify_audio2tool_jev_context_result_v1.py`。
没有重复运行processor，也没有据此接受NPU engine显存容量或推理效果。

## 严格Jev评分与native桥接

`audio2tool_jev_score.py`直接解析原始输出，禁止去Markdown围栏、JSON修复、
重复键、非有限值、bool/int混淆、缺参填空与工具名纠正。每条必须有全部457项
输入SHA绑定的终态预测，哪怕该项gold不可用。预测字段固定为
`example_id, decision_id, request_sha256, raw_text, error`；运行失败仍在分母内，
基础设施尚未产出的预测不伪造成失败。参数null只允许出现在合法JSON参数内部。

Noul、参数列表、工具序列分别报告可评分分母与正确数，参数缺失只排除对应列表
exact分母；格式有效率仍覆盖全部输出。参数类型校验目前是对象列表的结构校验，
没有把原始自由文本signature/constraints宣称为已执行的完整JSON Schema验证。

present与absent各自转换为“是否需要工具”的语义正负类，分别统计TP/FP/FN/TN、
正类无效和负类无效数。recall分母包含正类无效，specificity分母包含负类无效；
balanced accuracy平均这两个值。precision在没有有效预测正例时为null；F1把正类
无效算作漏检，同时单独报告全部无效数及valid rate，绝不把负类无效算成正确TN。
汇总和各tier都保留完整分母及两种极性结果，正反一致性不代替语义正确率。

native调用仅从预测的参数列表与工具序列按次数和顺序重组；不能用Noul判断过滤、
补齐或重排调用。有效调用即使与Noul矛盾也照常进行native评分，矛盾单独计数。
某参数列表格式错误或序列次数不符则native重组失败，仍占原生分母。native指标
复用已锁定官方primitive和本地严格JSON协议；缺失附加参数的全量strict exact
保持null，完整25969条子集单列，不冒充论文原样复现或全量exact。

`tools/jev/score_audio2tool_jev.py`固定release/registry/官方metrics哈希，流式要求
原始发布顺序与bank决策顺序，核验全部30733×457以及6623不可评分列表。评分前后
校验输入和预测字节哈希，只在完整成功后写出报告，不支持静默子集。CLI使用
`--release --registry --official-metrics --predictions --output`五个路径参数。
报告明确`actor_execution_verified=false`：尚须未来生成runner与独立验收器证明
实际音频、模板、输出token/raw text和精确容器来源；单独拿到评分不能声称模型成绩。

46项相关测试通过，其中新评分/完整流覆盖30项，覆盖类型差异、重复和顺序、
缺参/null、Noul矛盾不修改native、恒不调用退化、运行失败与不完整预测拒绝。
这些是fixture控制，没有生成外部测试预测、调用教师或据此设计RL权重。

评分器随后修正native错误分类：模型JSON格式错误或调用次数不一致只计格式/重组
失败；只有参数列表或工具序列的原始终态带运行错误时才计native运行错误。Noul
运行失败仍占Jev分母，但不抹掉已有效生成的native调用。新增反例后48项相关测试
通过，包含独立控制构造器在同工具主调用已知、附加参数未知时保留主参数和合法null。

完整数据控制入口为`tools/jev/check_audio2tool_jev_score.py`。三个控制均遍历完整
30733×457，而非先排除缺参记录：恒不调用、全部运行失败、可用标注理想答案。
理想答案直接从native标注独立构造，不调用Jev的answer；未知附加参数仅在内存中
的合成预测使用{}占位，gold保持未知并禁止对该工具列表评分。已知主参数、调用
次数与顺序必须保留。它们只验证评分，不作为模型预测文件写出，也不提供actor或
教师反馈。控制用固定镜像CPU执行，预登记2 CPU/8 GiB/3600秒上限，无网络或设备；
成功须完成三组报告与完整终态，失败保留且不自动重跑。具体派发和验收另存记录。

完整控制容器`1e2ee3aede770bdf5408929cf59ea1fa6031c4cf86b38ab256da2d9d714db823`
已用源码`01cfdc3c99c04b7e5c99c103e122230f07fc330b`派发并实际inspect确认运行。
见[派发身份](audio2tool-jev-score-controls-v1-launched.json)。运行中不算三组控制通过。
另已完成[全量推理工作量审计](audio2tool-jev-workload-v1.md)：完整独立决策的逻辑
输入约982亿tokens，是native的462.7倍；这不是实测NPU计算量或吞吐。
