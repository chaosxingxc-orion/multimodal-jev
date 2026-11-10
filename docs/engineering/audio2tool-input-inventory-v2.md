# Audio2Tool 无答案输入索引

`prepare_audio2tool_inventory.py` 从固定下载清单和独立 `tools_registry.csv` 生成
可复用输入索引；不打开 query metadata、转写、参考调用或音频波形。
固定 release `f1388da9a3189541ab82adac88824a0661670c43`，核对 registry 的原始
Git blob SHA1，再记录 SHA256；工具定义保留 CSV 顺序、类型签名、描述、默认值及约束。
无基于答案的候选筛选，也不调用官方依赖 gold 的备用 schema builder。

独立工具表有 152 个工具：smart_car 86、smart_home 53、wearables 13；名称/ID 无重复。
文件清单包含 16,843 个 query 目录和 36,419 WAV，与卡片 query 数一致，
WAV 仍比卡片 36,421 少 2 个，原因尚未核定。

| Tier | Query 目录数 | WAV 数 | 索引语义 |
| --- | ---: | ---: | --- |
| 1 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 2 | 3,160 | 6,320 | 每条 2 个独立说话人变体 |
| 3 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 4 | 2,132 | 4,276 | 2,125 条各 2 个、2 条各 3 个、5 条各 4 个变体 |
| 5 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 6 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 7 | 2,146 | 7,834 | 有序多轮对话，不能当作说话人变体 |
| 8 | 821 | 821 | 每条一个混合音频 |

Tier7 按文件名中的数值 turn 排序并保留 user/agent 角色，拒绝重复轮次：
1,445 条 3 段、18 条 4 段、677 条 5 段、6 条 7 段。当前 SFC Omni 运行配置
最多接收 2 段音频，因此不能直接迁移到此 Tier，更不能截断来满足接口。
若其余 tier 每个音频变体独立评测、Tier7 每段完整对话评测一次，则文件清单推导
出 30,731 个请求。这只是未来覆盖计划，不是已执行评测或论文分母认证。

清单还确认 `*_audios/query_*/query_metadata.json` 存在；因此旧 tier loader 有
目录级对应，修正此前“只有平铺 JSON、旧 loader 不兼容”的过强推断。
字段级兼容、成功/失败音频筛选及多轮目标语义仍待单独验证。

产物位于 `/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-input-inventory-v2/`：
`registry.json` 与 `audio-inventory.jsonl`。首次临时脚本生成的 v1 保留；正式工具的
v2 产物哈希与 v1 一致，证明此次改为可复用入口未改变内容。
来源、计数和哈希见 [结果](audio2tool-input-inventory-v2.json)。

音频哈希继承下载清单，没有重新读取或校验所有当前 WAV。
这不是完整 inference request：还需固定 prompt、全部工具的上下文预算、多轮音频
接口和正式评分适配。数据全部保持外部 TEST，不用于训练或教师反馈。

## 独立工具表的文本上下文下界

固定 Omni processor、Transformers 5.14.1 实际渲染和分词，全部 152 个候选：

| 工具描述 | 文本 token | 4096 上下文剩余 | 8192 上下文剩余 | 16384 上下文剩余 |
| --- | ---: | ---: | ---: | ---: |
| 官方客户端式签名和描述 | 3,580 | -508 | 3,588 | 11,780 |
| 加默认值及约束的完整定义 | 6,777 | -3,705 | 391 | 8,583 |

剩余量均已预留 1,024 输出 token，尚未加入音频占位/特征展开及其他任务文字。
此结果不是完整音频 fit：4096 已被文本下界否决；16384 只作为后续完整 processor
检查候选，不能据此承诺所有长音频可用或 NPU 内存足够。
保留完整候选，不使用 gold 选择工具来缩短提示；未来正式 prompt 需另行固定。
此次精简项只是采用官方客户端的定义行格式，并不宣称复原完整历史论文提示。

只读 CPU tokenizer 容器
`0739fd467c71f572384f2e982240e1abcce635a83eea0edd82b90a8554376f11`
退出 0；源码 `ca65890`，2 CPU/8 GiB/300 秒，无网络，未加载权重或音频。
已独立核对终态、源码和输出 prompt 哈希，见 [上下文结果](audio2tool-registry-context-v1.json)。

## 当前 WAV 头部全量复核

固定源码 `9336b06` 的四线程只读扫描在 30.15 秒内检查全部 36,419 WAV：
当前文件大小均等于清单，PCM header 全部可读，均为 24 kHz、单声道、16-bit。
独立复核完整行数、唯一文件名、哈希和 frames/sample_rate 时长计算通过，
见 [头部报告](audio2tool-header-audit-v1.json)。这仍不是全量音频内容 hash 验收。

150 个文件超过 30 秒；本地 Omni 序列化配置写有
`n_samples=4800000`、`sampling_rate=16000`，算术上对应 300 秒。
**后续运行时核验修正**：WhisperFeatureExtractor 构造器根据默认 `chunk_length=30`
重算 `n_samples=480000`。Omni processor 默认 `truncation=False`，故这两个
数值都不是已证明的输入硬上限。下面的 300 秒计数只作时长统计。
32 个文件超过 300 秒：Tier1/3/4/5/6/7 分别为 1/3/1/1/15/11。
最长单段 654.64 秒，Tier7 最长完整对话 798.66 秒。
最长三条已另用 libsndfile 读取 header，并完整计算字节 SHA256 与原清单匹配，
见 [长音频复核](audio2tool-long-audio-v1.json)。未听取内容、未生成转写或读取答案。

正式评测不能自动截断或删除长音频。须验证保留完整波形的特征提取/分块协议和
音频 token 数量，核对其与模型端及 vLLM 的一致性，再确定上下文和资源预算。
未验证前，不能宣称全量 Audio2Tool 输入已可推理；若记录 unsupported，也必须保留
在完整分母与覆盖报告中。分块若改变模型输入协议，必须另行说明，不能声称原生路径等价。

## 最长真实输入的 HF processor 验证

按头部时长确定性选取最长单段与最长 Tier7 对话，不根据答案挑选。完整波形逐一
匹配清单 SHA256，24k→16k 使用既有 polyphase 重采样，验证全部采样数和特征帧。
未调用模型、未产生转写、未打开答案。完整 152 工具定义保持不变。

| 条件 | 音频段数 | 输入 token | 预留 1024 输出后，16384 剩余 | 32768 剩余 |
| --- | ---: | ---: | ---: | ---: |
| 最长单段，654.64 秒 | 1 | 15,289 | 71 | 16,455 |
| 最长对话，798.66 秒 | 5 | 17,206 | -1,846 | 14,538 |

实测特征覆盖全部帧，无截断、非有限值或音频 placeholder 数量不一致。
固定镜像容器 `5fbe35410a3b6984c9122f2e7be08349dc28cd039a99d60abdd73d8d439a57f1`
退出 0，源码 `93c07e3`，2 CPU/16 GiB/900 秒。见
[运行与独立复核](audio2tool-long-processor-v1.json) 和 [源码核查](omni-long-audio-source-v1.json)。

这一结果否定“超过配置 n_samples 就不能处理”的推断，支持下一步验证 32768 上下文。
但两个时长极值不证明全量 token 上界；Tier7 暂用同一 user 输入内保序音频和角色标记，
只验证 processor 能力，不宣称最终业务协议。vLLM 会预补齐到 hop 边界，须另做真实
processor 对比；vLLM STT 配置中的 clip 长度也不能直接套到端到端多模态任务。
尚无长音频模型推理、NPU 内存或 HF/vLLM 一致性结果。

## 后续真实 vLLM processor 对比

已完成相同两例的真实 CPU vLLM 多模态 processor 对比，ModelConfig 为
32768、audio limit=7，未创建 engine 或加载权重。v1 比较器遇到 BF16/FP32
`allclose` 类型不匹配而退出 1；失败保留，见 [v1/v2 记录](audio2tool-vllm-processor-v1-v2.json)。
vLLM processing context 会将浮点特征转换为模型 dtype，v2 明确比较 HF cast-BF16
并另报未量化 FP32 误差，原比较容差 1e-5 未放宽。

| 条件 | vLLM token 数 | 原始 HF 对比 | 相同 hop padding + BF16 对比 |
| --- | ---: | --- | --- |
| 最长单段 | 15,290 | 原始 HF 少 1 帧和 1 token | token 一致，特征最大误差 0 |
| 最长对话 | 17,206 | token 相同，但原始 HF 特征少 1 帧 | token 一致，特征最大误差 0 |

vLLM 将每段波形补零到 160 samples 的倍数；真实语音未裁剪。未量化 FP32 与
BF16 特征最大差异约 0.003906，不能把量化差异冒充音频处理算法差异。
这也证明只检查 token 一致不足以证明音频特征一致。

精确容器 `fa8a704d07def626fd37bf965b1bf33ce8eb4e8b6b88138033db880b63d9cd6e`
退出 0，源码 `8e3a9bc`，2 CPU/20 GiB/900 秒，无网络、无设备，禁用 Ascend plugin
用于 CPU processor 检查。两例处理分别 2.66、7.61 秒；与模型生成时延无关。
终态、源码和采样数/帧数/预算算术已独立核对，见
[v2 结果](audio2tool-vllm-processor-v2-verified.json)。
本结果不回写正在运行的 SFC/SLURP/FSC 输入，不证明当前 verl 数值差异由此引起；
未来 actor/rollout 配对须明确采用相同波形边界及 dtype，并重新验证。

## 全请求清单的上下文预算

固定源码 `524eb1e` 的 `audit_audio2tool_context_population.py` 使用已核验的
36,419 条 WAV 头部、完整无答案索引和同一完整工具定义，检查全部 30,731 个请求。
重采样长度用整数向上取整，再按 160 samples 补齐；音频 token 数调用固定镜像
真实 HF 长度函数，文字用真实 tokenizer。全部 7 种文字布局分别验证特殊音频
token 展开可加性，并与此前两例真实 vLLM processor 的帧数/token 数完全匹配。
Tier7 仍使用原 provisional 保序角色标记布局，没有据此确定最终业务/评分语义。

| 上下文 | 预留 1024 输出后超限请求 |
| --- | ---: |
| 4096 | 30,731 |
| 8192 | 728 |
| 16384 | 11（均为 Tier7） |
| 32768 | 0 |

最大输入 17,206 token；32768 留出输出预算后仍余 14,538。所有变体与对话均保留，
没有根据答案缩减工具候选或删除长输入。这个结果支持将 32768 作为当前布局的
engine 验证候选，不等于全部真实波形已重处理、NPU 显存已满足或质量评测已完成。
完整请求级预算保存在实验目录 `audio2tool-context-population-v2/run/requests.jsonl`。

v1 容器 `8969803df78770ea4f99304cfa5bb096f7ea532e77c37d8e21362cf9d918b12b`
因快照漏带包初始化依赖而导入失败，退出 1；未创建 run、未加载权重。
v2 修正快照包含完整包，精确容器
`c98f54290f9da8e52526b895e847d45629c9242012276b653300d8b327629a0f`
退出 0，2 CPU/8 GiB/600 秒上限、无网络/设备，主体 3.44 秒。6 项本地测试通过。
已独立核对源码哈希、终态、30731 唯一请求、36419 音频覆盖及分 tier 超限算术；
见 [运行与复核记录](audio2tool-context-population-v2-verified.json)。

## Tier7 结构与决策时点补充

后续对全部发布记录的结构核对确认：目录按发布 `id` 对齐，不能使用 `query_idx`
拼路径；4 条对话在最后 user 后还有 agent 音频，14 条含连续 user。
最后用户触发调用的正式协议应使用截至最后 user 的完整前缀，并保留决策后材料与
完整对话分母。原全对话容量审计不是该业务协议的已完成实现。
证据、标注文件读取范围与下一步接口约束见 [Tier7 协议](audio2tool-tier7-protocol-v1.md)。
