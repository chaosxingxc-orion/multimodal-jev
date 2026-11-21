# 完整评测流程的时延与复用

用户澄清：10倍是**完整评测流程的总wall time**，不是每次冷启动或每次模型forward都要10倍。
本规约取代此前 `evaluation-cache-efficiency-v0.0.98.md` 中单轮冷启动准入的解释。
当前尚未达到效率门槛，正式TEST/FSC/MINDS仍暂停。

## 权重与对比对象

本轮策略为Stage2 step634，policy ID
`bcb44314a3b695a6271195f28e81a82fadb22deeb61399b072bcedd4beae30f0`。
原生vLLM通过LoRARequest加载28层q/v共56个模块的r4、alpha8适配器，
另加载可训练Transformer决策head与marker。LoRA参与计算，但没有merge进base权重。
需要区分“具有LoRA训练能力”和“物理合并权重”；保存/切换只涉及adapter/head。

比较logit时，两边都是这份Stage2策略，固定同样输入、LoRA、head和marker，
比较旧已验收runtime与新cache候选runtime。原先称“冻结基座”容易误解，
以后写“同策略的冻结运行时”。不是要求训练后模型输出等于原始未训练模型。
探针把LoRARequest传入每次encode；v0.0.101的NPU worker已核验实际加载的112个A/B张量、
缩放、slot和覆盖逐项与保存策略一致，`adapter_verified=true`、`base_weights_merged=false`。
收据：`/srv/multimodal-jev-xingchao/operations/evaluation-cache-v0.0.101/collected/adapter-verification.json`。
该次容器exit0，冷/热输出精确，但对旧同策略运行时仍存在logit差异，尚未准入正式评测。
仅配置enable_lora不算完成加载核验。

## 已测时延：只能支持一轮的部分分解

固定脚本 `tools.reports.report_evaluation_campaign_latency` 读取旧容器inspect、
完整推理result、每macro收据及每replica收据。step634、9226条、四实例：

| 范围 | 实测秒 | 含义与优化方向 |
| --- | ---: | --- |
| 容器总运行 | 841.58 | HF面板、加载四服务、全量推理、关闭等；不含全部控制器准备与外部收集审核 |
| 评测客户端 | 492.30 | 输入读取/绑定、宏推理、验证、写结果 |
| 宏调用累计 | 406.71 | 含传输、调度、服务执行与宏收据；不能全部算纯NPU kernel |
| 客户端中宏调用以外 | 85.59 | 加载、重复校验、输出；旧日志不能继续准确拆开 |
| 容器中客户端以外 | 349.28 | HF、引擎启动、关闭等；不能全部归为加载 |
| 每macro最慢服务时间之和 | 400.42 | 当前并发关键路径代理 |
| 同工作量理想均衡代理 | 364.26 | 假设完美负载均衡，不是实测收益 |

这是嵌套口径，不能逐行相加。准备/打包、逐阶段HF、输入加载、输出、传输和CPU审核
没有完备时间戳，均标unknown，不以文件mtime或猜测补数。
也没有完整10点评测组的实测总时间，不能把841.58×10写成已测基线。

## 新流程与优化优先级

1. 固定完整job清单（checkpoint×dataset、population、模板、输出合同、阈值），启动一次四实例。
   基座模型常驻；同一策略连续完成计划中的可执行评测，然后在四实例静止边界切换LoRA/head。
   选模所依赖的源域selection先完成；TEST/迁移只消费冻结的已选策略，不能提前泄漏选模结果。
2. 每个输入语料的processor结果一次加载并认证，使用有预算的CPU常驻tensor池。
   I/O用配置线程数并行。内存不足按固定shard加载、轮转复用，不能OOM后偷偷裁样本。
   政策无关的输入可跨checkpoint复用；含权重计算的KV/最终hidden/acoustic embedding必须
   按依赖身份隔离。不能把前一LoRA的language KV用于下一LoRA。
3. 服务batch32/48、四个实例并发，macro≥四服务容量；测token预算、长尾和等待时间。
   一个rollout绑定一个实例。长度调度与下一macro预取减少空转，不能用结果缓存绕过新策略推理。
   现有 `BulkContextEngineClient.score_many` 每次torch.save到exchange，再读文件计算hash；
   服务端 `read_rows` 再读文件和torch.load。客户端CPU常驻池只消除原始cache的重复加载，
   尚未消除这个IPC往返。下一版需服务端语料注册/只读共享输入句柄，请求传样本索引与身份，
   支持跨job复用、内存预算和显式释放；仍逐策略计算输出，不能把预测当输入缓存。
4. 结果写入和CPU评分审核与下一job的NPU推理重叠。最终wall time止于全部CPU审核排空。
   复用已认证的不可变依赖；每策略保留必要数值面板，不重复审核整套源训练轨迹。
5. 本轮正确性优先复用已验收数学路径。KV修复独立推进，不再把单条prefix探针当主要效率工作。
   下一轮训练采用稳定定义前置模板；本轮checkpoint不改变模板冒充同协议加速。

总时延按实测timeline计算：首次准备/加载/启动 + 有重叠的job执行关键路径 + 最后收集审核排空。
各阶段记录start/end、工作秒数、job和policy身份；并行工作秒数不等于总wall。
正式准入仍需≥10倍；五分钟完整TEST作为独立服务目标报告，不与任务组总提速混为一个门槛。
仅去掉每轮容器内349.28秒非客户端开销，即便理想地全部消除，相对此单轮也只有
841.58/492.30≈1.71倍；这不是完整任务组的上限，但说明“只做引擎常驻”不能证明10倍。
必须把控制器准备/审核、输入与IPC、宏推理、收集和CPU尾部一起测全，再用总wall验收。

## 本次实现边界

- `evaluation_performance_gate.py` 改为完整campaign准入：逐job核对策略、输入、模板、完整数量
  与输出合同；允许跨job同音频与输入复用；不要求每个job冷启动；精度标准保持不变。
- `evaluation_campaign.py` 实现一次有界并行输入加载、同pool多job、静止切换策略、CPU审核
  与后续NPU执行重叠，以及统一timeline。调用方负责发布、设备租约、source admission和评分。
- `context_macro_evaluation.py` 新增job命名空间避免复用pool时macro日志冲突，加入输入绑定、
  推理和结果写入计时。默认历史行为不变，冻结运行包不修改。
- CPU测试验证真实macro协议复用、加载次数、策略切换和审核重叠；尚未声明NPU常驻任务组上线。
  本次相关CPU回归92项通过；NPU仅完成独立cache/LoRA有限探针。
  需将组件接入新版发布/运行控制器，在匹配完整job清单上验收，才能恢复正式评分。

只保留最小预测、指标、配置、时延、策略身份；共享输入/模型以引用复用，不按job复制归档。
