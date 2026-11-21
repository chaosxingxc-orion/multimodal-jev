# 四卡常驻评测候选与完整收益验收

当前候选仅在selection上做效率与完整输出一致性验证，不派发TEST/FSC/MINDS。
比较对象固定为Stage2 step634和step64各9226条，共18452条预测。
相同策略/音频/候选定义/marker的prob和logit必须与旧运行时精确一致。

## 结构

四个TP1进程各持有独立语料分片和一个vLLM引擎；各片按输入长度与音频帧数均衡，
每片内部按长度组batch32。输入只读已有独立审核过的processor cache，文件sha与各tensor
身份核验一次，驻留CPU内存。不同检查点只切换原生LoRA/head，切换收据核对112个A/B
张量与head/marker；不存在新checkpoint复用旧预测或旧language KV。

此候选直接在模型owner进程消费自己的驻留输入，省去每macro的外部IPC文件写入、
再读取hash及torch.load。vLLM自身worker传输仍存在，不声称全部传输消失。
每个检查点评测结束后异步CPU核对完整旧预测，NPU可继续下一检查点；四个进程均完成
预测核对和engine关闭后controller才收集，另独立审核完整人口、输出与切换收据。

数值runtime从v94冻结源只读加载；新源码只提供流程编排。既有冻结队列不改动。
新增容量设置通过候选配置传入，不修改v94服务配置。KV/prefix数值修复仍为独立工作，
本候选没有宣称prefix准入或10倍门槛已通过。

## 启动与固定脚本

- `tools.reports.run_resident_evaluation_candidate`：设备lease、source冻结、四卡容器、
  exact docker wait、精简结果收集与分阶段时延。
- `tools/reports/evaluation_resident_runner.py`：四进程并行、单次启动、所有审核排空。
- `tools/reports/evaluation_resident_worker.py`：输入一次加载、模型常驻、策略切换、长度批处理、
  完整预测核对。所有读写地址/卡/内存预算来自配置。
- `tools.reports.audit_resident_evaluation_candidate`：独立核对完整ID、相同策略输出、切换、
  初始化次数和输入加载次数，报告实际收益及缺失口径。

共享模板 `configs/experiments/resident-evaluation-candidate-v1.json`，机器宏放本地。

```bash
PYTHONPATH=src:. python -B -m tools.reports.run_resident_evaluation_candidate \
  --config configs/experiments/resident-evaluation-candidate-v1.json \
  --local /LOCAL/experiment-macros.json
```

长任务通过脱离终端的controller运行，普通监控900秒；故障保留确切容器，不自动重试。
所有热写在/data和/srv，模型/数据/旧运行包只读；无全模型checkpoint复制。

## 有限验收及容量根因

v0.0.102：128条固定selection输入，每checkpoint128条，四卡两检查点共256条。
独立审核全部prob/logit精确，输入加载总数128，引擎初始化4次；四卡都完成真实策略切换。
容器223.28秒、controller225.41秒，其中各引擎启动约180–184秒；每卡两轮推理约12.8–13.3秒。
这是有限验收，**不与旧18452条全量结果计算速度比**。

新发现的调度容量瓶颈：旧服务虽然max_num_seqs=32、token预算131072，KV只分配1GiB，
固定镜像实际显示9344 KV token，4096token最大上下文并发2.28。实际输入约3000token，
配置batch32没有转化为足够的调度容量。v0.0.103只将KV容量提升为16GiB，保留同样
128条/两个策略/四卡，先验证数值，再放行完整输入候选。容量加大不等于启用prefix缓存。

v0.0.103已独立通过：所有256条prob/logit精确，单卡两轮推理比1GiB版本减少约9%–14%。
实际KV token容量149760，4096token上下文最大并发36.56。v0.0.104开始两检查点完整9226条。

batch128容量核算：9226条输入token最小2807、中位2834、P95=2868、最大2953。
128条最大377984token，按当前16GiB/149760token估算KV约40.38GiB（还需block rounding）。
44GiB是可测候选；60GiB不合理——机器初始化仅60.52GiB空闲，当前16GiB配置实测
HBM占用约27.5GiB，存在约11.5GiB的非KV开销，且batch增长会增加激活/工作区。
max_num_seqs、max_num_batched_tokens与KV必须联动；新候选`capacity`可配置batch32/48/64/128，
最多48GiB KV。128的token预算应至少覆盖实际最长批，例如393216；通用4096×128则为524288。
先验证batch64/24GiB和batch128/44GiB的真实峰值、吞吐与精度，不以能分配KV等同于吞吐提升。

用户进一步要求探索256：新增submission宽度独立于scheduler并发。
v0.0.106用scheduler64/196608token/24GiB；v0.0.107用scheduler128/393216token/44GiB。
各运行在同一1024条固定输入上分别实际执行submission32/64/128/256，四卡每片256条。
一次提交256时由原生scheduler按容量分波执行，chunked prefill仍关闭；不能称256个请求
同时驻留或把OOM后的隐式降级冒充batch256。实际前向请求数通过首层attention hook读取，
同时记录NPU allocator峰值。有限性能结果带hook开销，不直接外推完整集合。
本次容量扫描按32→64→128→256顺序单遍执行，包含先后次序与首次算子预热的影响；
因此用于识别容量、实际分波、OOM和显著瓶颈，不能单独证明小幅吞吐优势。
若候选据此进入正式完整对照，仍需等输入/策略的新整组计时和全量输出核验。
控制器 `tools.reports.finish_resident_evaluation_sequence` 按精确pidfd等待v104、独立审核
全量结果、顺序执行上述两个固定候选并审核；任一步失败即停止，不自动重试，不调用模型。

v106已完成并独立审核（固定1024条、四卡）：

| 一次提交请求数 | 最慢worker推理秒数 | 实际模型前向最大请求数 | 最大allocator峰值GiB |
| --- | ---: | ---: | ---: |
| 32 | 33.27 | 31 | 29.26 |
| 64 | 29.56 | 63 | 32.67 |
| 128 | 30.29 | 64 | 32.77 |
| 256 | 29.69 | 64 | 32.76 |

四档完整prob/logit均精确。提交256可运行，scheduler64分波；没有证据表明比提交64更快。
allocator数不等于整卡HBM，结果含hook与顺序预热效应。

v107的scheduler128/44GiB在submission32/64/128时完成，四分片prob/logit均精确；
submission128观察到的实际前向最大请求数只有65–70，不能称已验证真实128并发。
submission256时四个EngineCore均OOM：RoPE的`x*cos + half*sin`需要再分配1.38GiB，
已分配57.13GiB，空闲仅944–966MiB，NPU总容量60.96GiB。失败不是LoRA未加载，
不是磁盘错误；44GiB KV挤占了大批前向的激活/临时张量空间。该候选不准入。
证据：operations/evaluation-campaign-v0.0.107/collected/worker-{0..3}.log及terminal.json。
本轮容量扫描到此收口；下一轮部署候选采用scheduler64/24GiB、submission64，
256仅作为可分波的大队列能力，不宣称有额外吞吐收益。

## 报告口径

v0.0.104全量已完成独立审核：两检查点各9226条，共18452条prob/logit与原输出全部精确一致，
所有阈值判决与Top1保持。输入加载9226次，引擎初始化4次。旧两轮容器1680.34秒，
新容器813.89秒，**同容器边界2.065倍，减少51.56%**。新controller含准备/收集816.24秒，
独立审核4.74秒，合计约820.98秒。旧完整controller时延未知，不宣称端到端10倍。
最大单worker阶段工作时间：输入11.26秒、引擎184.28秒、两轮推理586.98秒、
策略切换1.90秒、输出1.83秒、CPU核对2.41秒、退出13.12秒；并行/嵌套不可直接求和。
完整JSON及Markdown：operations/evaluation-campaign-v0.0.104/{audit.json,report.md}。
v0.0.106/107容量探索只在该全量审核通过后开始。

必须报告：准备、输入manifest校验、输入读取、模型初始化、策略切换、推理、结果写入、
CPU核对、关闭、收集、独立审核。四进程工作秒数之和不能当wall time。
旧两轮容器时长实测合计1680.34秒；旧完整controller准备/收集/审核间隔证据缺失，
所以只能先给同容器边界的实测速度比。完整端到端10倍只有可比完整时间证据才算通过。

正式训练收益/测试集成绩不由本候选产生；这里检验效率提升后预测完全保持。
用户最新决定取消继续追逐10倍的前置门槛，转向末点错误分析和下一轮训练。
后续TEST和迁移仍按既定源域选模、冻结策略与固定阈值评测，不自动恢复历史队列。
