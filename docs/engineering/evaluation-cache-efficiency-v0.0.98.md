# 四卡评测效率与缓存恢复候选

> 用户已澄清10倍门槛针对完整评测任务组，允许跨任务常驻服务和输入复用；
> 本页下文单轮cold准入的解释已被[完整流程规约](evaluation-campaign-reuse-v1.md)取代。
> 841.58/492.30秒只保留为一轮的历史诊断，不作为整个任务组的总时延基线。

2026-11-19 用户要求在正式TEST/迁移评测之前至少10倍提速，且KV/prefix cache必须支持。
正式selection控制器已在核对PID/start_ticks后SIGSTOP；不再派发下一点。
634点评测已审核，64点容器已结束、收集审核待接续。没有派发TEST/FSC/MINDS。
暂停记录位于operations/evaluation-efficiency-hold-v1.json。

## 数据角色和横向参照

训练40513录音，selection9226为独立开发/选择集，clean TEST12991为最终源域测试。
634步selection Top1=83.22%、集合F1=67.38%；不是TEST成绩。
锁定文献SpeechBrain HuBERT direct官方recipe报告TEST意图87.54%（real+synth）；
其SLU-F1=75.10是含实体的指标，不等于我们的意图候选集合F1。
详见docs/research/07-academic-baseline-lock.md和speechbrain-history-v1.md。
当前训练数据、清洗分母与划分不同，不将83.22%和87.54%作公平排名或算法差值。
项目历史Qwen3-Omni clean TEST意图40.08%为完整JSON协议，格式失败保留分母；
同样不能冒充同协议Jev比较，见docs/research/60-omni-clean-native-baseline.md。

## 已测瓶颈

634点容器841.58秒；全量宏推理492.30秒；73次macro总406.71秒；
实际服务关键路径合计400.42秒，四服务均衡下限估计365.72秒（只是假设下界）。
因此仅修调度不可能提供10倍收益。推理外约349秒，包括HF串行256条、
四实例重新加载和预处理/终态；不是单纯网络服务开销。

物理4–7卡一次快照AICore 83/83/87/29%，HBM约12/64GiB；并非稳态平均。
service batch32、macro128、token上限131072，attention每次最多8请求。
只读processor输入缓存已存在且不允许miss重建；语言KV/prefix复用被关闭。
固定投影128×8 tile循环存在吞吐开销；NPU有限组件探针扩大tile批次最高4.28倍，
但部分形状无收益、另一形状不精确，因此没有准入正式模型，不声称端到端收益。

## 缓存路径缺口

1. context_switch_rollout显式enable_prefix_caching=False。
2. vllm_context_model和context_segmented_worker_candidate拒绝prefix/chunk模式。
3. 分段注意力只接受PrefillNoCache，不写读原生paged KV。
4. 固定镜像vLLM pooling_params.py默认token_classify.skip_reading_prefix_cache=True。
5. Jev head读取所有历史token；KV不含最后层完整隐藏状态，须同步恢复head输入。

新候选context_prefix_history.py为有界的隐藏状态伴随缓存；键包含策略、runtime、
累计token前缀及影响该前缀的音频身份。缺失时明确失败，不能补零或返回旧prob。
context_cached_history_model.py采用原生KV/prefix路径与该伴随缓存；请求显式不跳过
prefix读取，保留native LoRA，cold/warm/reordered四次检查。两条音频有限实验，
不评分、不改变当前冻结runtime、不作为10倍全量实测。

注意实际输入：两条样本均先有36token，再音频，再2764token候选定义。
9226条音频hash全部不同。相同候选定义位于不同音频之后，因果状态不同，不能跨音频
误共享这些KV。重复同音频同策略可以复用；跨策略必须隔离。改变任务定义位置会改变
模型输入及训练语义，不能在效率优化中悄悄重排。支持缓存不保证本数据分布有高命中。

## 验收

CPU8项通过：恢复完整历史、冷/热、策略/runtime/音频隔离、token后缀变化、LRU淘汰、
错误长度和同键隐藏状态变化拒绝。尚非NPU精度准入。

NPU候选必须同时报告：实际computed/scheduled token数、history cache字节、冷/热
logit/prob最大误差、阈值翻转、top1变化、相对旧基座的输出差异、不同音频/策略拒绝误命中。
以后补充四卡常驻服务、并行输入预取、精简结果输出和CPU评分异步化；按原population
测全程wall时间并保持精度门槛。不得以重复同请求的热缓存microbenchmark代替全量独立音频。

10倍门槛的局部参考：9226条selection旧容器841.58秒对应84.16秒；同口径推理492.30秒对应49.23秒。
这两个数字分别是容器/推理边界，**不是已测的控制器准备至收集评分完成的端到端基线**。
正式效率证据需从既有日志还原该完整边界；不能将新完整流程耗时与旧容器耗时混比。
TEST12991条还没有旧版全量实测，不把selection时延当成已测TEST基线。
目前未达到门槛，正式TEST/迁移继续暂停。

## 可执行门槛与下一步

工作树新增 `src/multimodal_jev/evaluation_performance_gate.py`，四个控制器的
validate/check 已接入：`run_committed_selection`、`run_multidevice_evaluation`、
`context_full_transfer_operations`、`context_replica_transfer_operations`。
缺少经哈希绑定的证据即在准备/派发前拒绝。冻结v94控制器不变，仍由原SIGSTOP保持暂停；
本改动仅供新候选发布，不能宣称历史包已自动获得这项保护。

新配置复用 `configs/experiments/evaluation-performance-admission-v1.json` 的
`performance_admission` 字段。本地宏绑定 evidence 路径/hash 和待部署runtime hash。
独立CPU检查命令：

```bash
PYTHONPATH=src:. python -B -m tools.reports.check_evaluation_performance_gate \
  --config configs/experiments/evaluation-performance-admission-v1.json \
  --local .local/config/macros.json
```

证据索引含baseline/candidate/parity三个 `{path, sha256}`；固定脚本读取并重算速度比，
不读取模型、不创建容器、不放宽精度。计时记录须同一完整TRAIN/selection人口、
四卡、同策略、同模板、同输出合同、完整完成量和独立音频计数，包含冷启动与cache构建，
时间边界为准备前至收集和评分完成。parity记录需绑定新旧runtime与策略，并覆盖冷/热、
换音频/定义/策略、重排和淘汰；保持当前精确logit/prob及判决要求。
这些收据必须由实际测量及独立审核产生；JSON字段通过不等于实测完成。
新部署包还须将此runtime hash与实际镜像/源码/配置身份绑定，不能任意填入通过过的hash。

顺序：修复原生paged KV与已对齐数学路径的兼容 → 四卡常驻、宏批调度和输入预取验收
→ 同人口完整性能审核达到10倍 → 冻结runtime与源域选择结果 → clean TEST12991、
FSC native31、MINDS native14正式推理评分。每个数据集比较Stage1与Stage2冻结策略，
统一报告Top1、macro-F1、集合F1、exact-set、NLL、Brier及完整耗时；目标集不调阈值/选模。
FSC/MINDS属于跨数据集同类任务迁移，不称跨任务。

当前缓存v99重复两条音频虽冷/热精确，但对冻结基座最大logit差0.71875、prob差约0.01150，
且启动约175秒，不能准入。新稳定前缀模板属于下一轮训练，不能靠重排本轮checkpoint输入
伪造原协议加速。正式TEST四卡五分钟目标另行报告，不能由selection外推冒充实测。
