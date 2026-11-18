# 语音 Jev 变长组批：开源实现与落地方案

日期：2026-11-14 UTC。结论：采用长度感知组批，global128保持；当前已派发配置为
DP4/micro8/累积4，同一global batch内分组；全量epoch分桶和按桶动态micro仍待验证。
micro32已在真实反向探针中发生OOM，不能作为默认值。本文件区分源码调研、离线估算
和实际运行证据；本次文档复核没有修改已派发配置、镜像或rc2。

## 调研结论与配置决策

开源实现支持“相近长度组批 + 按批padding + 长度预算”的组合，分桶不要求将音频
切成片段，也不要求全部补到桶上界。SpeechBrain适合显式桶界与预算，Lhotse还提供
跨卡桶协调；verl提供token预算及DP计算量平衡，接入时需要本项目真实多模态长度。
本次复核来源锁定中的7个本地上游源码文件，SHA256全部匹配，未更新依赖。

建议保持global128，先采用micro8，实测micro16作为容量/吞吐对照；micro32在既有
双卡真实反向探针中已发生设备OOM，不能列为默认值，也不能仅由0.6B/LoRA推定可用。
四个actor rank时，micro8/16分别对应每次更新累积4/2轮。micro8已通过DP4前缀、精确
恢复及相同checkpoint的后端验证并派发全量Stage1；micro16不是本轮运行配置。
前缀计时不代表稳态吞吐，分桶也不保证micro32能够通过。

先在保持全局batch成员和权重的前提下重排micro，再试全epoch随机桶。
全epoch桶必须核查各桶类别分布和更新顺序，避免长度与类别相关造成连续同类更新；
沿用全量样本和已定义权重，不为追求桶内均衡擅自过采样。固定shape档位另行实测，
不要把减少编译次数和减少padding当成同一个优化。

共同候选profile的HF/vLLM已在新DP4 step4 checkpoint上完成256条完整概率核验，
max logprob差9.536743e-7、argmax变化0；精确恢复后单条/micro8对照也已通过。
证据见[step4独立验收](slurp-class-spread-batch-dp4-resume-parity-audit-v1-accepted.json)。
早期候选对原HF最大差0.4580321的失败证据仍保留；新的共同profile验收不表示与原
eager实现等价，也不覆盖未来checkpoint、动态micro或vLLM多请求组批。
分桶策略、micro容量及同参数训推一致性分开验收；后文运行日志为历史快照。

## 概念与适用边界

| 方法 | 实际改变 | 本项目意义 |
| --- | --- | --- |
| 长度分桶 bucketing | 长度相近样本进入同一候选池，桶内/桶间按seed打乱 | 降低组内最长长度及padding；不切断音频、不改任务 |
| 按批动态padding | 每批补到当前最长值，或少量固定shape上界 | 已有批内右补齐；固定上界可能增加padding，须实测编译/内核收益 |
| token/frame预算动态microbatch | 短样本多装、长样本少装，用梯度累积维持global有效请求数 | 控制激活显存，不能只依据0.6B参数数目 |
| packing/varlen attention | 有效token紧凑排列，用累计长度隔离样本注意力 | 与分桶互补；样本边界、position、音频映射和Span位置必须保留 |
| vLLM连续批处理 | 按在线请求状态、token与encoder预算调度prefill/decode | 属于推理调度，不是训练集分桶；需另测共同数值profile |

“分段序列长度”指bucket边界或shape档位；不意味着把一个语音query切成多个训练标签。
分类任务默认保留完整音频和完整任务定义，超过预算转入小micro/单独长样本档，不截断或丢弃。

## 开源源码核查

已抓取下面各仓库的具体commit并保存SHA256，见
[来源锁定](variable-length-source-lock-v1.json)。只读研究，不安装或升级主线镜像。

1. **SpeechBrain DynamicBatchSampler**：用户可给`bucket_boundaries`和`lengths_list`；
   每桶条数为`min(max_batch_ex, max(1, floor(max_batch_length / bucket_upper_bound)))`，
   尾部超界桶为1。支持epoch/seed打乱。默认桶界是缩放的lognormal分位，不能误称数据经验分位。
   [源码](https://github.com/speechbrain/speechbrain/blob/c0241278f93ef02cd5191f03caf4e5f8e1292bf6/speechbrain/dataio/sampler.py#L514)。
2. **Lhotse DynamicBucketingSampler**：支持`duration_bins`、`num_buckets`、`max_duration`、
   `max_cuts`、shuffle buffer、`quadratic_duration`及`sync_buckets`。
   自动边界按排序后的长度累计总量近似均分，不是简单等样本数分位；`sync_buckets`让多卡
   尽量抽相近时长桶减少慢卡尾部等待。`FixedBucketBatchSizeConstraint`还可明确每档batch size。
   [接口](https://github.com/lhotse-speech/lhotse/blob/3a4f0971f4f05c3d012e36b85097ddb5cb235c33/lhotse/dataset/sampling/dynamic_bucketing.py#L94)，
   [自动边界](https://github.com/lhotse-speech/lhotse/blob/3a4f0971f4f05c3d012e36b85097ddb5cb235c33/lhotse/dataset/sampling/dynamic_bucketing.py#L511)。
3. **verl**：`get_seqlen_balanced_partitions`平衡DP分区；`rearrange_micro_batches`按有效
   token预算划分微批，支持DP统一micro步数并返回逆置换所需索引。它是计算量平衡，
   不等于把相近长度放在一起；在仍使用dense padding时，长短混配可能抵消分桶收益。
   [上游实现](https://github.com/verl-project/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/verl/utils/seqlen_balancing.py#L348)。
   **实际镜像**`/opt/verl/verl`版本另行读取/hash：工作量代理为`24576*L+L²`，其常数为
   7B hidden4096量级，不能直接当0.6B语音模型成本。已安装版先用token总量估微批数，
   再按工作量分区，没有本次上游commit中逐分区硬token预算重检循环，因此仍须自查容量。
   上游原`workers/actor/dp_actor.py`地址已404，不把最新接口当成已安装接口。
4. **Qwen3-ASR官方SFT示例**：collator调用`processor(... padding=True, truncation=False)`，
   示例TrainingArguments设置固定batch与梯度累积，没有显式开启长度分桶。
   因此官方示例能变长补齐，不代表已提供适合本项目的高效bucket sampler。
   [源码](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/qwen3_asr_sft.py#L115)。
   **实际镜像的Qwen ASR encoder**：先对`B*num_chunks`执行CNN和投影，再select有效post-CNN
   位置进入packed音频attention（源文件382–429行）。全零补齐块仍消耗CNN，encoder attention
   则已有去padding机制。不能按原始波形补齐比例等比推算总训练成本。
5. **vLLM V1 scheduler**：调度有token budget、最大批token和多模态encoder预算；有chunked
   prefill路径。它解决在线推理吞吐，不直接决定训练epoch sampler。
   [源码](https://github.com/vllm-project/vllm/blob/fee9686811b7c060c404dba0c6a5a6007de4659b/vllm/v1/core/sched/scheduler.py#L711)。
   本项目当前通过的vLLM profile限单条完整prefill，不能因为引擎支持就启用连续组批而继承验收。

## 全量SLURP测量与离线模拟

复用已验收的40513条learning processor记录，两epoch共81026条真实使用记录。
重新核验records SHA、schedule manifest SHA和两个epoch文件SHA、每条坐标/ID及跨epoch长度一致。
不使用selection/test选择bucket。音频时长为已处理16kHz波形样本数/16000。

| 分位 | 音频秒数 | 含音频占位及任务定义的模型输入token |
| --- | ---: | ---: |
| min | 0.512 | 1952 |
| P25 | 1.984 | 1971 |
| P50 | 2.624 | 1979 |
| P75 | 3.456 | 1990 |
| P90 | 4.480 | 2003 |
| P95 | 5.248 | 2013 |
| P99 | 7.232 | 2039 |
| max | 23.360 | 2249 |

当前固定60意图菜单使总token分布很窄；常见512/1024/2048/4096档中几乎全落2048，
无法有效区分音频开销。应同时记录有效音频帧F、100帧chunk对齐后的宽度、语言长度L、
音频attention窗口长度，以及任务模板ID。音频和token长度不能互相替代。

保持原global128成员、原始权重及每epoch完整覆盖，仅模拟内部重排：按(L,samples,ID)
排序，连续micro组按轮次蛇形分配四卡，每卡32条。下表为所有634个更新的总量统计；
包括实际执行的末批零权重padding副本。不是正式sampler，不是NPU速度测量。
这里使用原日程`fc36b3af…`；当前class-spread日程为`8c979d47…`。学习人口相同，
因此长度分布和下文桶人数可复用；global成员组合已经改变，表中padding改善不能
直接作为当前日程的实测值，需对新日程重新计算或采集运行指标。

| micro | 原顺序token padding | global内分组后 | 原波形补齐代理 | 分组后代理 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1.387% | 0.177% | 42.78% | 8.64% |
| 16 | 1.901% | 0.397% | 50.75% | 17.50% |
| 32 | 2.457% | 0.856% | 57.25% | 31.47% |

token padding = 1−ΣL/(Σ每micro的B*maxL)。波形代理类似，但不是精确feature padding或FLOPs。
只在原rank内排序时micro32完全不变；它无法改变该rank唯一micro的max长度。
全global排序的micro32把长样本集中一张卡，dense注意力代理的最慢卡/均卡平均比从1.026
升到1.060，因此“最少padding”并不保证“最快多卡更新”。必须联合考虑跨卡负载。
当前候选语言attention已经varlen，只在attention节省padding与MLP/投影/CNN仍有padding
的成本不同，也不能把L²代理视为当前真实kernel时间。

结果：[全量分析JSON](slurp-length-bucketing-analysis-v1.json)。
复现：`python3 tools/jev/analyze_slurp_length_bucketing.py`（默认读取.local原始已验收记录）。

## 本项目落地顺序

**A：先保持global128的成员和学习分布，优化内部排列。**

- 以原全量epoch日程选出128条，再按真实长度生成micro组并分配四卡。
- 分组目标同时约束padding、每卡总预算和同步micro步数；固定micro8/16/32各作对照。
- 每条request的ID、标签、权重、行为概率、参考概率、G16采样动作和advantage随同一个
  permutation一起搬移；不得只排序输入tensor。处理后可逆置换返回原评测顺序。
- 保持全局真实请求数分母；每卡分类不一定严格均衡，但原global batch类别构成和权重完全不变。
- 这一步可以和真实反向容量测试合并实施，不需先重做全量标签或新增训练数据集。

**B：容量与成本测准后，再定义全量epoch的随机长度桶。**

初始可研究音频`≤2s / 2–3.5s / 3.5–5.5s / 5.5–8s / >8s`，边界依据上述learning
分布作起点，不是已经测出的最优阈值。用实际processor帧宽转换执行档位；保留溢出桶。
按epoch0的40513条唯一learning记录复算，各候选桶如下：

| 音频时长桶 | 真实请求数 | 占比 |
| --- | ---: | ---: |
| ≤2秒 | 10549 | 26.04% |
| (2, 3.5]秒 | 20401 | 50.36% |
| (3.5, 5.5]秒 | 7891 | 19.48% |
| (5.5, 8]秒 | 1453 | 3.59% |
| >8秒 | 219 | 0.54% |

桶是sampler的索引集合，无需复制音频或生成五套数据集。预处理若改变语速、裁剪静音，
应按变换后的帧长组批，或采用可靠上界并在collate时检查容量，不能沿用失效长度。
不先给每个桶写死未经反向测量的micro值：候选为32/16/8，容量验证后形成配置表。
DP4且各rank每轮micro一致时，global128对应micro32×累积1、micro16×累积2、
micro8×累积4；混合大小时按实际有效请求数归一化，不对各micro均值直接等权平均。

长样本桶可采用更小micro；四卡同一步协调档位与micro次数，保持global128。
桶内随机、桶顺序随机、seed/epoch可重放；全epoch不drop、不额外过采样某桶。
若采用固定shape档，padding到桶上界而非当前最长会增加计算，应对比动态图和编译缓存收益。
固定shape不是分桶的必要条件，也不改模型context limit/RoPE。

### 桶界、执行长度和batch预算分别配置

建议将配置分成三层，避免一个`max_length`同时控制采样、padding与截断：

1. `duration_bins_seconds=[2,3.5,5.5,8]`只定义候选池，尾桶开放；精确开闭区间
   遵循所用sampler并写入manifest。上述人数采用右闭区间，换成Lhotse左闭区间需重算。
2. 每条记录保存增强后的`audio_frames`、实际chunk对齐宽度、`input_tokens`和模板版本。
   collator默认补到本micro实际最大长度；固定shape档位单独配置并以实测决定。
3. 预算同时限制音频侧和语言侧，并设置`max_examples_per_micro`。dense分支需考虑
   `B*max(F)`和`B*max(L)`，有效token总和不足以约束其padding内存；packed分支则记录
   有效长度总和、最大单条长度及实际内核限制。预算数值来自真实反向测量，不从参数量推算。

一个micro8组若有7条2秒和1条8秒音频，波形dense padding代理为
`1-(7*2+8)/(8*8)=65.625%`；同长组该代理接近0。这里只解释组批收益机制，
本模型encoder包含chunk和去padding路径，因此不是实际训练节时预测。

每epoch先打乱桶内索引、生成批列表，再打乱批列表或按剩余样本调度，直到每条真实记录
恰好消费一次。不要按固定步数等概率抽五个桶并有放回取样，否则会放大仅219条的长尾桶。
各桶末尾统一合并/回填或显式零权重补齐；采样器状态保存epoch、seed、桶内游标、剩余批
顺序及长度元数据版本，以便恢复时重放。全epoch分桶与现有类别时序分散日程需要共同设计，
不能承诺每个长度桶内仍然具备全局类别比例。

该阶段会改变global batch成员及类别共现，必须发布新schedule版本，比较每桶类别分布，
保留全量class weighting。不能声称和原schedule逐步等价。稀少长尾桶应合并/跨桶回填或
显式零权重补齐；补齐不计真实分母，不能把尾部数据直接丢掉。

**C：变长micro接入verl及训推一致性。**

当前`span_hybrid_actor.make_batch`外层`input_ids/attention_mask`是长度1占位，真实序列
在`multi_modal_inputs`中；原生动态分批会误判所有样本一样短。当前builder明确拒绝
`use_dynamic_bsz`，不能只改开关。要从真实输入提取长度并纳入可审计metadata。

已安装verl动态路径按`micro实际条数/local mini batch条数`缩放loss；现有Jev注册loss
则乘固定`local_batch//micro`抵消静态梯度累积除法。两者直接拼接会改变loss权重。
接动态路径时必须只做一次全局归一化：各micro累积贡献的和应等于
`Σ真实请求 w_i*(RL_i+λCE_i+βKL_i)/Nreal`；DP平均梯度补偿保持一致。
GRPO/RLOO每请求G16作为完整组保留，先计算组内advantage再分配；本模型同请求一次
读出60类概率，16次类别采样不是16份音频，不按128×16音频预算估显存。

FSDP各rank需要一致的collective顺序及受控micro次数，不能各卡独立耗尽不同数量微批。
验收覆盖概率、加权梯度、Adam更新、末批padding、恢复与候选actor/vLLM共同profile。
吞吐以预热后真实global128更新时间、有效音频秒/秒、有效token/秒、峰值allocated/reserved、
各rank等待为准。分桶可优化计算形状，但不自动修复BF16批形状概率偏移。

## 下一实验矩阵

同一冻结起点、同global128成员：micro8/16/32 × 原顺序/长度感知分组，先通过数值再
测完整更新；固定BF16/LoRA/奖励/权重/全量人口。端到端计时包括collation、搬运、前向、
反向、同步、Adam，checkpoint与评测开销另列。取得成本后才决定是否继续做全epoch桶和
按token/frame预算的动态micro。此前单次前向micro8最快，只是候选，不预定最终结果。


## 候选执行进展（2026-11-14）

已新增独立`span_length_groups`索引规划器（原顺序/长度相近组+轮次蛇形分rank），
保持固定global成员和每rank条数，调用方移动整个请求载荷。14项测试覆盖DP1/4/8、
micro1/8/16、完整permutation及可逆载荷，连同packed attention共21项通过。
这仍不是正式dynamic-bsz sampler，也未声称解决micro32四卡长尾负载。

新增单rank FSDP/真实verl更新探针`probe_span_batched_update_npu.py`：从已认证
learning256按帧长度分位取32条、固定seed打乱，绑定官方clean标签及全量class weight，
固定BF16执行/FP32 master Adam，同起点重复比较micro1/8/16/32及长度分组。
保存原始112 LoRA梯度/master/Adam和前向概率，独立collector复算。
计时包含CPU概率/梯度取证，不是纯训练稳态测量；单rank不是DP4/global128验收。

v1 a829027a已确切退出1/non-OOM，在训练前标签join发现日程复合ID不能直接作录音主键；
失败保留。v2修正为音频路径/哈希及复合ID前缀绑定，1ebf8a54于03:39:39 UTC
退出1/non-OOM，独立收据核验320个固定文件。首个训练forward在FSDP恢复共享参数视图时
触发`as_params=True type(prim_param)=<class 'torch.Tensor'>`，没有完成optimizer更新。
日志显示单rank使FULL_SHARD切换为NO_SHARD；共享权重与该路径的具体因果仍待隔离，
不能据此认定分桶算法有误或多卡训练也必然失败。
见[终态收据](slurp-true-batch-update-npu5-v2.json)。反向容量和训练加速尚未验证；
原生产/rc2原1360文件未变。

### FSDP更新探针的隔离结果

v3 `ae277061`退出1/non-OOM，记录证实异常来自共享的
`language_model.embed_tokens.weight`/`lm_head.weight`，恢复后primary仍为Tensor。
v4 `464f2de9`把参考概率计算移到FSDP包装前，仍退出1/non-OOM，因此不能将先做参考
推理认定为必要触发条件。两者均没有完成optimizer更新，独立收据各核验320固定文件。

新增[最小CPU复现](../../tools/jev/diagnose_fsdp_frozen_shared_cpu.py)，不依赖Qwen、verl、
NPU或本项目候选kernel：冻结embedding/head，单独FSDP包装可训练Linear，执行两次Adam。
PyTorch 2.10.0下，共享权重DP1第一次更新成功、第二次出现同一断言；非共享DP1以及
共享/非共享DP2均完成两次更新。控制实验支持单rank NO_SHARD冻结共享权重生命周期
问题，不能据此宣称已修复PyTorch或完成NPU训练验收。
见[CPU对照收据](fsdp-frozen-shared-cpu-v1.json)。

真实双卡候选`1e6f2197`已在175物理5、6卡启动，独立快照，64条learning/每rank32条，
固定全局成员、class weight，比较micro1/8/16/32及分组，保存各rank梯度/master/Adam。
使用真实FULL_SHARD，未解开模型共享权重或绕过断言；等待确切终态及独立收集。
这仍非正式DP4/global128、GRPO/RLOO、稳态吞吐或跨后端一致性验收。

双卡v1终态更正：1e6f2197于04:02:46 UTC退出1/non-OOM，torchrun standalone的
TCPStore尝试连接容器hostname超时，尚未启动训练worker，也没有梯度或更新结果。
v2启动器改用同容器显式127.0.0.1静态rendezvous，不改模型或计算配置。

### 类别时序与长度桶必须一起审核

完整Stage1评测准确率20.2038%，但89.96%预测集中于calendar_set/play_music。
learning-only审计发现原epoch日程按类别轮流无放回取样，小类提前耗尽：每轮最后8批
仅剩这两类，最后65个真实样本全部calendar_set。此为确定的日程分布漂移，尚不能
证明它是效果不足的唯一原因。原冻结日程不修改，见
[日程审计](slurp-stage1-schedule-class-drift-v1.json)。

候选`span_class_spread`按各类normalized progress和seed相位将原真实样本均匀分散到
全epoch，保留每类内部原随机顺序、全量ID和权重；7项测试通过。全量两epoch离线对照
最后8批覆盖59类，末批分别42/40类，见[候选复算](slurp-class-spread-candidate-v1.json)。
尚未发布新训练日程；后续长度组批应先保证这种时间覆盖，再在每个global128内优化长度。
全epoch长度桶也应检查各桶类别和时间分布，避免重新引入类集中或短到长的固定课程。

### 真实双卡容量结果与新日程（2026-11-14）

双卡v2 `7cfcdc40`完成micro1、8原顺序/分组、16原顺序/分组各两次更新，每rank共10次，
micro32首次forward设备OOM，容器于04:09:25退出1（Docker OOMKilled=false，不能据此
否认框架报告的NPU显存不足）。独立collector核验321固定文件及20份梯度/master/Adam；
20组前向概率均通过，最大logprob差9.54e-7。

global64更新耗时micro1为67.6–70.4秒，micro8原顺序12.5–13.9秒、分组13.1–13.7秒；
micro16原顺序13.2–15.7秒、分组13.1–13.2秒。micro8峰值allocated约28GiB，
micro16约52GiB。计时包括取证和同步，仅两次重置实验，不能冒称稳态速度或分桶优势。
相同micro1重复的梯度relative-L2约0.00495、master最大差约2e-4，故尚不准入。
v3 `2fe649ea`启用torch deterministic算法和HCCL_DETERMINISTIC=true，对micro1/8
原顺序及分组做同起点重复，待终态；不放宽概率或梯度门槛。
详见[容量收据](slurp-true-batch-dp2-capacity-v1.json)。

新类别分散日程位于`.local/simple-final-v1/schedule-class-spread-v1`，manifest SHA
`8c979d47bd77f7fa3391484a20cad127a372c7acac3fc51d84e13e66a91fb682`。
既有SimpleRLSchedule独立复核两epoch/40513条/634更新/每轮63个零权重补齐及全量
来源组、类别权重与父版本一致；处理器坐标预检仍待做，未派发训练。
生成入口`tools/jev/prepare_class_spread_epochs.py`，见
[日程收据](slurp-class-spread-schedule-v1.json)。

最终Stage1的已认证learning256以原单条HF profile推理成功：26/256正确、NLL6.9385，
60类目标只预测13类，232/256预测两大类，身份/输入/目标映射独立核验。说明集中现象
不只存在于selection；小样本仍不能替代全训练集指标或证明唯一根因。固定G16独立采样
的预期全错组比例77.19%，混合组19.31%，均为概率推算而非实际RL观测。
见[学习侧诊断](slurp-stage1-learning256-diagnostic-v1.json)。

确定性v3 `2fe649ea`已成功退出0；独立收集并比较每个rank/case两次保存张量，
micro1原序、micro8原序、micro8分组的梯度/master/Adam全部逐值一致。
micro1与micro8之间仍有relative-L2约0.0038–0.0042的梯度差及最大约2e-4的master差，
因此只能确认可重复，不能宣称跨micro训练等价。v4 `a3537b09`增加每次更新后相同
canonical32/rank概率采集，等待原门槛比较。见[确定性收据](slurp-true-batch-determinism-v1.json)。

新日程CPU真实processor预检`f9af7340`已启动，全部40513条音频、两epoch81026次使用，
重新生成坐标账本；没有把旧坐标账本直接改名当作新执行证据。当前训练准入仍false。

### 更新轨迹与同一checkpoint训推一致性的区分

双卡v4 `a3537b09`于2026-11-14 04:28:26 UTC成功退出；独立核验321项依赖、
12份梯度/master/Adam以及两卡各32条更新后完整60类概率。相同micro1重复更新后的
概率完全一致；micro8原序相对micro1最大logprob差0.34936，分组最大0.36202，
均未通过原概率比较门槛。见[更新后对照](slurp-true-batch-update-dp2-v4-postupdate.json)。

该对照比较的是不同micro产生的不同参数，证明不能把候选当作与micro1逐步等价的替换。
它不直接证明同一checkpoint的actor与推理实现不一致，也不能用跨轨迹差异代替后者的
验收。应分别报告数值梯度差、学习轨迹变化和同参数跨执行形状/后端概率差。
后续若采用新micro，必须作为新实验版本，仍需同参数完整候选概率和恢复/导出验收，
不能通过放宽现有阈值声称等价。

v5 `7fa8cdff`已在原任务结束、确认5/6卡空闲后启动；同一post-Adam参数分别以
单条和当前micro读取canonical32/rank，保存原始概率并按原门槛独立复算。
仍是候选HF形状对照，不是vLLM或正式RL验收。原运行快照和rc2不修改。
新class-spread processor在04:32 UTC到期观察仍running，账本进度7680/81026；
下一普通观察不早于该次观察+600秒。独立收集器已就绪，完成后核全量坐标及输入身份。

同参数对照v5 `7fa8cdff`已成功退出0，独立收集并复算：micro8原序和长度分组在
更新后各自同一参数下，双卡canonical64输入的单条/批量最大logprob差均9.53674e-7，
argmax变化0，原概率门槛通过。见
[同参数对照](slurp-true-batch-update-dp2-v5-same-update-batch.json)。这不撤销v4的
跨轨迹差异，也不代表跨后端或连续训练通过。

在固定镜像CPU容器`0a474b21`成功重建三种更新后的112个LoRA张量，检查DP2 FP32
分片拼接、BF16导出与safetensors重载；随后只对新导出权重补0644读取权限并核hash不变。
原序micro1、原序micro8及分组micro8三个产物已收取并独立核对文件hash。
分组micro8导出已启动`a6f14378`（175卡5），使用既有原始HF/vLLM profile核完整
learning256，另对其中驻留64条绑定除policy_id外完全相同的输入身份。等待终态及
原始概率独立复算；尚未声称PEFT执行/跨后端通过。

class-spread processor最新到期观察仍running，19840/81026次使用；新增收集器完成后
还会直接调用未修改的`load_full_schedule`，验证实际processor结果能被正式全量入口消费。
新日程改变批次成员，只能新实验起步，不能借旧日程checkpoint游标冒称恢复。

后端v1 `a6f14378`已退出1/non-OOM：HF256完成，vLLM尚未产生结果，在EngineCore
初始化报`Cannot re-initialize NPU in forked subprocess`。启动器来自HF-only探针，
遗漏`VLLM_WORKER_MULTIPROC_METHOD=spawn`；新v2 `19e09d4e`显式设置spawn，只重跑
vLLM并固定原已完成HF结果与同一adapter。原失败快照不修改、不把启动失败当数值失败。

独立完整输入身份匹配后，驻留候选64条与原HF重载的最大logprob差0.4416466，
原门槛失败。单条/批量候选自身一致不能替代该跨实现结果。新`65c153ef`在卡6使用
同一导出adapter与候选fixed-projection/packed-attention实现重载，对照原驻留64及HF256；
用以区分权重导出/FSDP状态和计算实现差异，结果未出，不能预先归因。
收据[原HF对照](slurp-batch-updated-backend-v1-parity.json)。

候选profile重载`65c153ef`已exit0且独立收集：同导出adapter重载的候选实现与原驻留
64条完整概率最大差0；相同256输入上候选实现与原HF最大差0.4580321，门槛失败。
见[候选重载对照](slurp-batch-updated-candidate-hf-v1-parity.json)。该证据支持本次
112张量导出/重载在被测输入上正确，主要后续工作转向fixed-projection/packed-attention
候选与推理计算实现的对齐；尚未区分两种候选改动各自贡献，也不外推所有checkpoint。
原vLLM spawn修正任务仍待终态；class-spread processor最新到期观察30080/81026。

### 2026-11-14 全量日程验收与组件隔离

class-spread processor `f9af7340`成功退出0/non-OOM；独立核验325项依赖、40,513条唯一
输入及81,026条使用记录的完整日程坐标，最大2249 tokens。第一轮真实处理音频，第二轮
复用同输入结果并生成新坐标；不是两遍独立音频处理。未修改的`load_full_schedule`接受
实际收据，加载634更新、每epoch317。见[processor收据](slurp-class-spread-processor-v1.json)
和[正式入口消费验证](slurp-class-spread-training-consumer-v1.json)。新日程未启动训练。

原HF/vLLM重跑`19e09d4e`成功，256条完整候选最大logprob差9.53674e-7。
投影与attention单独干预均已成功执行并独立收集：相对原HF，投影最大差0.5995884、
9/256条argmax变化；attention最大差0.4344473、14/256条变化。两者都需对齐，不能将
全部偏移归于一种组件或把各自误差线性相加。见[组件对照](slurp-batch-updated-components-v1.json)。

新增独立`span_batch_rollout_candidate`，为vLLM TP1单条完整prefill调用与候选actor
相同的TND sparse3/inner2 attention；保留原vLLM权重、LoRA、QKV及输出投影。
布局/投影流/拒绝非法TP及packed核心共9项CPU测试通过。不是生产profile，也未修复投影。
真实256输入attention-only HF/vLLM对照`c517e200`已在确认空闲的175卡5启动；
exact docker wait session42599，普通检查不早于launch+600。完成后独立收集
`collect_batch_updated_backend.py --tag batch-updated-attention-vllm-v1`，再运行
`compare_batch_updated_followups.py --kind attention_vllm`。正式训练仍未准入，原rc2不改。

### 2026-11-14 attention跨后端通过，推进完整候选

`c517e200`成功退出0/non-OOM，独立核验326项依赖及全部输入身份；attention-only HF
与采用同一TND sparse3/inner2内核的vLLM在256条上完整候选概率全部通过，最大差
9.536743e-7。该比较的参考是attention-only HF，不能误读为原eager HF。
见[attention对齐收据](slurp-batch-updated-attention-vllm-v1-parity.json)。

新增独立`span_fixed_rollout_projection`：使用vLLM当前权重，对音频Linear、拆分音频
QKV、语言基础QKV/MLP/o_proj及词表头执行与actor相同的128行×8 tile投影；
原vLLM LoRA选择与执行保留，尚未声明LoRA算子等价。偏置/返回结构、非等宽QKV、
原生LoRA调用、MLP/输出头及已有核心共29项CPU测试通过。
完整候选HF256对照`af376351`确认175卡5空闲后已running；exact wait session76247，
普通launch+600。完成后collect_batch_updated_backend.py --tag batch-updated-fixed-base-vllm-v1，
再compare_batch_updated_followups.py --kind fixed_base_vllm。冻结rc2原1360文件hash全相同；
正式训练仍false，DP4/global128/长期RL收益与跨任务迁移尚未完成。

### 2026-11-14 完整候选跨后端通过，DP4新日程前缀启动

`af376351`成功退出0/non-OOM，独立收集及完整输入身份比较通过：固定基础投影、音频
投影、输出头加packed attention、保留原生vLLM LoRA，与full candidate HF256最大
logprob差9.536743e-7。见[完整候选后端收据](slurp-batch-updated-fixed-base-vllm-v1-parity.json)。
这仅证明本次更新后adapter及256输入，尚非所有checkpoint/多请求vLLM/正式RL准入。

独立候选入口`tools/jev/train_slurp_batched_instruction_candidate.py`复用原生verl/FSDP
及既有loss、导出/本地存储机制；安装候选计算profile并使用真实批量前向，保持global128
成员、标签和权重，以真实长度整体重排请求载荷。新batch配置只允许micro8/16；
配置/零权重载荷/顺序可逆及既有长度规划共21项CPU测试通过。该入口当前明确拒绝继续
训练，以免借原生产lineage声称候选已通过恢复；需独立新恢复验收后开放。

`class-spread-batch-dp4-prefix-v1`在确认175物理4–7均空闲后启动，容器
`4944f7980c428e7fe3172ab801d124e3ab3d12f2d8ec5f36c384b33209b87f1e`。
BF16执行/FP32 master、LoRA q/v r4 alpha8、lr1e-4、seed317、global128、micro8×累积4，
读取已验收完整40513×2日程（634更新），本次仅执行新初始化的前两次更新并导出step2。
未复用旧日程游标，未挤占0–3卡其他项目。全部热写/data，Ceph只读，fsync保留。
exact docker wait session13393；普通检查不早于launch+600。完成后运行
`python3 .local/simple-final-v1/collect_class_spread_batch_prefix.py --tag class-spread-batch-dp4-prefix-v1`，
再独立核对日程/四rank state/BF16导出及同参数HF/vLLM；收集本身不代表这些验收已过。
原rc2的1360文件hash不变，正式全量训练与RL尚未派发，总目标未完成。

### 2026-11-14 DP4前缀、checkpoint与新权重后端验收

新class-spread global128/micro8前缀`4944f798`成功exit0/non-OOM，四rank各完成两步。
第二步实际更新约13.78–13.83秒，峰值allocated约27.42–27.68GiB；含准备/审计/checkpoint
总步时约21.23秒。见[实际计时](slurp-class-spread-batch-dp4-throughput-v1.json)。两步含冷启动，
不作为稳态吞吐或RL收益。全部256真实样本来自新全量日程的前两个batch。

独立CPU审计`dcb00ebc`exit0，实际音频processor重算长度及完整载荷排列、四rank112LoRA
FP32 master/Adam结构及游标、BF16导出、实际PEFT重载和durable commit全部通过。
报告与24项产物证据图再次hash复核。adapter identity
`1b2b661483ceae5f78f40c2eaa179b861b4fa24cf6012668b5a00cc7ddb8e113`，
见[checkpoint收据](slurp-class-spread-batch-dp4-checkpoint-v1.json)。

新权重HF/vLLM256任务`eba03a0e`成功，独立原始概率复算max9.536743e-7/argmax0；
CPU证据链审计`cf598711`exit0并完整graph复核，通过新权重、输入、checkpoint关联。
见[后端收据](slurp-class-spread-batch-dp4-parity-v1.json)。不是正式RL/全量效果验收。

新增候选独立恢复lineage及同参数驻留单条/micro8验证，配置/游标/原生恢复34项CPU测试过。
恢复启动器首次仅本地SSH argv过长失败，尚无远程运行；改为stdin传输预检脚本后，确认
4–7卡空闲并启动`0cd237c72a1c0cb4e640c9833bea568add9f2caaf5194fe5d903d9d82dddf454`。
从已认证step2精确恢复FP32 master+Adam，先256输入分rank单条/批量对照，再新日程step3/4。
普通检查launch+600；确切docker wait句柄见当前会话。完成后用
`collect_class_spread_batch_prefix.py --tag class-spread-batch-dp4-resume-v1`收取，独立核验恢复和
新endpoint；不能由启动推定恢复通过。原rc21360文件hash不变，正式全量/RL未派发。

### 2026-11-14 精确恢复通过并启动全量micro8训练

恢复任务`0cd237c7`成功退出0，四rank参数与Adam精确恢复；同step2的单条/micro8共512次
完整概率对照由独立CPU重算通过，max9.536743e-7；新日程update3/4完成。step4更新
约14.94–14.96秒，总步时22.45–22.47秒，峰值allocated最大28.30GiB，仍非稳态。
独立endpoint审计`0d868b60`通过32项证据图、真实processor排列、master/export/PEFT。
step4 adapter `7c33b29b0862bca1027cfb6b315b7c4dd8c2baa8b80d415898bc2d2fde3974ce`。
HF/vLLM256 `b19bab49`成功，独立审计`41607d09`成功并重新hash369项证据，max9.536743e-7，
argmax0。见[完整step4准入](slurp-class-spread-batch-dp4-resume-parity-audit-v1-accepted.json)。

恢复/批量配置/长度/固定投影/attention/原生Adam恢复等96项相关CPU测试通过。
新增训练指标聚合按各micro贡献求和再DP平均，不再把微批均值重复平均；实际四步加权CE
为6.3165、6.2986、5.5609、5.2211，各步样本不同，不能视为固定集改善或RL收益。
见[训练指标](slurp-class-spread-batch-training-metrics-v1.json)。

确认175物理4–7空闲后，全量接续已启动：
`934b106a6563f87f29e343f66b10e75485fd7c1b5de93864f920e00f02b931df`，
tag `class-spread-batch-dp4-full-v1`，从已验收step4继续update5..634（630更新）。
固定global128/micro8×累积4、两epoch完整40513学习录音、BF16/FP32 master、LoRA r4/alpha8、
lr1e-4、seed317；同global内部长度组批和类别时序分散日程，checkpoint间隔64及epoch/终态。
每次恢复先做同参数单条/批量概率检查；0–3其他项目不动。配置
`configs/experiments/slurp-class-spread-batch-stage1-v1.json`记录本次实际派发。

普通检查不早于launch+600，exact wait句柄见当前会话。完成后collect_class_spread_batch_prefix.py
--tag class-spread-batch-dp4-full-v1，随后launch_class_spread_batch_dp4_full_audit_v1.py；
完整selection入口launch_class_spread_batch_dp4_selection_v1.py已准备但必须等terminal634和独立
checkpoint验收才可运行。尚无新全量训练完成/selection/RL/迁移效果。原rc21360hash不变。
