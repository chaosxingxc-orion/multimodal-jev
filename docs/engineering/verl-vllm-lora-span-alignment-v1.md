# verl / vLLM / LoRA 多 Span 一致性与奖励方案

## 当前证据与结论

**最新用户精度决定：BF16 训练/推理。** 基座、LoRA 执行/导出、LM head 投影均 BF16；FP32 保留在候选归一化、损失及优化器 master/state。显式关闭 PEFT adapter 自动升精度，vLLM 指定 lora_dtype=bfloat16、head_dtype=model；两侧使用完全相同的未合并 adapter。已有 FP32 adapter 探针仅作历史诊断，不作为 BF16 的验收证据。新增本地 FP32 master AdamW 桥接支持 BF16 SFT，尚非分布式 verl optimizer 集成。见 `experiments/slurp-native-spans-bf16-v1.md`。

最新进度：BF16 candidate-CE LoRA 20步训练与独立保存重载已完成；48条学习录音重载候选概率最大误差为0。原生vLLM BF16概率gate失败，max logp差0.140574、完整动作比率[0.811465,1.199267]。分层及同进程复算依次确认语言RMSNorm、基础QKV打包、音频卷积internal-format与attention/rotary运算顺序的差异。最终组合探针 `language-rotary-v2` 在一条600-token真实录音上全部100个非stride激活trace点逐位一致，max logp差5.9604645e-08、完整动作比率1.0000001192，通过固定gate；保留vLLM原生LoRA add、未合并adapter和BF16执行。此前的单项干预未单独通过，不作为独立修复保证。证据见 `slurp-bf16-language-rotary-v2-verified.json` 和 `experiments/slurp-bf16-projection-audit-v1.md`。

已将组合对齐封装为命名profile `qwen3-asr-bf16-hf-order-v1`，限定TP1、单条完整序列、无cache/解码/量化。48条顺序请求suite已全部通过，max logp差1.1920929e-07，完整动作概率比范围[0.9999999851,1.0000001788]，覆盖全部八种Boolean监督组合，原始音频约1.7–8.4秒。训练入口必须验证suite结果文件hash、基座/adapter身份并独立重算每条完整动作gate；仍逐次验证更新前和新adapter，不能仅凭profile名称放行。目前真实语音RL更新仍为0，批量、分布式及跨任务效果未验收。

工程新增：单rank `span_actor_update` 使用实际verl GRPO与clipped PPO完整动作损失；BF16 optimizer已支持master/state恢复。`tools/jev/train_span_rl_step.py`连接真实音频HF参考、vLLM候选rollout、训练前gate、一次更新、新版本导出、独立actor重载和vLLM复核；后端顺序退出，避免NPU显存并存。当前该入口首次NPU验收已在真实语音更新前正确阻断（max logp差0.0652434，0次更新），不是分布式Ray/verl trainer已完成，不声称真实语音RL成功。详见 `experiments/span-actor-update-v1.md`。

下文保留设计和历史诊断；其中“LoRA未验证”“当前SFT为全词表CE”属于初始阶段描述。候选CE对照已完成，最新主线使用candidate CE、BF16执行策略。

### BF16 落地与 verl 接入要求

| 环节 | 锁定行为 | 验收依据 |
|---|---|---|
| 基座/音频编码/语言前向 | BF16，冻结基座 | 实际参数/输入 dtype 与共享特征身份 |
| LoRA | q/v，rank4/alpha8，dropout0，执行及服务快照 BF16 | 禁止只在 rollout 转换 FP32 checkpoint；检查 A/B、打包切片及缩放 |
| 输出投影 | 两侧 BF16；vLLM head_dtype=model | worker logits_processor 的实际 dtype |
| 概率/损失 | FP32 candidate log-softmax、joint logprob、KL/PPO | 同候选、同温度、同 mask；不能混用全词表 logprob |
| 优化器 | FP32 master/moments，按步舍入为 BF16 执行副本 | 恢复时另存 master/state；服务只消费执行快照 |
| 同步 | barrier 后完整 adapter 版本切换 | 新 LoRA ID + hash + worker 回执 + 更新前概率 gate |

固定镜像源码核验：`/opt/verl/verl/workers/fsdp_workers.py` 及 `workers/engine/fsdp/transformer_impl.py` 的 get_peft_model/from_pretrained 没有指定 autocast_adapter_dtype；FSDP另设param_dtype（BF16）、reduce_dtype（默认FP32）、buffer_dtype（默认FP32）。因此，静态LoRA FP32参数不必然说明FSDP前向为FP32，也不能反过来仅凭param_dtype就认定所有LoRA实际路径正确。需在所选FSDP包装/混合精度配置下核验实际A/B计算及同步快照；不要把本地master bridge直接叠加到已有FSDP master管理造成双份更新。小规模单卡桥接已验证，正式verl Span actor仍需单独接入与梯度/版本验收。

BF16候选CE实验已完成20次均衡更新，学习诊断93/142，静音80/142；112个导出张量BF16。独立重载48条所有候选logprob误差0。尚不能由此推出vLLM一致，后端矩阵乘/attention/scaling累积顺序仍可能不同。

BF16跨后端探针最新结果：实际28个LoRA wrapper缓冲和LM head投影均BF16，但同checkpoint单音频最大logprob差0.140574，完整动作比率[0.811465,1.199267]，一项argmax改变。adapter加载与返回base隔离通过；概率gate失败，不准入RL。下一步分层定位首次偏差，不能把“统一dtype”报告成“训推一致已完成”。过程奖励与精度修复独立，当前不调整奖励补偿后端误差。

分层定位与norm干预完成：原始61点trace不改变logits。音频投影有小差异，第一层文本norm亦存在独立差异；将113个语言RMSNorm按HF顺序计算后，第一层文本embedding/norm逐位一致，候选logprob最大误差降至0.101054、argmax全同，但联合概率比[0.844176,1.153852]仍失败。保留独立诊断配置，不将干预当作默认已验收服务后端。剩余音频塔、packed QKV及attention差异需继续定位；见 `experiments/slurp-bf16-layer-trace-v1.md`。

采用 LoRA 更新语言骨干，actor 与 rollout 共享基座及适配器身份；不将“同 checkpoint”当作概率一致证明。现有真实 HF Span 前向、LoRA SFT、保存重载已验证；vLLM pooling 无生成多 Span 前向已验证，但单样本候选 logprob 最大误差 0.063258 超过 0.02 门槛。尚未验证 pooling + 原生 LoRA 请求、热更新或端到端 verl Span actor 更新。普通生成 actor 的 responses 拼接逻辑不能直接复用。

均衡调试已固定八种标签组合各五条、每次更新三项任务各正负一例；单 epoch 仍全部预测 false。不能把类别平衡、前处理或 RL 中任何一项单独当作已解决方案。

## 配置与数值路径

1. 模型：锁定本地 Qwen3-ASR-0.6B-hf revision/权重哈希；语言 q/v LoRA 首先延用 rank4/alpha8、dropout0、bias none。音频塔、projector、原生 LM head 冻结；验证 adapter 参数和实际梯度覆盖。是否提高 rank 单独对照。
2. actor：HF/PEFT 梯度路径，eval 禁用 dropout 但允许 autograd；BF16 主计算、FP32 候选 log-softmax/联合概率/损失及梯度统计。优化器状态实际 dtype 要记录，不能假定 LoRA 全部 FP32。
3. rollout：vLLM pooling runner，保留 native LM head、同一 p−1 位置、同一候选映射。TP1、eager、无 prefix cache/chunked prefill、无量化先行。解码生成接口不承担 Span 采样。
4. 首选双方 base+unmerged LoRA，逐项验证 target mapping、qkv 打包位置、scaling、dtype。vLLM 当前 pooling 适配器的 LoRA 未验证，不能因父类 SupportsLoRA 就宣称支持。若未支持，可用每轮合并快照执行 rollout，但它和未合并 actor 数值不同，必须过概率门槛；也可实现一次真正等价的 actor 合并前向且保留 LoRA 梯度，不能用 detach 合并绕过训练。
5. 不在一侧启用 LoRA merge、量化或不同 attention 实现后只比较 argmax。BF16 分离矩阵乘与先合并权重存在舍入差异，不能承诺逐位一致。

## 输入身份与音频处理

原始音频哈希 + 前处理配置/算法版本 + 实际处理后 float32 波形哈希 + 特征张量/长度/掩码身份共同绑定请求。训练增强先固定实例，再复制给同组 G 个动作；随机增强不能在 actor 重算时重新抽取。不同增强实例为不同 request_id。静音仅诊断，不伪造训练负例。

当前 raw、首尾能量裁剪、RMS 增益、preemphasis、speed/pitch 扰动分别做单因素对照，HF 与 vLLM 使用同一处理后的波形/特征。预处理不同无法靠重要性权重修正。固定 processor、模板、音频 token 展开、position IDs、Span p−1、符号候选顺序、mask 和 temperature。初始 temperature=1，不加 top-k/top-p、重复惩罚等分布变换。

已发现旧生成路线的音频塔/worker 数值问题；新 pooling 上尚未定位。诊断顺序为输入特征 → 音频投影 → 语言层隐藏表示 → LM head → 候选归一化。共享实时 HF 音频塔在旧 worker 内并未修复误差，不重复宣称已解决。若音频塔冻结，可研究在服务内部统一实时音频编码或按音频/前处理/encoder 版本缓存投影，但必须另验协议和数值；模型仍直接消费原始音频，不经过转写或文本模型。

## 每次政策更新的同步

同步阶段初始采用全局 barrier：

1. 冻结 rollout adapter 版本 v，分发基座/适配器/模板/前处理签名；worker 回报实际加载身份。
2. vLLM 一次 pooling 得到全部候选分布，显式 categorical sampler 抽取完整 Span 元组；保存原始候选 logits、采样变换、逐 Span/联合行为 logprob、随机种子和 request identity。
3. actor 在任何 optimizer step 前重算同一动作概率，与行为政策逐项比较。不用 actor 重算值覆盖实际 rollout old_log_probs。
4. 通过 gate 后才训练 LoRA；KL reference 为固定 SFT 初始化，old policy 为本轮 v，两者不是同一概念。
5. 结束更新后将 v+1 适配器导入新引擎/新不可变 LoRA ID，核验加载后的探针。确认在途 v 请求全部结束后再切换，不复用旧 ID 隐藏替换，也不静默消费陈旧队列。

初期可以重建引擎，正确性稳定后再验收 hot swap 与多卡。单机四卡授权不代表必须立即分布式。

## 概率门槛

最初保留 max per-span |Δlogp|≤0.02 诊断门槛，并增加完整动作的联合比率检查：r=exp(ΣΔlogp)。每项 0.02 对 3 Span 最坏为 exp(0.06)≈1.062，因此单项门槛不足。初始建议未更新 actor/rollout 的联合比率在 [0.98,1.02] 内；现已在 span_parity.py 实现全候选精确检查（利用独立分布的极值求和，不只抽样）；并未通过真实 gate。基座实测联合比率范围为 [0.879938,1.076945]，argmax 均相同仍不能通过。记录 max/p95、候选 KL、近边界样本 argmax 及所有候选概率，不能只测已采样答案或高置信样本。

探针覆盖多音频长度、正负组合、单问/多问、不同候选数、预处理、基础/非零 LoRA、保存重载、更新前后和批量变化。先保留原始误差，不能靠放宽阈值、每后端单独温度校准或奖励改写隐藏差异。后续若研究 off-policy correction，需独立定义行为/目标概率和截断偏差，不能当作基础一致性验收替代。

## 分类目标与过程奖励

当前 SFT 用全词表 CE，但部署使用候选内归一化。建议在完全相同 40 条均衡数据、初始化与更新预算上对照候选 CE：L=−Σ supervised log p_candidate(gold)。全词表 CE 的降低可能主要体现输出符号学习；候选 CE 与实际有限分类政策对齐，但收益仍需测量。可靠中间标签的辅助 CE 是明确且低方差的监督来源。

三项 temporal 标签是并列分类，目前没有真实 Final 依赖链，应分别计分，不能给前两项冠以过程奖励并宣称推理收益。对未来具有独立 Final 的任务，分开记录：

- R_final：Final 判决正确性或正式任务效用，优化及模型选择的主要目标。
- R_aux：有审核标签的前置 Span 平均正确性，缺标屏蔽、unknown 视合法类；归一化按可用监督数，避免增加问题数量就增奖。
- R_consistency：仅对已验证依赖规则的逻辑矛盾给小惩罚。全错但自洽不应得正奖；没有规则或缺标则不启用。

首个复合候选可为 R=R_final+λR_aux−μC，0≤μ<λ<1 是有界 binary Final 场景的设计起点，具体系数只在学习/选择数据决定，不能声称这是证明过的最优值。若要求所有“Final 正确”的奖励都高于“Final 错误”，应明确约束 λ+μ<1（R_aux,C∈[0,1]），而不只是各项分别小于1。并且这一逐样本优先级不保证期望 Final 准确率最优，模型选择仍以 Final 和独立约束为准。提供 Final-only 与 auxiliary-CE 对照，不默认所有标签都转 RL。

关键限制：当前 sampled auxiliary answers 不反馈给 Final。对于固定输入、只依赖 Final 的奖励，中间动作 score-function 项的期望直接梯度为零；共享骨干依然收到 Final 梯度。这使“全元组联合 logprob × Final reward”包含不必要的采样方差。增加辅助奖励会训练前置判决，但不能据此证明它们对 Final 产生因果推理作用。

因此后续建议比较两种有明确语义的更新，而不混称：

A. 完整元组 joint PPO/GRPO，现有数学接口，作为基线，适用于非可分整体效用；联合概率比一次 clip。
B. 可分奖励的逐 Span 信用分配：Final 奖励仅对应 Final score 项，可靠中间奖励/CE 对应自己的 Span；共享骨干仍联合更新。on-policy 可分奖励下可去除交叉项方差，但逐 Span PPO clipping 与 joint clipping 是不同算法，有限更新下不可声称严格等价。非可分一致性奖励需另做 joint 项或有证明的条件期望估计。

有限小分类空间还有更强对照：当前三个 Boolean Span 仅 8 种动作，可枚举准确期望 reward/梯度，无需估计过程状态。任何 RL 创新都需与候选 CE 和可枚举期望目标比较；将分类正确率换名为 RL 奖励本身不构成学术贡献。跨任务、组合规则与非可分业务效用是更有意义的研究场景。

不要奖励输出长度、JSON 格式、额外问题数量、无参考的自信或“思考已触发”；这些与本接口性能目标无关。校准用独立 NLL/Brier/ECE 评价或显式 proper-scoring 辅助目标，不拿自信分数当正确性。

## 均衡、GRPO 归一化与执行顺序

均衡来自学习集采样及 batch 分层。若同一 request 的正类权重为常数，把它乘到组内 reward 后再按组标准差归一化，权重会近似抵消（epsilon 除外）。需要类权重时用归一化后的明确任务/样本损失权重，或不做 std 标准化；不要重复加权。保留真实评估分布，报告 macro/balanced accuracy、每类 recall、混淆矩阵、Final、全 Span 及静音对照。

顺序：固定均衡数据做候选 CE 对照 → 定位 base/非零 LoRA 的 pooling 概率差 → 实现并验证同步与完整请求身份 → 最小 verl Span actor 更新 → 等预算 SFT/Final-only/可靠辅助奖励对照 → 独立来源及规则/任务留出的泛化评估。预处理改进并行作为单因素实验，不以其收益替代 RL 方法收益。当前所有性能仍为学习诊断，完整学术复现和泛化目标尚未完成。

后续执行证据：相同40条、相同20次更新的候选CE对照完成，原48条学习人口 after=92/142 vs 全词表70/142；静音80/142，完整三问全对仍与静音均10/46。仅支持继续研究分类目标，不证明泛化。原生FP32 LoRA pooling诊断仍在验证，v1因dtype字符串不被API接受而失败，v2改为torch.float32。

实现进度补记：已增加候选CE、全动作联合概率gate及有审核mask的辅助奖励函数，51项相关测试通过。奖励接口未接入参数更新；vLLM LoRA kernel实际精度与actor匹配仍待本轮探针结果。候选CE从相同初始化出发的改善不能作为同精度后端一致性的替代。

LoRA探针v2已失败：pooling warmup的自动LM-head LoRA wrapper触发Ascend FP32 bgmv_expand限制（只支持half/bf16）。真实adapter仅q/v，下一步限制包装范围后验证q/v路径；未准入RL。此证据不等于所有FP32 LoRA运算都不支持。
