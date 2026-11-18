# Global128 / micro32 吞吐目标

用户提出0.6B LoRA应提升microbatch，优先保持global batch128，目标每卡micro32。
四卡时128 = 4 × 32 × 1；当前128 = 4 × 1 × 32。
这是保持有效batch和优化步数、减少累积次数的目标，不是将global batch改成512。

当前结论（2026-11-14）：global128继续固定，micro32为DP4候选目标，尚未获NPU训练准入。
这里micro是每个数据并行rank一次前向的独立音频数，不是GRPO的候选采样数G。
LoRA参数少不意味着反传激活少；需按最长音频/文本组合验证容量和稳态吞吐。

| 数据并行rank数 | 每rank microbatch | 梯度累积 | 全局独立音频数 |
| --- | --- | --- | --- |
| 4 | 1（既有Stage1） | 32 | 128 |
| 4 | 8（候选） | 4 | 128 |
| 4 | 16（候选） | 2 | 128 |
| 4 | 32（目标） | 1 | 128 |
| 8 | 16（仅口径说明） | 1 | 128 |

八卡获授权不等于八卡都能充当actor；需扣除rollout及其他服务占用，按实际DP计算。
先保持global128、学习率和真实请求loss分母不变，测量包括前向、反向、优化器及
数据准备的稳态吞吐；选择保留显存余量且通过训推一致性的最快档，不按最大micro选型。

源码核查发现`span_hybrid_actor.build_actor`的`_forward_micro_batch`仍逐请求调用
actor_module，随后拼接输出统一backward。提高旧micro参数会积累多份活跃计算图，
不构成真正的批量前向。当前simple_batch_config也只准入1/2/4。
runtime profile明确为single complete sequence，真正组批需独立数值验收。

已有micro2两步实验为2.2109请求/秒，对应micro1历史控制1.88157请求/秒，约提升17.5%；
micro2峰值allocated约28.77GiB、reserved约60.06GiB。两步结果不代表全量稳定性，
reserved不是全部活跃张量，不能直接线性外推micro32容量或断言必然OOM。
LoRA减少可训练参数及优化器状态，语言骨干反传仍需要激活；当前HF eager attention
使序列长度和batch对内存仍有重要影响。

独立设计配置为`configs/experiments/slurp-simple-true-batch-throughput-v2.json`。
先实现正确音频/文本mask、position与Span对齐的批量前向，再在相同global128、
初态和请求顺序下逐档测2/4/8/16/32，包含长短混合及末批padding。
使用实际吞吐、显存余量、概率/梯度/更新及训推一致性选择配置，不强制选择32。
优化attention kernel和activation checkpointing若需要，作为后续独立变量验证。

## 2026-11-14 实现与CPU验证

新增独立`span_audio_collate.py`：保留各请求原音频特征及其已有padding，
仅追加被mask的整块零padding；文本右padding，保留Span位置并显式设置position IDs。
逐请求核验音频占位符数量、音频先于Span、相同候选支持集与audio window边界。
`forward_span_audio_batch`只调用一次模型，按原请求顺序拆回可求导概率。
尚未接入生产actor，不把旧serial循环改名为batch。

本地21项相关测试通过，包括micro1/2/4/8/16/32概率、真实请求分母加权loss、
梯度和下一次Adam更新对照；这些使用合成骨干。
固定镜像独立CPU容器e843a623退出0，真实小型随机初始化Qwen3-ASR代码、
4条31/100/205/899帧混合输入只调用一次骨干；与单条路径maxlogprob差0，
最大梯度差1.1920928955078125e-7。远端源码hash独立核验，收据
`slurp-true-batch-cpu-v2.json`。首次精简包容器6135c8e4因缺包入口加载旧包而失败，
未进入模型计算；完整源码包新容器通过，原失败记录保留。

尚未运行0.6B预训练BF16 NPU容量测试，也未验证LoRA/FSDP或vLLM训推一致性。
下一步使用真实learning音频及固定权重验证，再接入独立候选actor及micro阶梯容量测试。
Stage1与rc2归档不变，rc2已登记1360文件内容逐一仍相同；新增候选模块尚未准入正式训练。

## 真实预训练NPU探针准备与资源变化

新增`tools/jev/probe_span_audio_batch_npu.py`，消费已认证前256个learning输入及
两步prefix BF16 LoRA；按音频帧数分位选择32条，包含该人口的最长/最短输入。
先执行逐条reference，再测micro1/2/4/8/16/32，每档保存全候选概率、独立比较、
骨干调用次数、单次耗时和allocated/reserved显存。数值准入失败停止扩大batch。
本探针无backward/optimizer，不能作为训练吞吐或最终Stage1策略验收。

`.local/simple-final-v1/launch_true_batch_npu.py --device 5 --prepare-only`已完成本地
源码归档、配置与runner准备，没有远端派发。正式启动要求实时检查目标卡空闲，
校验所有固定依赖，容器只映射单卡；观察器已接入该handle，普通观察至少600秒。
Stage1最近01:17 UTC原容器running，四rank完成604/634正在605。

资源复核发现此前空闲的0–3卡已被`omni-npu175-instruct-tp4-v1-run01`
容器（1a0eb0b8eedb）占用，运行Qwen3-Omni-30B-A3B TP4服务，端口8204，
每卡worker约57810MB。未停止或修改该服务；4–7仍为Stage1。
因此等待Stage1释放后用卡5执行探针，卡4优先供完整selection评测。
正式RL既定DP4+TP1需要5张空闲卡，当前资源不满足；需在派发时重新核验，
不能沿用“0–3空闲”的旧结论，也不能把候选探针准备记作NPU实测完成。

## 独立结果复算准备（2026-11-14 01:28 UTC）

新增`span_batch_probe_audit.py`及`.local/simple-final-v1/collect_true_batch_npu.py`：
要求确切容器成功终态、固定镜像/配置/源码/依赖hash，原始字节收集；逐请求身份
与认证learning256原记录绑定，重算每档32×60候选概率一致性，核验单次骨干调用数、
耗时算式与显存记录。负向一致性结果可审核但不通过，禁止summary自报覆盖原始结果。
12项本地测试通过，含缺失/重复/身份/概率/调用数/摘要篡改拒绝；collector仅语法验证，
尚无真实NPU终态可以收集。遥测内存/耗时是运行记录，不称独立硬件测量。

01:28:08 UTC实际观察Stage1仍running/OOMKilled=false，四rank完成613/634正在614，
下一普通观察至少01:38:08。真实批量探针仍只本地准备，无远端进程；下一先核Stage1
成功终态，再派发最终checkpoint CPU审核及空闲卡5探针，卡4保留完整selection。

## verl训练循环接入候选（2026-11-14 01:38 UTC）

新增独立`span_batched_hybrid_actor.py`，复用原build_actor的原生verl认证、
update_policy、准入及loss上下文，只给新建actor安装批量forward。
保留完整behavior概率检查，过程不读取gold；生产入口尚未选择此builder。
新`probe_batched_hybrid_actor_dp_cpu.py`固定global128、DP4、127真实+1零权重请求，
分别比较CE/GRPO/RLOO的serial micro1与true-batch micro1/2/4/8/16/32，
检查梯度、参数、Adam一阶/二阶状态及实际forward调用次数。

固定镜像CPU容器49dd624ba609a38831ffc7ee5818a49d6ab0a593f28ff36a146e85b80436db6f
已启动并观察running（873源码文件归档）；下一普通观察至少01:45:55 UTC。
它是合成音频骨干、FP32 Gloo/DDP，不代表预训练BF16/FSDP或NPU性能。
独立收集器`.local/simple-final-v1/collect_true_batch_verl_cpu.py`已准备，尚未收到通过结果。

01:46 UTC到期观察该容器exit0，随后独立收集4rank×21案例及873源码hash通过。
serial与batched路径最大梯度差4.2915344e-6（逐张量atol3e-6/rtol3e-5通过），
最大参数差2.9802322e-8、Adam一阶差1.4901161e-8、二阶差8.7311491e-11。
micro32每rank只调用一次骨干；global128中127真实请求+1零权重padding，
CE/GRPO/RLOO均保留统一真实请求分母。收据`slurp-true-batch-verl-cpu-v1.json`。
该结果只覆盖合成CPU更新，不自动提升到BF16/NPU或正式训练配置。

01:38:12 UTC Stage1原容器running/OOMKilled=false，四rank完成623/634正在624，
下一普通观察至少01:48:12 UTC。当前只剩11个更新，但仍需实际成功终态才能验收。

## 预训练BF16 NPU派发

Stage1实际634完成、原容器退出0后，启动器实时确认卡5空闲并校验固定依赖，
新容器45be989abdb7112f4d63b0aeb69fafca101fba2e84e59cc1c42e7f80da540177
已启动running。下一普通观察至少02:04:54 UTC；完整启动记录位于
`.local/simple-final-v1/true-batch-pretrained-npu5-v1-launch.json`。
本探针未获得结果，仍是两步prefix策略的独立推理容量/数值验证，无训练准入声明。
同时Stage1最终checkpoint独立CPU审核fa76793b运行，完成后卡4优先执行9226评测。

## 真实BF16结果：micro2未通过（2026-11-14 02:05 UTC）

容器45be989a实际exit0；结果为负，不能用进程退出码代替数值准入。
独立核验316源码/依赖文件及原始输入/概率后：micro1误差0；micro2全部32请求
超出既定概率门槛，最大logprob差0.8308458，argmax变化0。micro2只调用16次骨干，
真实组批已执行；单次计时7.59→8.16请求/秒（非稳态训练吞吐），allocated
2.13→2.80GiB、reserved6.64→13.53GiB。未运行micro4及以上，无容量通过结论。
收据`slurp-true-batch-pretrained-npu-v1.json`，原始文件在.local/simple-final-v1。

独立诊断`diagnose_span_audio_batch_npu.py`在同卡启动79d311e2，隔离audio batching、
text padding、language batching和full batching；短长混合、最长两条和相同输入重复
三类对照，仍使用已认证learning输入和两步prefix策略，不做优化或修改数值门槛。
该探针结果尚待观察；生产actor和rc2仍不变，批量路径不得正式派发。

Stage1最终checkpoint审核fa76793b在02:05观察仍running，前置87测试通过，
全训练日程/音频/张量审核尚未终态。另只读复算634末批65real+63零权重padding，
四rank梯度一致有限，CE-only分母65；不能以末批CE0.007066替代selection准确率。

## 隔离与逐层诊断（2026-11-14）

79d311e2实际退出0，原始概率独立复算通过。两条相同输入（255音频帧）时，
audio-batch-only logprob差0.32556，language-batch-only差0.39891/0.34989；
text-padding-only差0。另一个长音频对的短样本单独padding差0.47746。
故不是仅由跨样本音频混合或padding位置引起，也不能从CPU小模型一致性推定BF16一致。
收据slurp-true-batch-isolation-v1.json。

逐层容器84016f0a实际exit0，固定源码/配置/依赖hash核验通过，原始结果归档。
相同BF16输入的单独Linear形状对照：q_proj最大差0.001953、k_proj0.003906、
v_proj0.001953、gate_proj0.015625、down_proj0.000977。完整前向第一层输入已有
音频差0.002686，后续层累积到最终读出差；这是batch形状依赖的实际证据，尚未定位
唯一内核/累积精度根因，不能宣称修复。收据slurp-true-batch-layers-v1.json。
下一优化应对算子累积精度/稳定形状方案做独立对照，并重新验证actor/vLLM；
不能放宽既有门槛或将整层逐请求循环冒称true batch吞吐收益。

## 精度选项与矩阵布局对照

固定镜像torch_npu源码确认支持KEEP_DTYPE与USE_FP32_ADD，独立进程dee6ece3
保持模型/LoRA BF16、禁用HF32、must_keep_origin_dtype，两个variant都exit0。
独立逐概率复算发现所有case与原隔离实验完全相同；不能声称选项改变了BF16累积
或解决问题。收据slurp-true-batch-precision-v1.json。

57920f21矩阵形状探针实际exit0，固定源码/依赖核验完成。在一个真实样本第一层
q/k/gate/down的BF16激活上，显式bmm使用相同权重：batch1/2/4/8/16相对自身
batch1误差0；batch32重新出现0.000977–0.015625偏移。bmm本身相对原Linear
已有0.000488–0.015625偏移，不能原位替换并沿用原策略准入。该测试仅基础Linear，
不覆盖LoRA add、不同音频/长度、注意力、梯度或吞吐。收据
slurp-true-batch-projection-shapes-v1.json。下一方向是固定矩阵计算形状及训推共同
数值配置的独立验证，不直接提升bmm/micro16为生产设置。

02:31:57 UTC完整selection d375c9d7仍running/OOMKilled=false，下一普通观察至少
02:41:57 UTC。观察器补读取run/hf和run/vllm的progress/terminal用于下次到期观察，
未为补进度立即重复轮询；尚无完整选择集成绩。

## 固定形状投影局部结果（2026-11-14）

容器52a13810收到docker wait退出0通知后核验成功终态、非OOM，317个源码/依赖
hash及配置、归档、runner绑定一致。新primitive将每条序列切成固定行数，再以固定
tile batch执行BF16 bmm，不转换参数/输入精度，也未安装到生产模型。
两组tile配置(128,8)/(256,8)、batch1/2/4/8/16/32、额外padding0/73，
目标样本移到末位且相邻样本缩放后，q/k/gate/down共96项比较相对各自fixed单条输出
最大差均0。相对旧Linear仍有最大0.015625偏移，不能继承旧runtime的数值准入。

本地primitive的10项CPU测试覆盖数值和输入/参数/偏置梯度，但NPU结果仅为捕获的
第一层基础Linear激活，未覆盖LoRA add、完整音频塔、注意力、反向或性能。
因此micro32仍未作为正式配置；下一必要验证为完整模型的独立候选profile，随后
真实反向容量、稳态吞吐和actor/vLLM一致性。收据为
`slurp-true-batch-fixed-projection-v1.json`，原始结果保留在.local/simple-final-v1。

Stage1已完成634/634并通过最终checkpoint独立审核；完整9226 selection在原profile
继续评测，最近02:42:28 UTC观察HF4042/9226，下次普通观察至少02:52:28 UTC。
该进度不是完整效果指标，不据局部诊断切换Stage1策略或正式RL配置。

## 完整模型与注意力隔离（2026-11-14）

固定镜像候选进程7363c696在卡5退出0，317源码/依赖和config/runner/归档hash
核验；420个基础及LoRA Linear叶子使用固定tile，PEFT scaling/add规则保持。
原始概率独立复算：四条learning音频构成短长混合、最长两条、相同输入重复对照。
所有音频塔输出差为0，但language-batch/full-batch最大logprob差仍0.511181，
长音频对中较短样本text-padding-only差0.410649。故仅固定Linear不足。
收据`slurp-true-batch-fixed-model-v1.json`，不是最终Stage1策略效果。

后续独立进程7e2973ab同样exit0且317固定文件核验，保持投影batch计算，
只把语言attention按每条原始有效长度切片并独立调用原eager函数，再补齐输出。
此举只用于隔离注意力矩阵形状/补齐影响，不是新的批量attention实现。
同三种对照共6个请求位置的audio/text-padding/language/full四项概率均通过原门槛；
最大logprob差4.7683716e-7，音频特征差0，原始概率由本地比较器重新计算。
收据`slurp-true-batch-fixed-attention-v1.json`。

这些证据支持在所测输入上用固定投影和原有效长度attention消除数值偏移，
未覆盖micro4..32、完整learning人口、反向/优化器/恢复、vLLM或稳态吞吐。
候选与旧profile本身也不是相同策略概率，不能沿用旧行为分布直接启动RL。
后续优先评估可组批的attention实现及共同actor/vLLM数值路径；即使采用逐请求
attention作为过渡候选，也必须报告整体真实吞吐，不把局部batch当作效率收益。
生产入口、原Stage1评测及rc2归档均不改，正式micro32与RL仍未准入。

02:52:54 UTC到期观察原selection容器d375c9d7继续running/OOMKilled=false，
HF进度5988/9226；下次普通观察以observed.json的epoch+600为准。

## 原生变长批量attention与micro阶梯（2026-11-14）

新增独立`span_packed_attention.py`，将B,H,S,D按明确有效长度打包为T,H,D，
调用固定镜像Transformers已有的`npu_flash_attn_varlen_func`（底层TND fusion
attention，causal=True、dropout=0），再恢复右补齐布局；每层一次内核调用。
该模块不接生产入口，不支持缓存/解码/任意自定义mask。
CPU六项覆盖请求隔离、因果性、补齐梯度、真实长度和单内核调用；与既有collate、
fixed Linear及概率审核共43测试通过，NPU反向尚未验证。

7e79ed27完整模型两条组批诊断exit0，318源码依赖及原始概率核验，三对照均通过，
最大logprob差2.384e-7；收据`slurp-true-batch-packed-attention-v1.json`。
随后84b93c32在learning256固定32个长度分位输入上运行micro1..32阶梯，成功退出但
数值门槛在micro8失败，按协议未测试16/32。独立核318文件与全部原始32×60概率：

| micro | 最大logprob差 | 一致性 | 单次前向请求/秒 | peak allocated GiB |
| --- | --- | --- | --- | --- |
| 1 | 0 | 通过 | 4.05 | 4.37 |
| 2 | 9.54e-7 | 通过 | 6.85 | 4.37 |
| 4 | 9.54e-7 | 通过 | 10.41 | 4.38 |
| 8 | 0.268216 | 未通过 | 12.31 | 4.40 |

所有argmax均不变，不足以替代RL概率一致性。相同候选micro4比micro1约2.57倍，
这些是单次前向含collation/传输计时，非稳态训练吞吐，也非相对原生产profile的公平
加速比。收据`slurp-true-batch-packed-sweep-v1.json`；不能据此提升micro4到正式训练。

进一步7db22481用8条短长交替/长对交替/中长重复隔离，exit0且独立复算：音频特征
及单条text padding差均0，重复同长输入通过，混合长度语言batch最大0.23725。
4b39d609重复最短输入8次，逐层全部差0；2793a77b短长交替8条，较短输入首层
input norm、Q/K/V投影、Q/K norm都无差，第一次已观察偏差在o_proj输入（0.000977），
即位于Q/K norm之后至attention输出这段，尚不能区分rotary与fusion attention。
收据`slurp-true-batch-packed-eight-v1.json`、`packed-layers-v1.json`、
`packed-mixed-layers-v1.json`（后三者文件名前缀均slurp-true-batch-）。
下一需直接捕获旋转后的Q/K及同张量attention，区分位置编码与内核批形状；修复后
再做micro8..32、反向/更新和vLLM共同profile，而非仅以micro4通过收缩优化目标。

rc2原1360文件逐hash仍一致。原selection d375c9d7在03:03:17 UTC到期观察继续
running/OOMKilled=false，HF8006/9226，下次普通至少03:13:17 UTC；未取得完整成绩。


## 混合长度 attention 内核定位与候选修正（2026-11-14）

8927b4b3确切成功终态后收集：旋转后Q/K/V与单条均完全相同；把捕获张量
单独送入attention时与参考误差0，合批输出误差0.0009765625。该输入首层偏移
已定位到attention内核。收据`slurp-true-batch-packed-kernel-v1.json`。

变体探针v1（94a37f78）因报告变量覆盖导致Tensor布尔异常退出1，保留失败。
修正报告变量后的v2（ebb19fdd）退出0，配置/源码/依赖及原始概率独立核验。
相同首层QKV对照：TND sparse2/inner0及sparse0/inner0仍差0.0009765625；
TND sparse3/inner2、BSND固定宽2032和2048均差0。inner_precise=2是
无效掩码行处理选项，不代表切换FP32累加。以上仅局部前向，不证明整模型已修复。
收据`slurp-true-batch-attention-variants-v2.json`。

选择改动较小的TND inner2作为独立候选，新增显式实验配置，不改默认内核、
生产入口或运行快照。卡5新容器1c876fd9已启动长度分位32条micro1..32阶梯；
仍是BF16前向一致性诊断，未做训练反向/更新、稳态吞吐和vLLM准入。
用户目标保持global128=DP4×micro32×accum1；DP8则micro16×accum1。
Stage1已完成，最新到期selection观察HF9226/9226完成，整体评测尚未验收。

候选1c876fd9收到确切docker wait退出0后核验成功/非OOM，318源码依赖和
config/runner/archive绑定通过；独立重算32×60原始概率，micro1/2/4/8/16/32
全部通过，最大logprob差9.536743e-7，argmax变化0。单次前向req/s分别
4.19/7.36/10.30/12.08/11.51/11.25；micro32 allocated5.38GiB、reserved7.99GiB。
这是候选单条与批量前向比较，并非旧profile或vLLM比较；该32输入诊断不代表
全量训练容量或稳定性。micro8比32本轮更快，不能把micro32作为吞吐最优结论。
后续比较global128下micro8/16/32的完整反向、优化器与稳态训练时间，检查
LoRA梯度/更新和actor-vLLM共同数值profile后，才冻结正式训练micro。
收据`slurp-true-batch-packed-inner2-sweep-v1.json`。生产配置及rc2维持原样。
