# 不可达 K/V 候选接入真实 verl actor 的验证

原完整预训练CPU verl测试在microbatch=1/2时失败：两者差0.40625，相对无padding
参考最大差约0.9681/0.8296。后续定位与首条完整模型候选结果支持继续检验不可达
K/V移除方案，但首条恢复参考logits不证明混合音频microbatch一致。

在原`probe_omni_pretrained_verl.py`增加显式`--compact-keys-root`入口，默认不
启用候选。开启后仅在真实`DataParallelPPOActor.compute_log_prob`期间安装已测试
的作用域attention包装，保留原CPU device/logprob适配器、固定false+EOS、全部
六条原始音频、原顺序`[2,0,5,1,3,4]`与microbatch=1/2两条件。采样概率仍明确为
synthetic占位，不更新old_log_probs或模型权重，不把本项当作rollout或训练。

每个microbatch保存真实捕获的48层调用、query长度和逐样本保留key数，并与实际
DataProto mask逐项对应。完整执行前，`omni_compact_acceptance.require_candidate`
要求原精确候选容器成功、六条结果全部存在，并独立加载保存logits复算各行相对
原无text-padding参考的概率差，逐条<=0.1。还核验完整源码、模型/输入身份、
48层调用及固定容器命令/资源/挂载/环境。实际actor所用adapter SHA必须与该
候选已运行版本相同。验收文件或报告中的passed标志本身不能放行。

候选inspection必须由外部协调者实际inspect后固定SHA；一个可编辑JSON文件本身
不构成运行来源。输入预检查可在候选运行期间执行，仅构建实际verl DataProto，
不加载模型、也不声称候选已成功。完整actor执行必须在六条候选成功验收之后。

本地相关测试62项通过、1个模块因宿主未安装verl而跳过。其中新增13个候选门禁
case为合成容器/张量fixture，覆盖运行中/失败/OOM、资源/命令/源码改变、缺行、
放宽门槛、权重更新、缺层、logits变化和伪造成功标志但实算超阈值。另有真实CPU
attention数学语义及梯度测试；不以这些fixture代替预训练actor执行。

预登记固定镜像CPU预检查：16 CPU/16 GiB/2 GiB shm、1800秒加30秒，无网络设备。
完整actor比较预算16 CPU/160 GiB/2 GiB shm、14400秒加30秒，12条样本前向
（6条×microbatch条件2种），无网络/设备、模型与输入只读，不自动重试。先固定
源码与输入，create成功持久记录精确ID后start。当前只准备输入预检查，完整候选
尚未全部完成，未派发新的完整verl模型对照、NPU或训练。

## 固定镜像真实输入预检查通过

源码`dd1e447`的容器
`459db7b318cc73801b085727ac95dc04dc145543c11a47ddbd448b2ec749c2d5`已退出0，
intent SHA为`27def22be94fa95659037efb1c4718577f336c858305629920b86127067e6cd8`。
外部协调者实际inspect并核对命令/资源/挂载/环境、342份源码与Git、原前向验收
链及全部学习样本身份。实际verl DataProto形状`[6,1230]`，顺序`[2,0,5,1,3,4]`，
保持真实单/双音频特征。见[独立输入验收](omni-compact-verl-preflight-v1-verified.json)。

identity明确记录compact_keys_enabled=true、compact_candidate=null：此处只准备
输入，没有把运行中的候选当作已验收，也没有加载模型或执行actor.compute_log_prob。
宿主被跳过的测试模块仍属于宿主未执行；实际镜像预检查补充真实DataProto运行证据，
不冒充同一套单元测试已在镜像重跑。新的完整actor比较仍等待六条候选全部通过。

## 完整 actor 输出的独立验收约束

`verify_omni_pretrained_verl.py`现明确区分默认actor与compact候选。默认路径拒绝
混入未绑定的compact标志、候选或attention记录；compact路径要求三个完整且唯一
的候选参数，重新执行六条候选logits门禁，并逐项核对identity中的候选身份及实际
actor helper SHA。使用原始无padding前向的prompt长度和固定顺序重建每个microbatch
的48层调用，核对逐样本key长度、query长度和mask模式，并核对实际容器挂载/环境。
候选是否完成仍由外部协调者实际inspect确定，保存的inspection自身不是信任根。

相关70项测试通过，其中新增20项覆盖缺参数、候选/adapter错配、层/样本/批次错位、
原长度改变、依赖失败、默认路径混入compact及完整验收中的挂载/环境偏差。测试使用
合成信封并mock候选验收；真实候选张量检查由既有独立测试和后续实际运行验收承担。
本次未启动新的完整actor，也未改变已通过预检查的运行源码快照`dd1e447`。

## 一次性依赖等待与派发

`watch_omni_compact_verl.py`仅观察原精确候选容器；running/paused/restarting
继续等待，SSH超时或读回失败只重试观察，不据此重启容器。候选失败/OOM/dead
停止派发；成功退出后调用固定SHA的`launch_omni_compact_verl.py`一次。启动器再次
实际inspect候选、加载六条保存logits复算门槛、核对候选与预检查源码和Git身份，
然后复制已通过预检查的`dd1e447`快照，创建容器、持久记录ID，再start。

协调者依赖等待上限14400秒，启动器上限900秒；实际actor仍为16 CPU/160 GiB/
2 GiB shm、14400秒加30秒，无网络设备、无权重更新。等待和派发均记录终态，
文件锁防止并发协调者；任何已产生dispatch-intent的尝试不得自动重放。启动器
超时属于可能已创建/启动的歧义，须检查原实验目录和精确ID，不能重新创建实验。
协调者成功终态仅表示actor已派发，不能作为actor计算或数值一致性通过的证据。

21项协调者测试通过，覆盖运行/终态/OOM、暂时观察失败、最多一次派发、禁止重放、
来源和启动器改动、等待预算及执行器改变。这些测试的Docker状态和子进程返回值
为合成fixture；实际协调进程及容器身份将在独立派发记录中保存。

协调者已从固定源码`b7b58bf`启动：PID `2948247`，process start `291856436`，
计划SHA `a82700e270eb8524f2fc6ca0e822bcec2d0cb3e013ad3420f7747ce24dc11119`。
已通过`/proc`核对真实命令和进程身份，当前在等待原候选完成；候选已完成5/6条，
正在最后一条前向。见[协调者启动记录](omni-compact-verl-dispatch-v1-launched.json)。
此时完整actor尚未创建；后续须根据`omni-compact-verl-train6-v1/launch.json`中的
精确容器ID观察执行，不再次手工运行启动器，以免与现有协调者竞争。

### 首次协调者在容器创建前停止

原候选已成功退出，协调者进入启动器后，在预检查应用终态校验处退出1：预检查
终态同时包含`exit_code: 0`和`finished_unix`，启动器错误地要求字典严格只有
退出码。原协调者终态、dispatch-result和launcher日志保留。已实际确认目标实验
目录不存在，Docker也不存在`omni-compact-verl-train6-v1`容器；没有模型前向被重放。

修订启动器允许正常附带完成时间，仍要求退出码为整数0且没有error，并保留实际
Docker成功终态、预检查结果SHA、源码身份以及全六条候选logits门禁。修订版本
单独固定快照后手动派发，旧协调者不重启，不覆盖原失败记录。

### 修订后完整 actor 已派发

修订启动器源码固定为`929368a`，单独保存在`omni-compact-verl-dispatch-fix-v1`。
确认旧协调者已停止、目标目录和容器都不存在后执行一次，重新验收全部候选依赖，
随后创建并启动容器
`1e3972f736ae4b3aadc1cf71a21610cbc6c17ab87235bec9b730d35bb651144b`。
actor源码仍为已通过真实预检查的`dd1e447`，intent SHA为
`e2f92f58e2ece2c190047b3142c2026b05926424b48c3659b753ad37643cc7e1`。

外部协调者重新inspect，核对实际命令、挂载、环境、16 CPU/160 GiB/2 GiB shm、
无网络设备及342份源码与Git身份，状态为running。见
[派发身份](omni-compact-verl-train6-v1-launched.json)和
[派发独立核验](omni-compact-verl-train6-v1-dispatch-verified.json)。
修订派发成功不代表实际actor比较通过；microbatch=1/2的12条前向、原0.1门槛及
外部完整验收仍待该精确容器完成。不得重启旧协调者或手动重复创建本实验。

### 当前 actor 的只读终态验收观察器

已从当前验收代码`1ca0f87`固定117份源码（实际清单以observer-plan为准），并将
专用观察脚本和其SHA固定在`omni-compact-verl-train6-v1-acceptance`。观察器PID
`2962227`、process start `292025489`已通过`/proc`核对真实命令和启动身份；
计划SHA `f48f6d39eb4e7f7792c302aa1f69308ddeb0699d028ed9e595c0b15ffdb51187`，
见[观察器身份](omni-compact-verl-train6-v1-acceptance-queued.json)。当前观察原actor
`1e3972f736ae4b3aadc1cf71a21610cbc6c17ab87235bec9b730d35bb651144b`，不启动新模型。

观察上限15000秒；SSH临时失败只观察同一ID。原actor失败/OOM则记录失败终态，
保留原始comparison，不重放。成功退出后重新核对actor源码与Git、实际候选Docker
及全部六条候选logits，再调用已测试的完整verl验收函数，核对两种microbatch的
响应概率、逐层调用、原始输入身份和0.1门槛。结果写入独立`acceptance.json`。
观察器未包含Docker create/start或训练/推理派发，成功也不自动启动后续权重学习。
当前只确认观察器存活；actor结果及完整最终验收仍待运行完成。

### 第一种实际 verl microbatch 条件已独立核验

原精确actor已完成microbatch=1的全部六条，耗时2069.99秒，并进入microbatch=2。
外部协调者再次inspect，核对实际资源/命令/挂载/环境、342份源码与Git、原始
输入/模型身份、已验收候选helper，以及6×48=288条实际attention调用记录。
逐条读取保存的两个响应token概率并与原无text-padding前向比较，误差为
`[0,0,0,0,0,0]`。见[第一条件独立核验](omni-compact-verl-train6-v1-microbatch1-audit.json)。

旧未修复actor的microbatch=1参考最大误差为0.9680581093；本条件在保持原0.1
门槛下恢复参考概率。此处审计对象是实际actor保存的log probabilities，actor未
额外保存logits，不能称为再次独立计算完整词表概率。固定响应仍为false+EOS，
old_log_probs仍明确为synthetic占位；没有优化器、权重更新或任务质量提升结论。
microbatch=2、两条件互比和成功终态仍待完成，整体CPU actor一致性尚未验收。

### 镜像内补跑真实 verl 接口测试

此前宿主因缺少 verl 而跳过的 `tests/test_omni_pretrained_verl.py` 已在固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
内实际执行：2 passed、0 skipped，34.842 秒。源码固定为 `c77a08d`，容器
`14482f56005726cb41d74f5177817b1f1963a01108db949f1afb6b2a8d179389`
已退出 0。协调者实际 inspect 并核对命令、默认入口、2 CPU/8 GiB、无网络设备、
只读源码挂载、349 份源码与 Git 内容及 JUnit 中的两个具体测试名。见
[独立核验记录](omni-real-verl-interface-tests-v2-verified.json)。

测试使用合成 token/音频特征，调用真实 `verl.DataProto` 和
`verl.utils.model.extract_multi_modal_inputs`，验证单/双音频合并、特征补零、
左填充、响应因果边界，以及超长输入和缺少 pad token 时拒绝执行。
没有挂载模型权重、加载模型或执行优化器；不能据此声称 NPU 或任务效果通过。

首次容器 `18765059d1a4971dfeb4aef7d1e9fc2045931882e19a5d2d20680cbec19c96be`
绕过默认 runtime-entrypoint，缺少 CANN devlib 库路径，实际结果为 1 passed、
1 failed：多模态 helper 导入时找不到 `libascend_hal.so`。其源码、JUnit、日志、
实际 inspect 和失败终态保留在 `omni-real-verl-interface-tests-v1`。
第二次独立运行仅恢复固定镜像的默认入口，源码与测试内容相同；未安装依赖、
未挂载宿主驱动或 NPU，也未修改已有模型实验的运行环境。

### 完整两条件比较失败

容器 `1e3972f736ae4b3aadc1cf71a21610cbc6c17ab87235bec9b730d35bb651144b`
最终退出 1，无 OOM；应用报预登记 BF16 比较阈值超限。独立观察器也退出 1，
没有生成成功 acceptance。原始 comparison、两条件概率、调用记录和失败终态保留，
未自动重跑、未放宽 0.1 阈值。

外部重新核对实际容器、342 份源码与 Git、原始模型/输入身份及 288/144 条
attention 调用，并从保存概率独立复算。microbatch=1 六条误差仍全部为0；
microbatch=2 按原顺序 `[2,0,5,1,3,4]` 的最大响应 token 误差为
`[0.153104,0.131920,0.138639,0.266486,0.339120,0.607848]`，六条全部超限。
两个条件用时分别2069.99/2033.71秒。见
[失败独立核验](omni-compact-verl-train6-v1-failure-audit.json)。

结论：不可达 K/V 移除修复了这批输入的单样本 padding 对照，但没有建立实际
多样本 actor 一致性，不能宣布 CPU verl 基座通过。actor 没有保存中间激活或
完整 logits，当前不能定位差异首次出现于音频编码、QKV投影、MoE还是其他算子。
失败发生在最终权重复核之前，独立报告不声称完成失败后的全权重哈希复核。

新增局部算子诊断 `diagnose_omni_batch_projection.py` 使用既有逐层诊断保存的
首层有效输入和原权重中的四个张量，固定原六条配对、1230宽度、零值左填充，
比较batch1/batch2的RMSNorm与Q/K/V投影。该实验不是完整actor重放；零填充内容
是人工控制，不能直接证明实际失败的首个差异位置。完整模型训练仍不派发。

该局部诊断已实际完成：固定镜像容器
`95186eb6a9417cd3232d8358ac2aa724606b08aff26bad4c643abc6eb7272f29`
退出0，16 CPU/8 GiB，无网络设备，耗时20.99秒。六条输入×四个算子的24项
batch1/batch2比较全部逐元素相等。外部核对实际容器、源码与Git、六份原激活哈希、
权重分片身份及完整算子覆盖，见[局部算子记录](omni-batch-projection-v1-verified.json)。
没有保存投影结果张量，数值相等来自运行时比较；不能据此排除实际音频编码结果、
其他层投影或MoE的批处理影响。下一项区分性诊断应捕获真实双样本路径进入文本
首层之前的激活，再与原始参考比较，从而区分差异发生于音频编码/输入拼接还是
文本解码内部；不以继续串行化整模型来冒称微批次一致性已修复。

### 实际 verl 文本首层前缀诊断已派发

`diagnose_omni_actor_prefix.py` 从原失败 actor 的固定输入身份重新构建原六条
DataProto，并验证实际 verl/HF 源码及输入构建、CPU适配 helper 与原actor一致。
使用原顺序和全局1230宽度，六次单样本、三次双样本均调用真实
`DataParallelPPOActor.compute_log_prob`，在文本首层的 pre-hook 保存完整输入
激活、token mask、位置编码和attention mask，再以专用异常停止。
文本attention调用次数必须为0；此处没有完整解码前向或输出概率。

hook的三个本地测试通过，覆盖停止位置、张量副本、异常与hook清理、非法状态拒绝。
新容器 `d71172de780267ba68f43b587e22fe153489bf2db374ebe40eed14092b43b628`，
源码 `60fe38a`，固定原镜像，16CPU/160GiB/2GiB共享内存，无网络设备，
1800秒加30秒预算；已实际inspect运行状态、命令/挂载/资源和355份源码与Git。
见[启动记录](omni-actor-prefix-train6-v1-launched.json)。
原失败容器和原参考逐层诊断容器均重新inspect核对终态，原六份参考激活绑定已接受
的完整trace结果SHA。输出按批次保存真实tensor，独立验收须逐条加载并复算与原
参考及两种微批次之间的误差，不以运行报告中的equal布尔值代替张量比较。
当前仍在运行；没有提前认定音频编码或文本模块是差异源，也未派发权重训练。

### 实际前缀张量独立核验通过

上述精确容器已退出0，309.20秒，运行前后均完成模型文件核验。九次前缀执行
全部停在文本首层之前，未执行任何文本attention。外部核对容器和355份源码，
加载九份实际tensor并独立复算，见
[完整前缀核验](omni-actor-prefix-train6-v1-verified.json)：

- 六条样本在microbatch=1和2下的有效首层输入均与原参考逐元素相等。
- 两种microbatch之间，含padding的完整首层hidden states也逐元素相等。
- 六条的位置ID、cos/sin位置编码和有效query对应的允许attention边均一致。
- 原row2无padding，在单样本时使用隐式因果mask（None），双样本时使用显式
  布尔mask；两者允许关系一致，但表示和算子调用路径并不完全相同。

这些是从保存张量独立计算的结果，不只是运行报告中的成功标志。它们将当前六条
固定输入的差异定位到文本首层入口之后；不证明音频编码在所有batch/输入下都
数值不变，也未证明文本解码的具体首个差异算子。后续可直接使用已保存的真实
首层输入、位置和mask，隔离比较首层attention、输出投影及MoE，不需再次执行
音频前缀。此前局部Q/K/V投影控制只覆盖零padding输入，不代替该真实层诊断。
完整actor一致性仍失败，0.1阈值、冻结权重和不自动重放原则保持不变。

### 首层注意力局部诊断独立核验

`omni-batched-attention-v1` 使用上述真实前缀保存的完整 padding 输入、位置和
mask，调用原 HF 首层 RMSNorm/Attention 与同一 compact adapter，未执行音频
编码或后续 MoE。固定源码 `97fb9fa`、原镜像、16 CPU/12 GiB、无网络设备；容器
`0f9c6a3e2a0dff766a22b8089a3786d64183289a058ad5fbd1142e9d2f2f35b9`
退出0，无OOM，耗时36.49秒。

外部核对容器命令/资源/挂载、120份源码与Git，并加载12份保存张量独立复算：
六条样本的12个阶段共72项microbatch=1/2比较全部逐元素一致，覆盖输入归一化、
Q/K/V投影、Q/K归一化、旋转位置编码、SDPA及输出投影；12份首层attention
输出也全部精确重现原未填充参考。见
[张量独立核验](omni-batched-attention-v1-verified.json)。

因此这批局部重建输入中首层attention没有复现批大小差异；尚不能据此推断所有
层attention不受批大小影响。下一步在真实padding输入上执行完整首层，捕获
post-attention归一化、MoE路由和专家输出，比较原始逐层参考及两种批大小。
该诊断不运行完整48层actor，不改动预登记0.1阈值，不派发优化器或权重训练。

`diagnose_omni_batched_moe.py` 预登记执行原六条输入的六次单样本和三次双样本
完整首层前向，保存包含padding的各阶段张量。通过meta构造原文本模型保留
PreTrainedModel默认的`grouped_mm`专家调度，而不是直接实例化专家模块并落入
eager默认；只物化第0层权重，使用HF实际转换器并逐个核对384份原专家张量。
记录attention、残差、post-attention norm、router logits/weights/indices、
专家gate/up投影、激活、down投影、MoE聚合和最终残差。专家中间值按原token与
top-k位置恢复排序，保持实际grouped计算；对各阶段比较batch1/2并核对原逐层
参考的7个事件。诊断完成和数值一致是不同状态，发现差异时保留张量而非抹去失败。

### 完整首层（含MoE）独立核验

上述诊断容器 `ed0ea2dc72a8c9525dc6e4eff2066708db3d8bb876740540f86b191a179a6fbc`
退出0、无OOM，92.33秒；源码`d07f726`，16 CPU/16 GiB，无网络设备。
外部核对实际容器及121份源码，从九份完整padding阶段tensor独立复算：
六条样本×12个阶段的有效及完整padding张量比较全部相等，12条路径×7个
原逐层参考事件的84项比较也全部相等。独立由专家down投影和路由权重重建
MoE聚合、重建两次残差相加，并核对路由计数与grouped投影offset，均精确一致。
见[首层MoE独立核验](omni-batched-moe-v1-verified.json)。

因此在局部重建的真实输入路径中，差异尚未出现于完整第0层，不能将MoE泛化
认定为故障源。默认grouped_mm路径已被保留和验证。下一步从这批已保存的完整
第0层输出继续逐层定位，遇到首个有效token差异即停止并保留该层中间值；
完整actor一致性仍未通过，不将这个局部结果写作训练基座验收。

`diagnose_omni_decoder_divergence.py` 的后续定位协议固定保留全部六条原始输入和
原配对，读取已核验的完整第0层输出，从第1层开始逐层加载原权重并执行九次
单/双样本调用。每层保存全部padding阶段张量，并比较有效token、完整padding及
原始逐层参考；任何有效阶段或原参考出现首个差异即停止，不继续传播到输出头。
若未出现差异，最多检查到第47层并明确报告未复现，不宣布actor通过。模型文件
按实际触及的原分片校验前后SHA，保留默认grouped_mm与compact attention，
预算5400秒加30秒，16 CPU/16 GiB，无网络设备，不自动重放。该定位不读取任务
标签、不重跑音频编码、不计算输出头或完整actor概率，不派发权重更新。

后续定位已实际启动，容器
`52e739eb5ede5c98e2047fcbb5ff21a5e392b6c145b60c01bbbb17cd435578b0`，
源码`8878b32`、同一固定镜像；已inspect核对命令、挂载、资源及122份源码。
见[启动身份](omni-decoder-divergence-v1-launched.json)。此处只记录运行中状态，
具体差异层和张量独立复核结果待该诊断完成后更新。

### 首个差异定位到第1层MoE输出

逐层诊断已退出0，无OOM，100.10秒，并依协议在第1层停止。外部核对实际容器、
122份源码、保存tensor从第0层输出到第1层输入的连续性、两次残差重建及全部
原逐层参考，见[逐层独立审计](omni-decoder-divergence-v1-verified.json)。
第1层输入、input norm、attention、post-attention norm、router logits/weights/
indices在两种batch大小下仍逐元素相等。原row5和row4各有1个MoE输出元素不同：
最大误差`0.00006103515625`，最终残差各有1个元素不同、最大误差
`0.0001220703125`。其他四条输出一致；六条单样本路径均精确重现原未填充参考。

这是局部decoder重建中首次捕获的数值分歧，不足以证明它独自解释了完整actor
最终概率误差。下一诊断`diagnose_omni_layer1_experts.py`只加载第1层MoE，读取已
保存的真实post-attention norm输入，保留全部padding和原九次调用；捕获按原
token/top-k顺序恢复的gate/up、激活、down、加权结果与聚合。要求路由及MoE
输出精确重现上述完整第1层捕获，才解释子算子差异；不执行attention或后续层。
预算900秒加30秒、16 CPU/16 GiB、无网络设备，不自动重放，不更改一致性阈值。

### 第1层第36号专家up投影出现首个差异

专家局部诊断容器
`27e2efc4052ec98b12ff49a829758524b5540c6364da48c338718d93342f44bd`
退出0、无OOM，72.17秒，源码`af5d90f`。全部九次调用均精确重现前一完整第1层
诊断的router与MoE输出。外部核对123份源码、实际容器、九份完整阶段tensor、
路由计数、逐元素加权和聚合，见[专家独立核验](omni-layer1-experts-v1-verified.json)。

row5与row4首个差异都在有效token66、第4个top-k位置（0起编号3）、专家36，
gate_up通道1531，即up通道763。单样本值`-0.0306396484375`，双样本值
`-0.030517578125`；两者相差一个该量级BF16步长。各自随后只有1个激活元素、
7个down投影元素、7个加权元素、1个最终MoE元素不同；其他四条无差异。
输入与路由仍相等，不能归因为路由top-k改变，也不能据此声称修复或完整actor
误差已被解释。下一步只提取专家36对应的真实输入和原gate/up权重，比较BF16
grouped调用、独立mm及FP32投影，并用FP64点积检查该具体差异通道的舍入位置；
保留原九种输入形状和padding，不改生产实现。预算300秒加30秒、16 CPU/8 GiB。
