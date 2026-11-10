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
