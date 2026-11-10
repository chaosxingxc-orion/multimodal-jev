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
