# 精确集合效用：PyTorch可微候选

继续[精确期望梯度设计](63-exact-singleton-reward-gradient.md)，新增独立候选
`tools/reports/exact_singleton_f1_torch_candidate.py`，没有改动src、tools/jev、任何冻结包
或训练队列。完整单正例、独立Bernoulli动作语义不变；该候选尚未注册到verl actor。

候选使用B×C概率张量，以负例入选数的Poisson-binomial递推计算每query的精确期望
集合F1，并由原生torch运算自动求导。输入直接接收调用方产生的FP32/FP64概率，
不自行再做一套sigmoid，也不clamp概率。BF16执行模型应继续按冻结概率协议先产生FP32概率。

输出loss为负期望效用的加权和，乘world_size再除global_real_requests，以匹配DP平均
和microbatch累积。padding只能权重0；完整真值必须恰好一个正例且所有候选observed。
缺标、多正例、全负例直接拒绝，不自动删候选或填负例。候选没有采样ratio、组标准化、
clip或KL/辅助项；将来如派发，组合目标、尺度、优化器与比较预算须先单独冻结。

固定60候选时递推算术量O(B C²)，自动求导保留的中间张量也是该量级；实际运行速度、
NPU内存与通信尚未测量，不能从没有G16采样直接推出整体加速。共享reference与评测
一致性仍需按新版本目标定义，不能借候选跳过当前实验保存恢复或训推检查。

## 已验证范围

本地torch 2.10.0+cpu，非固定训练镜像、非NPU。新增11项与既有标量11项共22项检查通过：
小空间所有动作的期望和梯度、probability及sigmoid组合gradcheck、分片梯度归一化、
padding零梯度、候选置换、单候选、端点、BF16上游张量和不完整真值拒绝。
分片检查是单进程代数模拟，不是实际分布式通信验收。

进一步读取第74步已绑定TRAIN输入/标签的behavior及reference共256分布、15360坐标，
对照独立纯Python反向递推：

| dtype | 最大reward误差 | 最大prob导数误差 | 最大logit导数误差 |
| --- | ---: | ---: | ---: |
| FP64 | 2.22e-16 | 3.33e-16 | 5.55e-17 |
| FP32 | 4.98e-7 | 1.12e-6 | 1.39e-7 |

证据和源码hash见[CPU核对](context-exact-f1-torch-cpu-v1.json)。标量实现此前另有独立
多项式积分验证；本次没有执行模型更新，不能称当前模型已经受益。

## 与正式实验的边界

G16的组内无差异会让GRPO/RLOO优势为0，精确积分可避免有限采样没有见到稀有动作，
但其sigmoid logit梯度仍乘p(1−p)，错误饱和仍可能非常难纠正。原始期望reward也没有
GRPO的组尺度处理，因此梯度大小和学习率不可直接类比。候选不是新的通用RL算法发明，
也不能用于缺少可计算真值的环境任务。

当前保持GRPO与RLOO完整曲线。只有在完整效果与成本报告支持下一轮比较时，才冻结
精确效用混合目标、尺度规则与起点，验证固定镜像和真实NPU路径后接入统一工程。
本候选不成为现有基线继续运行的条件。训练派发仍为false；目标仍包括学术复现、
语音Jev任务构造与跨任务收益，CPU候选通过并不完成这些目标。

## 固定训练镜像CPU核验

随后在固定镜像`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
执行相同22项检查，全部通过，零跳过。容器
`b1d620dddd0515a6a134bef115db53c5a4a5cd4390534eee660efcd717a5012a`已exit0、非OOM。
限定2CPU/8GiB、network none，无NPU设备及模型权重挂载；只读驱动库用于镜像依赖，
不执行设备运算。热目录经host/container两侧存储核验，均写本地/data；日志轮转开启。

实际360项源码/测试/runner快照hash、容器镜像与命令、设备及网络范围均独立核对；
终态、JUnit、测试日志和实际FILES清单收集后再hash比对。证据见
[固定镜像报告](exact-f1-tools-fixed-image-cpu-v3.json)。没有mock torch数值计算；
DP分片仍为单进程代数模拟，模型/NPU/verl更新和完整真实概率面板不在本次容器测试范围。

首次诊断包遗漏__init__依赖reward，第二次存储检查误用了host home源路径而非容器/root，
两次均在pytest之前exit1、非OOM。失败现场分别见
[v1](context-exact-f1-fixed-image-failed-v1.json)、[v2](context-exact-f1-fixed-image-failed-v2.json)。
v3补齐包依赖并检查实际挂载目标，没有改候选算法、测试、阈值或正式训练运行包。
准确docker wait确认终态后才收集，不因观察超时重启。
