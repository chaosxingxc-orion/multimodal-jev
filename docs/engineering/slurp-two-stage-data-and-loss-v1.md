# 全量官方意图发布与混合目标实现

2026-11-12。新主线采用独立的`slurp-official-clean-intent-v1`协议，不修改旧
`slurp-reviewed-training-release-v1`或其拒绝pending的校验。标签依据如实标为
official_clean_v7，不伪造逐条语义审核。该实现没有准入未知槽位负例。

## 完整数据路径

`slurp_official_training.source_population`从已有独立来源包验收凭证出发，检查
两侧完整文件、原请求/目标绑定、60类canonical映射、来源组和音频hash跨角色隔离。
项目已有pending条目按来源组隔离：learning 40,713→40,513候选，selection
9,673→9,226候选。选择侧仍为官方TRAIN内部隔离开发集，不称官方DEV/TEST。

新发布器逐文件重算SHA并解码，检查有限、非空、非全零、单声道16kHz波形；
保存时长、削波比例、RMS和解码波形hash，拒绝跨角色相同解码波形。技术失败单独
隔离并列出原因，不据模型表现排除。没有音频语义复听、官方标签全量正确性承诺。
旧请求、目标、optimization_allowed和审核记录均不修改。

发布于共享目录执行，进程与输入身份：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-official-intent-publication-v1/launch.json`。
预期输出为`datasets/slurp-jev-benchmark/official-clean-intent-v1`；必须以实际terminal、
MANIFEST和独立load重建为准，不以预期路径代表完成。

实际发布已于14:46:38 UTC退出0。全量49,739候选音频身份与解码检查通过，无额外
技术失败隔离。最终learning为40,513录音/9,181来源组/32.2721小时，selection为
9,226录音/2,146来源组/7.3503小时，两侧均覆盖60类。已知pending共647录音仍
隔离。跨角色来源组与解码波形交集为0。学习录音时长中位数2.6237秒，P95为5.248秒，
最长23.36秒；没有超过30秒。此统计不能代替ASR实际processor上下文检查。

独立工具未调用publisher重建函数，重新核对全部录音的原标签、请求/目标身份、
pending来源集合、人口分母、类别计数及保存的音频凭证；[验收结果](slurp-official-intent-v1-verified.json)。
发布manifest SHA256为`38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`。
验收复核保存的波形检查结果，没有再次解码全部音频，未进行语义复听。

loader只允许learning用于optimize，selection用于evaluate，并重新从原始来源
恢复标签与pending隔离，拒绝重新哈希后篡改标签/角色。音频字节的实际训练前使用
还须绑定发布资产；载入清单本身不是NPU预检。

一次Span编译支持direct、structured、scenario、action四种指令视图。候选完整，
先构造输入再从语义真值映射输出索引；标签变化不能改变该样本的输入问题。
目前沿用已登记canonical问题，尚未接纳新8个判别问句候选。
候选干预沿用native/shift1/shift30/permutation317，它同时改变顺序与符号绑定，
不能称已实现二者独立变化；更强的独立绑定接口仍待实现。

## 全覆盖日程

`slurp_instruction_schedule`每epoch每录音恰好一次，按类交错且类内可复现打乱。
没有尾类重复抽样，没有额外逆频率权重；大类剩余人口继续覆盖，不能称严格均衡batch。
视图选择由seed、样本ID及epoch确定，绑定干预由独立hash确定，不依赖gold类别。
跨四轮每来源轮到四种视图；当前两轮预算并不意味着每来源见过全部四种。

global batch候选128，DP4每rank32请求、micro1时累积32。末批补零权重副本，
仍执行必要的分布式前向，但不进入loss分母或有效曝光数。日程身份绑定完整人口、
种子、epoch、批量、world size与全部槽位；恢复游标不能跨身份使用。
准备器`prepare_slurp_instruction_epochs.py`在真实发布验收后生成两轮完整日程并
逐样本重建请求与监督映射；不派发训练。

## 混合目标

`span_hybrid_objective`实现Stage1候选归一化CE和Stage2真实verl有限候选PPO加CE。
任务项和过程项采用独立显式权重；每个来源的总loss按有效来源权重归一，而非按
候选数或padding数平均。多rank各自使用同一全局分母，并补偿DDP/FSDP平均归约；
梯度累积只求和，不能再除以microbatch数。

Stage1不需要伪造rollout。Stage2要求停止梯度的完整behavior，CE不被旧gold概率
重新缩小。lambda以全局更新数做余弦衰减，从1到0.1，默认在80%阶段完成。
实际trainer必须保存预算、全局更新与日程身份；当前loss函数不自行拥有生命周期。

## 实际验证

本地24项数据/日程/旧发布回归测试通过：pending整组隔离、音频失败显式隔离、
标签/角色篡改、selection拒绝优化、gold改变不影响问题输入、全覆盖和零权重补齐。

固定镜像`52ba94e1…`无NPU设备容器完成4项混合目标测试，实际退出0。
另一个固定镜像容器运行4个真实Gloo/DDP rank，micro1/2、各3次连续更新，包含
Stage1→Stage2、全mask rank、零权重样本、Adam文件保存/恢复。与独立单进程
CE及PPO参考相比，最大梯度误差5.960464477539063e-8，参数和Adam状态亦通过。
[凭证](slurp-hybrid-loss-cpu-v1-verified.json)。

这是真实verl loss和分布式归一化证据，尚未接入native actor完整生命周期，
不等于音频/BF16/NPU/vLLM混合训练验收。下一步接入常驻actor和新数据loader，
核验完整输入长度、策略同步与阶段1吞吐，再派发全量训练。

旧P首步容器`a4ae1a02…`已自然exited/0，14:25:41–14:26:36 UTC；未运行独立suite，
保持未验收，不继续小样本12步。原CPU共享梯度诊断观察时仍running，无完整结果。
