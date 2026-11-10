# 全六条学习输入的不可达 K/V 移除候选

原CPU BF16诊断中，五条新增左padding样本都超过0.1响应logp差门槛。row0的
冻结attention分解和14项形状对照进一步证明，仅在有效K/V前增加被屏蔽的位置
就能精确复现该行attention差异；填充值与Q轴变化没有解释这组差异。
据此测试一个完整模型候选，而不把局部attention证据直接推广为全模型修复。

`omni_compact_attention.compact_text_attention`在单进程诊断作用域内包装真实
Transformers SDPA接口，仅处理Thinker text attention。对显式布尔mask，逐样本
移除所有query都无法访问的key及对应value，同时裁剪mask的key轴；query、位置
编码、输入token、响应布局、音频输入、权重及MoE计算保持原定义。重建原坐标mask
验证有效边完全一致。无mask直接走原实现，其他audio attention也直接走原实现。
异常退出恢复原注册函数；拒绝嵌套、cache形状、浮点/多头mask、dropout与位置bias。
这是显式候选，不修改正式actor默认路径，也不用于并发服务或权重训练。

9项本地CPU测试通过：非连续排除位置、不同样本的不同有效keys、全mask query、
前向与梯度一致、排除K/V梯度为0、不支持模式拒绝、其他模块旁路以及异常/嵌套
注册清理。测试采用FP64小张量验证数学语义，不证明预训练BF16数值修复。

`diagnose_omni_compact_keys.py`绑定已成功验收的原精度容器、固定HF/SDPA源码SHA、
模型资产和原六条学习输入。仍使用原始waveform边界、共同音频feature宽度、
左padding prompt、false+EOS固定诊断响应与CPU BF16 autocast。每条只做一次候选
完整前向，检查48层实际调用与保留key数，保存完整响应logits，分别比较原
audio_padding和text_padding的已保存结果，FP32归一化。六条全部执行，即使某行
不通过门槛也保存剩余观察；门槛保持0.1，不以平均值掩盖失败。

先运行真实镜像输入/注册预检查，2 CPU/16 GiB/1800秒加10秒，无网络设备，
不加载模型权重。完整候选预算同一固定镜像、16 CPU/160 GiB/10800秒加30秒，
只读源代码、模型、数据和原实验，无网络/设备，不读取监督值、不生成任务答案、
不使用优化器。预检查成功后独立验证容器与源码，再create持久记录精确ID后start。
原逐层诊断可继续运行，不改其快照；新候选不覆盖旧失败，也不自动重试。

结果解释须区分：候选成功执行、六条响应概率门槛通过、真实verl actor一致性、
NPU/rollout一致性、优化器更新及任务收益是不同要求。后五者不能由启动或CPU
预检查推断。即使候选通过，逐样本压缩的性能、梯度路径和分布式行为仍需实测。

## 实际预检查与完整派发

固定源码`64e530d`的CPU预检查容器
`a75d16ede6f4b730dc8a76aca877d8194d35b88a41bb4aff79062ade2a8bc71c`已退出0。
协调者重新核对精确容器命令/资源/挂载/环境、源码与Git一致、原六条身份/响应、
模型文件清单、padding计数`[154,252,0,251,519,402]`和原30条件成功证据。
真实HF registry安装/恢复检查通过，未加载模型权重。
见[预检查验收](omni-compact-keys-preflight-v1-verified.json)。

随后使用同一只读源码快照派发完整候选容器
`742651b35fe69b6f7679a56136fd3643c03c07f915799937f70d6e6d87bd9da4`，
intent SHA `07902e61933c77a245530c681110a9cf39842485a0c203159244caea301b42e9`。
实际inspect为running，核对16 CPU/160 GiB、10800秒上限、无网络设备和完整命令。
协调者已记录启动时宿主可用内存与共享盘空间；原逐层诊断继续运行。
见[完整派发身份](omni-compact-keys-train6-v1-launched.json)。当前没有完整候选结果，
不能将输入预检查或容器启动记为概率一致性通过。

## 首条完整模型候选结果

row0已完成，完整冻结前向364.03秒。独立协调者复核实际容器/源码和保存logits，
48层均记录1230个query、原1230个key缩减为1076个可达key。左padding输入不变，
候选的完整响应logits与原audio_padding（无新增text padding）参考逐元素相同，
响应logp最大差0；与旧text_padding相比logp最大差0.1445865631，完整logits最大差
1.125。见[首条独立审计](omni-compact-keys-train6-v1-row00-audit.json)。

这表明候选在首条原始CPU完整模型样本上恢复了参考数值；原逐层诊断中出现的后续
MoE分歧并未在这条修正后的响应logits中留下差异。仅此还不能推定全部层内部完全
一致或其余五条通过；当前进入row1，原六条0.1门槛、实际verl/NPU与训练仍未验收。

row1随后完成，前向343.70秒。独立复算保存logits确认：252个左padding保留在
输入中，48层K/V各由1230缩至978；候选响应logits与原无text-padding参考再次
逐元素相同，响应logp最大差0。旧text_padding相对该结果最大logp差0.3880100250。
见[第二条独立审计](omni-compact-keys-train6-v1-row01-audit.json)。当前进入row2，
六条完整成功终态尚未取得，不能以两条成功放行真实verl完整模型比较。

## 全六条候选通过

原精确容器退出0，应用主体耗时2265.34秒。外部协调者重新inspect并核对完整源码
与Git身份，独立加载六条候选及两个原始条件的保存logits，复算响应概率。
**六条候选的完整响应logits均与原无新增text-padding参考逐元素相同**，响应logp
最大差为`[0,0,0,0,0,0]`，全部通过原0.1门槛；六条×48层记录也逐项核对。
见[完整候选验收](omni-compact-keys-train6-v1-verified.json)。

该结果支持移除不可达K/V在原六条CPU完整前向上恢复参考响应数值。未据此声明
实际verl混合microbatch、NPU/vLLM、优化器或任务质量通过；下一步是原六条顺序
`[2,0,5,1,3,4]`、真实verl microbatch=1/2的独立比较。模型资产清单及固定运行器
结束前的资产校验保留，本次外部审计没有再次读取全部权重文件。
