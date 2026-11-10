# 预训练 Omni CPU 概率差异诊断

当前完整 verl 对比的 microbatch=1 对未填充参考最大误差为 0.968058，超过
预登记 0.1；证据见 [部分诊断](omni-pretrained-verl-train6-v1-partial-diagnostic.json)。
原实验及其固定源码保持不变，microbatch=2 结束后仍按原阈值验收。

原完整比较现已结束并独立确认失败：microbatch 1/2 对参考分别 0.968058/0.829609，
互相差异 0.40625。见 [终态报告](omni-pretrained-verl-train6-v1-failed.json)。
已确认原容器退出后，使用与已验收 preflight 完全相同的源码快照 `2ff1e6a`
启动完整五条件诊断，精确容器
`0de5e886048040b9fa5489586f4a7fcddea88085701bd1818a8a6a94946e36db`。
16 CPU/160 GiB/18000 秒加 30 秒退出宽限，模型冻结，无网络/NPU，无重试。
保留原六条全部输入、30 次前向，不是旧失败实验的自动重跑，也不是权重训练。
命令和前置证据见 [诊断启动凭证](omni-precision-train6-v1-launched.json)。

已修正工作树 CPU logprob 适配：FP16/BF16 logits 先提升 FP32 再做 log-softmax，
FP32/FP64 保持原精度。近确定 EOS 的小负概率不再因 BF16 归一化变成零。
测试同时检查 autocast 环境、交叉熵参考、梯度和输入不变。
这只解决归一化精度，不证明 0.968 的模型前向差异已修复；尚未派发完整新模型比较。

`tools/jev/diagnose_omni_pretrained_precision.py` 保留原六条 TRAIN 首决策、原声波、
固定 `false`+EOS，不读标签，不训练。使用已通过的完整参考前向验收与模型哈希，
五个条件按下表逐项改变。音频特征仍逐请求计算，不跨请求重算 STFT。

| 条件 | CPU BF16 autocast | 输入音频特征 | 音频特征补齐 | 文本左补齐 |
| --- | --- | --- | --- | --- |
| reference | 关 | BF16 | 无新增 | 无 |
| autocast | 开 | BF16 | 无新增 | 无 |
| fp32_features | 开 | FP32 | 无新增 | 无 |
| audio_padding | 开 | FP32 | 六条公共长度 | 无 |
| text_padding | 开 | FP32 | 六条公共长度 | 六条公共长度 |

每条输入运行全部五个条件，共 30 次冻结前向，不能选择性删除失败输入。
这是相邻单因素诊断链，不是完整析因设计，不能排除因素交互。
每次保存两个响应位置的全词表 logits、其哈希、两种归一化结果、相邻差异与参考差异。
最长无文本 padding 输入也保留，作为控制。输出 diagnostic_complete 不表示 parity 通过。
定位原因后还需完整六条实际 verl 微批复验，仍用原 0.1 阈值；正式 HF/vLLM/NPU
一致性与重要性比率另行验证。

运行前先执行 `--preflight-only` 验证真实 processor 和全部输入条件；完整权重加载须等
当前完整 CPU 比较终止，核验原终态并登记新源码快照、镜像、输出路径和预算。
按前一完整六条前向 1831 秒与 padded microbatch=1 的 2135 秒估计，30 次约需数小时；
拟定 16 CPU、160 GiB、18000 秒加 30 秒退出宽限，无网络/设备，无自动重试。
预算是派发上限而非承诺耗时，正式派发另存精确容器 ID。测试容器不挂载模型或数据集。

## 已执行验证

源码 `2ff1e6a`：本地 22 项测试通过；固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
中另覆盖真实 verl DataProto，共 24 项通过，耗时 29.43 秒。
测试容器 `5d2be3c814b8303e71c4b4a5b4879c12625a1438e4b06d9bba680349c54d3b3e`
退出 0，2 CPU/8 GiB/300 秒上限，无模型数据挂载。

同一源码的真实 processor 预检查容器
`6db2e4c565fa7f80c0fb78d7b3d43aa236f4e553bd52f444494800534b18de4c`
已启动，2 CPU/16 GiB/1800 秒上限，模型与数据只读；不加载模型权重。
预检查已退出 0，完整六条、五条件输入准备通过，源码清单与容器资源已独立核验，
见 [验收记录](omni-precision-preflight-v1-accepted.json)。这不代表完整模型诊断完成。
精确启动信息与观测见 [测试/预检查凭证](omni-precision-tests-preflight-v1.json)。

## 已保存 logits 的独立分析

新增 `tools/jev/analyze_omni_precision_logits.py`：只读取已完成记录和保存的张量，
核验 identity/记录对应关系、张量 SHA、形状、有限性、BF16 精度，并重新计算
两种归一化及相邻条件的全词表差异。允许读取运行中的连续记录前缀，不重跑模型。
完成 30 个条件也只标记 artifacts_complete，不替代容器终态、源码/权重核验，
不自动判定 actual-verl parity 通过。输出必须放在原 run 之外且不得覆盖已有报告。

[首份部分报告](omni-precision-train6-v1-logit-audit-partial-01.json)覆盖第一条输入的
reference/autocast 两次前向：两位置的全词表 logits 完全一致，相邻最大差异为 0。
两份 `.pt` 文件 SHA 不同不能作为张量不同的证据；已实际加载张量比较。
FP32 归一化的 EOS log-prob 为 -0.000034808505，原生 BF16 归一化为 0；
两 token 最大归一化误差 0.0053167343。该观察不能外推其余五条或解释此前全部偏差。

本地系统 Python 的 6 项测试通过，覆盖篡改哈希/概率/身份、缺失中间记录、
完整记录不等于 parity 验收，以及“响应概率相同但其他词的 logits 不同”的反例。
与报告同时保存的[运行状态](omni-runtime-observation-20261110-precision-partial-01.json)
来自精确 Docker ID 的实时 inspect；当时 CPU 诊断、SLURP 和 FSC 基线均仍在运行。

[第二份部分报告](omni-precision-train6-v1-logit-audit-partial-02.json)增加首条输入的
fp32_features 条件：保存的全词表 logits 与 autocast/reference 完全一致。
目前仅覆盖 3/30 条件，不能据此排除其他输入的特征精度影响；音频/文本补齐及
实际微批形状仍需后续完整诊断和 actual-verl 复验。

[第三份部分报告](omni-precision-train6-v1-logit-audit-partial-03.json)增加首条输入的
audio_padding 条件，两个响应位置的全词表 logits 仍相同。仅完成 4/30 条件；
公共长度对具体输入可能不增加 padding，因此此结果也不能外推为音频补齐普遍无影响。

## 首条输入的五条件差异

[第四份部分报告](omni-precision-train6-v1-logit-audit-partial-04.json)覆盖第一条输入全部
五条件（5/30）。reference/autocast/fp32_features/audio_padding 的全词表 logits 相同；
text_padding 对前一个条件的响应 log-prob 最大误差为 **0.1445865631**，超过原 0.1 门槛。
首条原 prompt 1074 token，公共宽度1228，因此增加154个被 attention mask 排除的
左 padding；有效 token 的 position_ids 仍从0递增。音频条件与前一个 cell 完全相同。
两个响应位置的全词表 logits 最大误差1.125，全词表 log-prob 最大误差1.24999237。
实际 false token 的 FP32 log-prob 从 -4.4115667343 变为 -4.5561532974。

这提供了文本补齐相关差异的具体证据，但没有区分 attention kernel 数值、MoE计算
形状或其他内部实现原因，不能将 BF16 归一化误差当作该变化的解释。仍需全部六条、
实际 verl 微批1/2及后续HF/vLLM的独立检查；不修改原门槛、不重跑原失败实验。
观测时同一精确容器继续运行并进入 row1/reference。

进一步的[mask/rotary/SDPA机制检查](omni-padding-mechanism-v1.md)确认合成输入上
有效因果边与位置完全一致，但BF16 SDPA在增加masked位置后仍可产生小数值差异。
这不是预训练模型根因或已解决证据；实际专家路由与逐层激活仍待验证。
