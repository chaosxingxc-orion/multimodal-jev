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
