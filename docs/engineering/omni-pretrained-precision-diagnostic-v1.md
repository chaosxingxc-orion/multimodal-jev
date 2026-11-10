# 预训练 Omni CPU 概率差异诊断

当前完整 verl 对比的 microbatch=1 对未填充参考最大误差为 0.968058，超过
预登记 0.1；证据见 [部分诊断](omni-pretrained-verl-train6-v1-partial-diagnostic.json)。
原实验及其固定源码保持不变，microbatch=2 结束后仍按原阈值验收。

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
