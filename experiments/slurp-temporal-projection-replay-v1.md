# SLURP 固定音频投影回放诊断

目标：在同一条学习录音、同一合并策略与同一批固定采样响应上，检验 HF 使用 vLLM 实际音频投影后，响应 logprob 差异是否仍存在。此为定位干预，不训练，不生成新的奖励或替代 rollout，不将干预结果作为 RL 准入。

输入是[逐层探针](slurp-temporal-audio-stages-v1.md)保存的 `[32,1024]` 投影、对应三个提示、六条采样响应；已核实这些采样响应与[原概率探针](slurp-temporal-rl-parity-v1.md)一致。仅使用已审核 learning 录音，没有 selection/TEST。源数据、checkpoint、投影哈希和展开后的 prompt IDs 验证通过后才执行。

同一 HF 模型逐响应比较三个臂：原生音频路径、回放保存的 HF 投影、回放保存的 vLLM 投影。全部 BF16 完整前向，报告各自与原采样 logprob 的最大误差；HF 回放相对原生前向是干预管道控制。保留实际 projector 与回放 tensor 的误差、替换调用次数和逐 token 概率。若控制本身不稳定，则不能将 vLLM 回放的变化完全归因于投影。

物理卡仅 4，固定镜像不变，无网络、无优化器。工具语法检查通过；最终依据真实运行记录验收，不能以脚本存在视为实验通过。

运行根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-temporal-projection-replay-v1`。
容器：`d1e1f76eceb1c8ed760a3beece025a3705d117e3092b7745653b1640f7e52e5d`。
详见[启动凭证](../docs/engineering/slurp-temporal-projection-replay-v1-launch.json)。

## 完成结果

容器退出 0、无 OOM、仅物理 4 号卡。独立检查所有输入 pins、逐响应身份及 token、原采样概率、有限性和分母后重新计算，见[核验凭证](../docs/engineering/slurp-temporal-projection-replay-v1-verified.json)。三个提示、六条响应、36 个响应 token：

| HF 前向音频投影来源 | 对原 vLLM 采样概率的最大绝对 logprob 差 |
| --- | ---: |
| 原生实时音频路径 | 0.13464361429214478 |
| 保存的 HF 投影回放 | 0.13464361429214478 |
| 保存的 vLLM 投影回放 | 0.003516079857945442 |

HF 自身投影回放与原生前向逐 token 概率完全一致；被替换的 HF 实时投影与保存的 HF 投影逐元素相同。控制通过。对这条录音，仅替换音频投影已使最大概率差降到原参考门槛 0.02 以下，支持音频编码数值路径是当前差异的主要贡献因素；剩余语言路径误差非零。不能据此声称所有录音、微批或训练状态均一致。

后续修复方向是让两端实时音频编码采用一致实现，并检查权重、padding、dtype 和参数更新的对应关系；继续直接消费音频，不能将保存的投影作为正式 actor 的输入、用缓存隐藏模型更新，或替换 old logprob 绕过验证。优先在当前两个来源/六提示复验，再扩到短音频边界及实际微批。共享编码实现仍须真实采样验证，不因本次干预成功就准入 RL。
