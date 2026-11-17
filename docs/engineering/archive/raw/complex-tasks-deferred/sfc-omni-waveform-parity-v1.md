# SFC actor 与 vLLM 波形边界核查

固定实际 processor、完整既有 TRAIN6/78 个决策，CPU 检查已完成：

- 原 `prepare()` 的 raw 波形路径：78/78 token ID 相同，78/78 特征不一致。
- HF 逐段补零到 160 samples 倍数后，按 vLLM 模型 BF16 边界比较：78/78 token
  相同、78/78 特征最大误差为 0。没有变更特征误差容差 `atol=rtol=1e-5`。
- 全部音频帧与补零后长度一致，全部 78 输入仍容纳于 4096 上下文并预留 1024 输出。

容器 `dd91ad6517bd93c46949fd48c7cede9974a799a5d9089f8b95cb9e860e08b8ea`
退出 0，耗时 58.26 秒，源码 `3d95dc1`。固定镜像 2 CPU/16 GiB/900 秒，无网络、
设备、模型权重加载或标签读取。覆盖/源码/帧数算术已独立核验，见
[完整结果](../../../sfc-omni-processor-parity-train78-v1.json)。

## 工程处理

`omni_actor_batch.prepare(..., waveform_boundary='vllm_hop')` 新增显式对齐选项，
记录每段原始与补齐采样数。默认 `raw` 保持历史实验输入；运行中的固定源码不改。
单元测试覆盖原样本保留、零尾巴、多音频顺序、重复对齐幂等、非法 dtype/形状/数值。
本地相关 24 项测试通过。

v2 以真正的 `prepare(vllm_hop)` 替代 v1 内联补零，再次验证完整 78 决策，
不是运行模型或重放采样；启动见 [v2 凭证](../../../sfc-omni-processor-parity-train78-v2-launch.json)。
v2 已退出 0，实际 actor prepare 路径的 78 项 token/特征对比全部通过，最大特征误差 0，
见 [v2 独立核验](../../../sfc-omni-processor-parity-train78-v2-verified.json)。
后续采用新选项需重新绑定输入协议和源码，再验证 actual actor logits 与 rollout。
当前只比较转换到相同 BF16 后的特征，不证明 CPU autocast 或模型输出等价。

## 对既有实验和后续 RL 的影响

FSC/SLURP/SFC 冻结 vLLM 原生评测不因这项 actor 诊断而重跑。
原 CPU 五条件诊断继续按原 raw 输入定位 HF 内部差异；本次不能宣称已找出
原 verl 0.968/0.830 误差的原因。

尚未派发的 K2 sampling 队列使用固定 raw actor token 校验；本次完整 78 token 均相同，
所以没有 token 不匹配的证据要求停止它。其生成结果若通过原验收，可用于冻结
生成质量和 TRAIN 奖励分布诊断；**不能直接认证为 raw actor 的 on-policy 概率**。
用于 PPO/GRPO 前须以相同 waveform boundary、特征 dtype、position 和模型身份重算，
在真实采样响应上通过 logits/logprob/importance-ratio 比较。保持禁用权重训练。
`capture()` 的 token 一致校验是必要条件，不替代此多模态概率门槛。
