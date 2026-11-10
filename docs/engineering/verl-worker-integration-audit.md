# 原生 verl worker 接入差异

从固定 NGA 子镜像的已停止容器只读提取八份源文件，未改镜像或运行中的工程。
镜像为 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
已记录 verl 版本为 0.8.0.dev0、上游提交为
`e9405d70b77d56a23e72bb2e40a17b906856e913`。镜像内实际文件哈希见
[源码记录](verl-worker-source-audit-v1.json)；判断以这些实际文件为准，不能假定
镜像内容与上游提交完全一致。主要 worker 文件标明 Apache-2.0，`utils/device.py`
另有 BSD 风格许可说明；本次仅阅读，不把源码复制进 Git。

| 固定源码位置 | 观察到的行为 | 对接要求 |
| --- | --- | --- |
| `workers/actor/dp_actor.py:149` | 转发 `multi_modal_inputs` 到模型 | 可复用音频特征与 mask；需要实际 worker 前向校验 |
| 同文件 `:155` | 自动混合精度使用配置 dtype 和检测到的 device | 显式固定执行设备与 dtype，不把 CPU 自定义 actor 结果外推到 NPU |
| 同文件 `:549`、`:590` | 一个 minibatch 且一个 epoch 时，默认以当前 log-prob detach 作为 old log-prob | 接入现有算法需显式 `use_rollout_log_probs=True`，保留独立 rollout/actor 数值门槛 |
| 同文件 `:576`、`:660` | 静态微批按微批数量平均，动态微批按样本数缩放 | 变长 response 下不等于当前全局有效 response-token mean；需修正权重并验证梯度/参数一致性 |
| 同文件 `:504`、`:671` | 切 train 模式，逐 minibatch 调 optimizer step | 验证 dropout 为零；显式处理零优势，不能让已有 AdamW 动量在“无更新”批次仍前进 |
| `workers/rollout/vllm_rollout/vllm_async_server.py:430` | generate 仅接 image/video，并构造 TokensPrompt | 需要音频请求及正确的一次占位符展开；不能只在客户端增加 audio 字段 |
| `experimental/agent_loop/agent_loop.py:95` | server manager 同样只转发 image/video | manager/server/loop 要同时扩展，传递采样身份与音频特征 |
| 同文件 `:642` 及单轮 loop | 默认多模态处理面向图像/视频 | Qwen ASR 需音频 processor 路径、attention-mask 累计 position IDs；不得走视觉位置编码 |
| `workers/fsdp_workers.py:155`、`:242` | 建立进程组、设备网格并按并行度调整 batch | 多 rank 下必须先组成完整配对奖励/GRPO 组，再验证分片 loss 缩放和全局 token 计数 |
| `utils/device.py:114` | `get_device_id` 调设备模块的 `current_device()` | CPU 分支不能假定支持此调用；不得通过挂载忙碌 NPU 绕过 CPU 验证问题 |

`ActorConfig.global_batch_info` 还提供全局 token/样本数量字段，因此微批缩放审计
必须同时检查实际传入的全局计数与 loss 实现，不能仅凭局部缩放表达式断言所有
原生配置均错误。当前自定义 actor 使用空 global_batch_info 并显式按 token 加权；
迁移时不得同时启用两种归一化而重复缩放。

补充实测：原生 actor 的 CPU 前向在 log-prob 阶段失败，固定版本的
`utils/torch_functional.py` 根据已安装算子而非 logits 设备选择 NPU 交叉熵。
见[失败与显式 CPU 适配协议](../../experiments/native-verl-actor-forward-v1.md)。
v2 在独立进程中显式替换 log-prob 分发，已通过真实音频前向对照，最大误差
3.0517578125e-5；见[结果](native-verl-actor-forward-v2-result.json)。不得称为
未修改 worker 的成功执行。真实模型梯度/AdamW 对照仍在运行。

后续[原生 update_policy 验证](../../experiments/native-verl-update-policy-v1.md)
已登记并排队，必须等待上述梯度对照成功。它直接调用 compute_log_prob 和
update_policy，显式保留 rollout 概率、验证零 dropout，并与同一初始化器的
参考梯度/参数/优化器状态比较；CPU 设备与 log-prob 适配均明确记录。
本次仅验证单个完整微批；变长微批缩放、零优势动量处理与真实分布式模型路径
仍是独立的必要验收，不能用这个队列代替。

[两进程 Gloo 损失验证](../../experiments/distributed-verl-loss-v1.md)已通过：
固定 core_algos.py 的 global_batch_info 在不等 token 分片和不等微批数下恢复
全 batch 梯度，错误的局部平均负例能明显区分。此结果仅覆盖损失与 collective，
不覆盖模型 forward/backward、原生 update_policy 或分布式状态保存恢复。

以上是源码发现，不是原生 worker 的运行结果。目前 `rl_update.update_actor`
已保留真实 rollout log-prob、更新前概率校验、全 batch GRPO 分组、按有效响应
token 加权的微批梯度与零优势 no-op。接入原生 worker 时不能静默改变这些行为，
否则原有数值验证和对照协议失效。

验收顺序应为：先验证原生 actor 的音频前向与现有实现逐 token 一致，再验证
不同响应长度、微批边界下的梯度和一次 optimizer 更新；之后跑两个实际 rank，
检查全局组身份、同步参数和保存恢复；最后接异步 vLLM 音频 server，使用新策略
做下一轮采样并核验旧概率与导出身份。仅初始化一个 Gloo rank、构造 DataProto
或导入类都不能宣称分布式训练完成。现有 45 份 CPU 预算对照配置不会因本审计
自动切换执行后端，任何切换需先通过上述数值与状态验证并重新登记运行版本。
