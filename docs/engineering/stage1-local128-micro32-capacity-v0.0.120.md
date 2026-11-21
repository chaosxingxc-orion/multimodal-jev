# Stage1 每卡128 / micro32 容量验收

用户要求停止旧训练，并优先验证 micro32、每卡有效batch128。原正式容器已停止，完成40步，保留已提交step32 checkpoint；未自动恢复正式训练。

候选使用原始初始化、同一全量40513条clean TRAIN、新前缀缓存、BF16、DP4（卡4–7），global512，每卡128，micro32，累积4。只派发2步验收。76项固定镜像CPU测试通过，含local128/micro32全批、尾批和fresh resume梯度验证。

NPU在第一次更新前OOM：Qwen3语言骨干RMSNorm计算hidden_states.pow(2)，已分配55.72GiB，预留59.01GiB，剩余661.55MiB，申请710MiB失败。容器退出1；这是设备分配器OOM，Docker的OOMKilled=false不意味着无NPU OOM。没有完成更新，无吞吐成绩。当前训练入口未启用activation/gradient checkpointing；训练use_cache=false，KV缓存不是这里的显存原因。

结论：现有实现下micro32不可用，micro64尚未NPU验证，不能宣称可用。每卡有效batch128可通过较小microbatch+梯度累积表达，CPU算术已验证，但其NPU性能仍需另行验收。下一工程候选优先在语言层增加可配置激活重计算，并做参数梯度/更新一致性与显存吞吐验证；不能仅更改allocator参数就声称解决。

证据：/srv/multimodal-jev-xingchao/operations/context-stage1-local128-micro32-v0.0.120/capacity-summary.json；完整日志在同目录jev-context-stage1-local128-micro32-v0-0-120-collected/supervision/actor.log。旧step32的恢复必须显式处理新global batch的样本游标、schedule与optimizer contract迁移，不把旧step32直接解释成新batch下step32。
