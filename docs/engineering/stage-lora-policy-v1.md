# 分阶段 LoRA 配置和继承规约

Stage1 使用 InfoNCE 单一损失；Stage2 使用排序/存在性双信用 GRPO。两阶段分别声明 `lora`，通过 `stage2.reuse_stage1_adapter` 显式决定是否复用；复用时还要求规范化后的完整执行配置相同。当前版本配置见 `configs/experiments/context-two-stage-v0.0.11.json`；已封存的旧包不原位修改。

比较项包括 rank、alpha、实际解析后的完整 target_modules、dropout、bias、RSLoRA/DoRA、fan_in_fan_out、执行 dtype。候选模块顺序和数值 8/8.0 不产生新 adapter；学习率、warmup、loss 和阶段名不属于 LoRA 执行配置。

| 情况 | Stage2 actor | 可训练参数 | reference |
| --- | --- | --- | --- |
| 配置相同且 `reuse_stage1_adapter=true` | 原 base + Stage1 adapter，继续更新 | 同一 adapter + Jev head | Stage1 终点的固定完整输出 |
| `reuse_stage1_adapter=false`，配置可以相同或不同 | 原 base + 冻结的 Stage1 adapter + 新 Stage2 adapter | Stage2 adapter + 继承的 Jev head | Stage1 终点的固定完整输出 |

不复用时采用加法残差组合：`W_eff = W_base + s1 B1 A1 + s2 B2 A2`。新 adapter 使用随机 A、零 B，使切换时保持 Stage1 函数；不得换掉 Stage1 后从原始 base 开始 RL。阶段切换采用新的 optimizer；相同配置复用的是权重和 adapter 身份，不是沿用上一阶段的 Adam 状态。BF16 舍入与计算顺序仍需实际检验，代数等价不等于训推数值验收。

checkpoint 保存当前可训练 adapter/head/optimizer；冻结 Stage1 以不可变策略 ID、路径及文件哈希绑定，可独立共享存储。推理必须加载全部有效增量和同一 head/marker/processor；adapter 单独不是完整策略。禁止合并并重复保存主模型。恢复和 reference 缓存同时绑定阶段转换契约，不能把配置不同的 adapter 当成相同策略复用。

当前实现：`context_stage_lora.py` 提供配置规范化、身份、继承计划和准入检查，正式 Stage1 封包绑定实际初始策略的 LoRA。当前两阶段都是 q/v rank4 alpha8，配置显式设 `reuse_stage1_adapter=true`，复用已验收的单 adapter 模型路径。新 adapter 的组合 actor/export/vLLM 路径尚未通过验收，准入明确拒绝；不能把支持配置声明描述成已支持双 adapter 的正式 NPU 训练。

验收必须覆盖：同配置不重复创建；不同配置冻结 Stage1 且只有新 adapter/head 有梯度；切换前后初始输出一致；组合权重导出/加载及恢复一致；HF/vLLM 在真实音频、batch32 和尾批上的概率一致；checkpoint 不含主模型。未通过的路径不得派发正式训练。

`reuse_stage1_adapter` 必须是 JSON 布尔值，不接受缺省推断、字符串或0/1。配置不同却设置 `true` 在封包时拒绝。设置 `false` 时即使配置完全相同也分配新的参数对象和独立 optimizer 状态，保留 Stage1 冻结增量。开关纳入阶段转换身份和恢复契约。

固定训练镜像CPU73项已通过，包含真实PEFT三个分支的初始输出一致、Stage1冻结与Stage2梯度检查。证据 `/srv/multimodal-jev-xingchao/operations/context-stage-lora-cpu-v0.0.10/report.json`。双adapter的vLLM/NPU准入仍未完成。
