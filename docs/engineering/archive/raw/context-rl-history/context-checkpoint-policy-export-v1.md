# 从训练 checkpoint 直接导出联合策略

2026-11-14。连续RL需要在每次更新后把LoRA与浅层head一起交给vLLM，不能只更新
LoRA而遗留旧head，也不能为了打包权重每次重新加载完整Qwen骨干。

`context_checkpoint_policy.export_checkpoint_policy`从经过验证的源联合策略读取冻结
marker、processor与contract，替换完整组装的BF16 head/LoRA张量。拒绝缺参数、
混入marker、形状变化、FP32执行张量和未认证的contract；FSDP wrapper路径按既有
canonical规则处理。完整新policy写入临时目录，重新读取验证后fsync并发布。
源policy不可变，新policy ID绑定全部执行资产和契约。它不恢复Adam，不授予训推准入。

`train_context_instruction_npu.py`的checkpoint提交新增完整联合policy目录及manifest
hash。每个保存点的四rank分片、BF16执行导出与policy共同受提交记录约束。
已通过的远程prefix/replay快照没有改动；新增保存分支尚需实际全循环NPU验收。

## 实测

21项相关checkpoint/FSDP/export本地tests通过，包含新head/LoRA重载、冻结marker
保留、不可覆盖、错误张量/契约拒绝。

固定CPU容器fcf9199100bb8fe5ba02f8390397aa66f8d16c914f47dbe0b151cdfc23b6326e
exit0；独立collector逐文件hash、128完整BF16张量与真实RL DP导出对照、冻结marker
及原生PEFT适配器验证均通过。结果`context-checkpoint-export-cpu-v1.json`，新policy ID
a937aaff0241468247cfb14e7a7b463cac525822866f2661369367c60676dd9d。
此CPU产物保持源contract，包括历史通用request_version和test_only；不把它称为正式策略。

并行运行的b01da14a则通过原HF加载/导出路径修正exclusive_event请求版本，随后执行
HF/vLLM256。两条路径用途不同，不要求policy ID相同：contract变化本身必须改变身份。
其exactwait27582及driver292736仍在等待，未报告新概率验收。

## 后续依赖

- 真实修正contract的HF/vLLM256终态及独立收集；
- 正式token与runtime准入记录，以及完整Stage1源策略导出；
- 常驻vLLM或明确进程切换方式下的joint-policy激活、下一批新行为概率、连续RL更新；
- 完整selection曲线和冻结FSC迁移。

不能用本次CPU打包成功代替上述生命周期与任务效果。新增actor记录每个micro的
behavior_parity_history，后续收集可报告全窗口最大误差；当前883b000e历史日志只有
各次更新最后micro的误差，不追溯伪造缺失记录。

本轮镜像到期observed_unix1794660721仍running/nonOOM，13824/40513、1537.15秒。
SSH复用通道曾拒绝新session，ssh自动退回独立连接后命令成功；不属于容器失败，
未重启镜像、RL导出或旧审核任务。
