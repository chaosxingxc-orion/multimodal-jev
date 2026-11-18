# 双信号有限训练的独立归档核验

`tools.reports.context_dual_group_audit` 独立枚举候选对和候选存在性组件，从保存的
实际 logits/prob、固定日程监督、候选 ID 和随机种子重算 G16 动作、奖励、旧策略
log probability、组件权重与 RLOO/GRPO-floor 优势。不调用训练端的
prepare/components/advantages/verify。两个分支分别统计真实 query 的加权奖励和
零信号组件，检查 padding 但不把它计入学习统计。

`tools.reports.audit_context_dual_training` 将上述核验接到以下已有证据：

- Docker inspect 与发布命令、镜像、挂载、设备和环境一致；transport 和归档 hash 完整。
- 原 SLURP 官方清理版 TRAIN 40,513 条及固定 634 步日程，前两个 batch 的真实监督
  和 DP4 分配顺序逐项重算。
- runtime identity、行为/参考策略、实际 wire 输入身份及两分支信用记录相互对应。
- 四卡四 microbatch collective parity、正常进程终态、BF16 执行记录完整。
- 每步 all-rank commit、FP32 master/Adam 合法性、BF16 execution 与发布 policy 一致，
  原生 LoRA adapter 可验证，marker 冻结；保存参数白名单与 checkpoint-scope 对齐。
- 有限两步实际改变 LoRA 和 head，发布策略与激活 ID 对应。

配置入口：`configs/portable/operations/context-dual-training-audit.json`。本地宏分别
绑定 serial/batch4 的 collection、transport、launch、inspect、数据发布、日程、Stage1
policy 和输出；没有自动搜索“最新”checkpoint，也不读取 TEST 反馈。

目前尚无真实 dual NPU 训练归档，因此没有实际审计通过报告。新增核验器已做
18 项本地测试：16 项独立组件重算/篡改拒绝、2 项实际 trainer AST 中 checkpoint
contract 的一致性检查。完整 NPU run audit 的组合路径仍待真实产物验证，不能用
这些测试代替。固定镜像 18 项全部通过、零跳过，容器
`7fba8ca8a97f6c3085e0e95f56e0fd4f1f0f9b0830dac4a674d662c6e8c1c8d3`
准确 wait0，独立收集确认非 OOM、415 文件。结果见
`context-dual-audit-fixed-image-cpu-v1.json`。

后续完整 CPU 归档集成检查见 `tests/test_context_dual_archive_integration.py`。
它实际生成 128 个小尺寸可训练参数、四份 FP32 master/Adam 分片、BF16 policy、
原生 safetensors adapter、两批各 128 条/60 候选的 G16 信用记录和 tar/hash，经过
完整 `audit()`、恢复封包和递归父归档审核。只有数据集/日程入口及存储设备检查
替换为夹具；模型输出、Docker inspect 和恢复记录由夹具提供，没有执行模型、
Docker 内训练或 FSDP。恢复比较使用复制的第二步记录，不能称新进程恢复实测。
完整路径通过意味着这些记录能组合验证，不意味着记录证明了真实 NPU 执行。

首轮固定镜像检查发现实际审计缺陷：训练端对完整候选对向量执行 FP32 sigmoid，
审计端逐标量执行，在 60 候选时发生 CPU 内核末位差异，错误拒绝合法概率。
修复只调整独立审计端：仍自行枚举候选对、重算采样/奖励/优势，但先组装差值
向量再 sigmoid；精确概率、动作、logprob 检查不放宽。训练算法及已封存运行包不变。
失败报告 `context-dual-archive-fixed-image-cpu-v1.json` 保留，不能将其七个负例通过
当成有效拒绝覆盖（正例当时未通过）。v2 对每个负例还核对具体拒绝原因。

修复后固定镜像 24 项全过、零跳过，容器
`bfe33dabe61e0b731aa85973bae71ca49660ce3a6215ef0b701772145a7a51d9`
准确 wait0、独立收集正常退出，419 文件快照。包含完整前缀/恢复归档正例、七类
重签 transport 后的内部矛盾负例，以及原 16 项组件检查。报告为
`context-dual-archive-fixed-image-cpu-v2.json`，共享入口为
`configs/portable/operations/context-dual-archive-cpu-tests-v2.json`。本机缺 safetensors，
没有安装替代依赖或跳过导出检查，完整检查在固定镜像内完成。

新增 `context_master_export_audit` 将四 rank 的同名原参数分片按 rank 顺序拼接，
核对 FP32 master 转 BF16 后与 execution 完全一致，避免两组各自合法但不对应的
保存文件通过检查。已在完成的 GRPO step634 真实归档上验证 128 个参数、6,878,209
个元素全部一致；commit 和四分片 hash 一并验证。证据为
`context-master-export-observed-v2.json`，通过固定脚本和本地宏生成：

```bash
PYTHONPATH=src:.:tools/jev python -m tools.reports.context_master_export_audit \
  --config configs/portable/operations/context-master-export-audit.json \
  --local .local/config/context-master-export-audit-v1.json
```

此检查复组可训练参数，不构建完整基座或复算梯度，也不重跑模型；实际输入和 forward
的正确性依赖已锁定 runtime、原数据入口及运行时行为 gate。CPU receipt 一致不能
独自证明真实模型执行。新进程恢复重放仍是独立必要检查；该脚本输出明确保留
fresh_process_replay_verified=false 和 training_dispatch_allowed=false。

这是一条按具体缺口补齐的工程路径，不改变现有 RL 目标、冻结运行或评测协议。
