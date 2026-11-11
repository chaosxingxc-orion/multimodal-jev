# 清理后 SLURP 的逐决策训练适配

2026-11-11。实现见 [slurp_training.py](../../src/multimodal_jev/slurp_training.py)。完成审核投影和现有 SFT/RL 入口保护；尚未发布可训练数据、运行 SLURP SFT 或获得模型质量结果。

## 实现范围

`project(request, target, tasks, reviews, source_release_sha256)` 消费 clean source bundle、固定任务定义和独立审核记录，返回可用原子 `JevExample` 及逐任务状态。支持 scenario/action Choice 和类型存在性 Noul，均要求显式 thinking。

审核记录绑定原始请求、原始 target、完整任务定义的 SHA256，并包含审核者、证据引用、状态和确认值。负例与正例同样需要审核；任务列表不依据当前 gold 选择。原始 clean target 保持禁止优化，新监督单独形成。

- `reviewed` 且绑定有效才转换训练样本。
- `pending/conflict/unavailable` 不进入训练样本，因此不贡献 SFT loss 或 RL reward；不会转成 actor 的 null 或 false。
- 审核值与原标注不同时拒绝；纠错需要独立数据修订。
- 新样本绑定音频、schema、thinking、target、来源组和音频 SHA，修改后拒绝。审核内容及原始实体值不进入 actor 输入。

`sft.select_training`、显式 `sft.training_batches`、`rl_preflight.validate_schedule`、`rl_run.load_rollout_run` 及直接 `rl_update.update_actor` 共享检查。SLURP 身份（ID、schema、dataset/source_dataset 或音频路径）没有审核记录时拒绝；其他数据显式禁止优化或非 learning 角色也拒绝。非 SLURP 既有实验不强制新增审核字段。

这是数据一致性检查，不是防篡改安全边界：审核必须来自可信发布流程，哈希不能证明语义正确，也不能防止人为同时重写所有数据及哈希。当前没有独立签名批准数据库；推理和诊断评分入口也未全部加此限制。不能把写入 reviewed 字段视作完成审核。

## 验证

运行 `PYTHONPATH=src .venv/bin/pytest -q tests/test_slurp_training.py tests/test_sft.py tests/test_rl_preflight.py tests/test_rl_run.py tests/test_type_definitions.py tests/test_decision_rules.py tests/test_thinking.py`：**92 passed**。

覆盖部分审核、负例证据、跨来源/旧定义/旧 target 审核拒绝、输入或标签变更、旧 SLURP/selection/禁用样本被训练入口拒绝、合成合格样本通过显式 SFT 批次。fixture 审核不构成真实数据批准。

[真实学习集空审核预检](slurp-training-pending-dryrun-v1.json)认证并读取全部 40,713 条 learning 录音的请求和标签：对 person/date/time 三个草案任务得到 122,139 个 pending，训练输出 **0**。未读取 selection/TEST、音频波形或模型权重。三个类型只用于覆盖检查，不代表首轮训练范围已定。

## 仍需完成

1. 首条决策链的定义、正负例和声学证据审核，固定 selection 与规则族协议。
2. 不可变发布器认证 source receipt、源文件、任务和审核账本，写出 manifest 与拒绝统计；内存投影函数不等于可训练 release。
3. 学习批次、来源组权重及完整评测分母；不能只评分审核过的容易样本来夸大指标。
4. 固定运行快照包含新增模块及依赖，完成真实 SFT 与 verl/vLLM 验证；历史镜像中的代码不会自动更新。

本次单次 thinking 训练适配不等于实现有界复核或学习停止。后者仍须通过[数据—决策链实验](../research/11-decision-chain-dataset-codesign.md)检验。
