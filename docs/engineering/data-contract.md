# 数据与奖励契约 v1

`JevExample` 将 audio/schema 与 target/metadata 分开；模型输入使用
`model_input()`，不传入参考转写或原始标签。Choice 使用稳定候选身份；
Noul 是布尔值，Score 只接受明确有序等级，open_slot 保留开放字符串。
`null` 表示未知，不能和 FSC 原始字符串 `none` 混淆。

构造 FSC：

```bash
.venv/bin/python tools/jev/build_fsc_manifest.py \
  --root /mnt/cephfs/xingchao/datasets/fsc/extracted/fluent_speech_commands_dataset \
  --output-root /mnt/cephfs/xingchao/datasets/jev/fsc \
  --config configs/tasks/fsc-v1.json
```

内容地址目录已存在时拒绝覆盖。每份输出固定源 CSV、构造器、配置、音频和
清单哈希；同录音派生的 6 个决策都在同一官方 split。候选表仅从 train 获取。
参见 [实际构造记录](fsc-jev-v1-receipt.json)。

`score_rollout()` 校验类型、语义正确、错误拒识和置信度 Brier 项。
`compute_score(data_source, solution_str, ground_truth, extra_info=None)` 是
verl 自定义奖励接口。ground_truth 为完整 example 的 JSON，必须放奖励列，
不能进入 prompt。模型输出为 `{"value": ..., "confidence": 0.8}`，confidence
可省略；无效 JSON、额外执行声明、NaN 等返回失败奖励。

当前奖励只针对单决策。成对一致性、任务执行、延迟成本需要真实分组或环境观测，
尚未实现，不接受模型自报结果作为奖励。

离线预测评估：

```bash
.venv/bin/python -m multimodal_jev.evaluate \
  --manifest /absolute/path/valid.jsonl \
  --predictions /absolute/path/predictions.jsonl \
  --output /absolute/path/metrics.json
```

每条预测至少包含 `example_id` 与 `value`。缺失预测计入失败分母，不能被
当成正确拒识；重复或多余预测报错。逐变体报告 accuracy、coverage/risk、
unknown precision/recall、Brier 及成对正确率。总体派生样本准确率只作描述统计。

## SLURP transfer v1

构造器 `tools/jev/build_slurp_manifest.py` 使用
`configs/tasks/slurp-transfer-v1.json`。calendar/email/transport 域的 train
录音全部排除，官方 valid/test 再分别拆为 seen-domain 和 heldout-domain。
保留 native intent/action/entities，构造 scenario Choice 及 date/time/person/place_name
开放槽位。所有视图共享原录音划分；同一句话的多录音保留 utterance_group。

[协议](../../experiments/archive/raw/frozen-baselines/slurp-transfer-v1.md) 与
[发布记录](slurp-transfer-v1-receipt.json) 包含原始哈希、排除量和说话人重叠。
独立验证器 `tools/jev/verify_slurp_release.py` 对写出的所有记录重新检查类型、
划分、域留出、ID 唯一性和派生视图数。它不声称已完成模型评估或官方 SLU scorer。
