# 完整采样响应的概率数值核验

实现位于 `src/multimodal_jev/omni_probability_audit.py`。这是后续真实
Omni rollout 与 verl actor 概率比较的离线工具，不加载权重、不调用推理，
不读取任务标签或奖励。当前验证使用合成记录；尚无真实 NPU 概率通过结论，
也未设定正式 NPU 数值容差。原 CPU 诊断的 0.1 不作为默认门槛。

## 输入和运行约定

`records.json` 是经独立验收的完整原始采样记录 JSON 数组，保留
`example_id / decision_id / replicate`、策略/指令哈希、结束原因、完整
`token_ids / log_probs` 以及原记录其他字段。哈希采用项目 `digest`
规范化 JSON 规则，不是原始文件字节 SHA。不能先过滤格式失败、length
截止或低奖励样本再计算哈希；原始学习分区、K 响应覆盖和采样真实性须由
采样验收及概率构批入口核验，本工具仅核验所绑定 roster 的完整覆盖。

`runs.json` 是运行列表，每项为：

```json
{
  "name": "registered-run-name",
  "responses": [
    {
      "example_id": "original-example-id",
      "decision_id": "original-decision-id",
      "replicate": 0,
      "record_sha256": "digest of the complete original record",
      "token_ids": [123, 151645],
      "log_probs": [-1.0, -0.5]
    }
  ]
}
```

以上只是字段示例，不是模型结果。每项 actor 响应须来自原始同一序列的
teacher-forcing；按有效 response mask 提取，不能包含 padding，不能删去
EOS 或提前截断。运行和响应的到达顺序可不同，身份、token、数量须精确匹配。
原始 JSON logprob 不替换为 DataProto 的 FP32 副本。

`protocol.json` 精确包含以下字段，并在 actor 执行前登记其 `digest`：

| 字段 | 含义 |
| --- | --- |
| `schema` | 固定 `omni-probability-comparison-v1` |
| `records_sha256` | 完整原始记录数组的 digest |
| `policy_sha256` / `instruction_sha256` | 与原始记录一致的策略、指令身份 |
| `run_names` | 至少两个唯一运行名，固定比较范围 |
| `max_abs_logprob_error` | actor 相对原始 rollout 的逐 token 最大绝对误差上限 |
| `token_ratio_lower` / `token_ratio_upper` | 正且有限、包含 1 的逐 token 重要性比率界限 |
| `max_between_run_logprob_error` | 任意两组 actor 运行的逐 token 最大差异上限 |

没有隐式默认阈值。工具校验协议哈希，但哈希本身不能证明登记发生在运行之前；
时间顺序必须由不可变启动记录与 source/container 证据独立证明。

```bash
PYTHONPATH=src .venv/bin/python -m multimodal_jev.omni_probability_audit \
  --records records.json \
  --runs runs.json \
  --protocol protocol.json \
  --protocol-sha256 "$PROBABILITY_PROTOCOL_SHA256" \
  --output probability-audit.json
```

退出码 0 表示数值门槛通过；1 表示完成核验但数值不通过，两种情况均写报告；
2 表示输入、算术或输出文件错误。拒绝重复 JSON 键及非有限常量。输出以
独占创建方式写入，已存在的证据文件不会被覆盖。

## 指标解释与边界

每个有效 token 计算 `actor_logprob - rollout_logprob`；其指数为该 token
的概率比。分别检查绝对误差和概率比界限，任何 token 超限均失败。全部
登记运行两两比较，不只比较一个参考批次。报告保留全部逐 token log-ratio，
逐响应失败项、最大/平均误差、token 加权均值、响应等权均值及 nearest-rank
p99；不能以平均值或 p99 掩盖最大误差失败。

整段响应的 log-ratio 是逐 token 差值之和，其指数仅为描述性指标，不作为
独立准入门槛。正负差值可能抵消，即整段比率为 1 时，逐 token 检查仍可
失败。该比率也不是完整词表 KL 或任务质量指标。指数溢出输出显式标记和
null，指数下溢输出 0 及标记，不裁剪成可接受值；无法有限表示的 log-ratio
求和直接报错。报告可用严格 JSON 序列化。

调用方仍须认证真实权重、容器、设备、音频特征、prompt 和采样输入。数字
自洽不等于 actor 与 vLLM 实现等价，也不能证明参数更新、权重导出/恢复或
跨任务泛化有效。后续真实 actor 输出将通过本入口核验，权重训练仍遵循
冻结基线 → 教师提示优化 → native/Jev 比较的当前优先级。

## verl 张量输出衔接

`omni_probability_capture.recompute` 将完整概率 DataProto 送入现有
`omni_actor_forward.compute_log_probs`，在 `torch.no_grad()` 下仅调用一次
actor，然后生成上述 `runs.json` 的一个运行对象。调用前核对原始记录哈希、
策略/指令、完整唯一身份、右填充 mask、原始 token 和旧概率 FP32 副本；
失败不调用 actor。调用后仅提取每条响应的有效 token，保留 EOS、length
截止和 actor 输出精度，不把 float64 降成 float32。padding 的 NaN 不进入
报告，有效位置的 NaN/正 logprob 拒绝导出。

```python
from multimodal_jev.omni_probability_capture import recompute

# batch 必须由 make_probability_batch 构建；records 为认证的完整原始数组。
run = recompute(actor, batch, records,
                name="registered-run-name",
                micro_batch_size=registered_micro_batch_size,
                pad_token_id=processor.tokenizer.pad_token_id)
```

完整 DataProto 可以重排，导出以 `record_identity` 对齐，禁止依赖原数组行号
猜测身份；遗漏响应的子批不能导出为完整运行。算概率时 worker 不接收
old logprob 或奖励，原概率不会被新结果替换。接口无自动重试、采样或训练。
也提供 `capture(data, log_probs, records, name=...)` 用于已获得的张量，
但此低层接口不能证明传入概率来自所声称的 forward，调用者必须独立认证。

集成测试使用真实 DataProto 与合成 actor 输出，覆盖两种音频数量、不同
响应长度、完整重排、遗漏拒绝及同长度响应概率被错误互换时的数值失败。
它验证数据衔接，不代表实际模型前向已经通过。

## 验证

`tests/test_omni_probability_audit.py` 使用合成记录覆盖：全量身份对齐、
重复/缺失/额外响应、token 截断和 padding、原记录变更、原 JSON 精度、
逐 token 正负抵消、运行间独立门槛、两种均值、尾部最大值、指数溢出/下溢、
非有限算术，以及 CLI 数值失败退出码与证据防覆盖。该模块不依赖 torch、
verl 或设备，因此本次使用项目 `.venv` 的纯 Python 测试，不启动新容器。

验证命令：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_omni_probability_audit.py`。
结果为 54 passed，0.17 秒；`git diff --check` 通过。
