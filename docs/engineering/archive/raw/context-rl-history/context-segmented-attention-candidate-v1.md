# Rollout真正批量前向：分段attention组件候选

当前正式入口仍为[冻结基线](../../../context-prob-v1-baseline.md)。本候选未接正式或排队运行，
没有新增设备任务，不改变任务、动作、reward、loss、batch或数据日程。

## 已确认的限制

现有bulk RPC减少IPC次数，但逐条score仍逐条模型前向。当前
`context_resident_rollout.py`固定`max_num_seqs=1`，并逐条`encode`。
不能仅提高该值或向encode传列表：

1. `span_batch_rollout_candidate.py::packed_attention_forward`要求positions恰为
   `arange(total_tokens)`，语言attention只接受单条完整prefill。
2. `span_audio_padding.py`明确要求`feature_lens.numel()==1`；多录音会被拒绝。
   短音频必须保留逐录音完整CNN窗口边界，不能直接拼接后卷积。
3. 固定tile投影能接平坦张量，但多请求拼接会改变tile位置；native LoRA的批量布局也变。
   现有单请求NPU训推证据不能推出批量数值一致性。
4. `ContextProbabilityPooler`与`ContextHistoryReadout`已经按完整请求长度拆分hidden，
   逐请求核对policy、完整token和marker，然后调用head；可复用此隔离契约。

## 本轮完成的候选

`tools/reports/context_segmented_attention_candidate.py`直接处理平坦T,H,D，使用显式
请求长度及累计边界调用一次现有varlen attention kernel，不创建B×最大长度padding。
保留QKV、q/k norm、rotary及输出投影顺序，支持GQA。所有位置必须逐请求从0连续增长。
显式长度必须最终绑定真实scheduler metadata；仅看到位置重置不能证明完整prefill。
候选没有installer，不在worker上放置跨请求共享的可变长度状态，不支持cache/chunk/decode。

本地执行：

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:tests:. python -m pytest -q \
  tests/test_context_segmented_attention_candidate.py \
  tests/test_span_batch_rollout_candidate.py tests/test_context_pooling.py
```

结果26 passed（2.66秒）。新增13项涵盖：独立dense oracle输出及梯度、单kernel调用、
跨请求隔离、未来token因果屏蔽、请求换序、长度/位置错误在投影前拒绝、CPU默认kernel拒绝、
错误kernel输出拒绝，以及候选projection flow对比原逐请求路径。
CPU浮点张量和替身projection/rotary，不是实际Qwen权重、BF16 NPU、完整音频模型或吞吐验收。
未声称已经减少正式模型前向次数或实现加速。

同轮补齐`tools/reports/context_segmented_audio_candidate.py`：接收明确的
`[mel,sum(frames)]`布局，完整校验批次后逐录音调用原单录音声学forward，保留CNN补齐边界，
按aftercnn长度校验并拼接音频token。声学计算仍串行，目标是供语言backbone合批。
新增10项和原短音频9项共19 passed（2.56秒）：小型真实三层卷积的短录音值/梯度与原路径
逐元素一致，邻居扰动/换序隔离，以及坏长度/非有限输入拒绝。长短混合只用替身验证委托
和拼接，未执行Qwen长音频模型。没有安装到任何worker。

## 下一步的具体边界

下一候选将两个组件接入真实scheduler长度和请求身份；然后在独立
profile上用相同真实输入对比串行/批量概率、请求顺序/邻居不变性及峰值内存、端到端耗时。
head可先保留逐请求执行，以便优先批量化语言backbone。只有完整实际NPU证据通过后才考虑
后续实验使用，不打断当前GRPO/RLOO与下游评测队列。

## 同轮原任务观察

- TRAIN processor：原edf4509f容器running/nonOOM，17920/40513；本地观察
  1794716403.765828，下次普通观察不早于1794717003.765828，live-v15。
- GRPO：原e564df04容器running/nonOOM，四rank均227/634；64/128/192已提交。
  本地观察1794716546.2381775，下次不早于1794717146.2381775，live-v37。
  首次调用观察器早于门槛约8秒，被本地时间保护拒绝，无SSH；到期后才观察同一容器。
- 正式1078、TRAIN1073、finalizer1167依赖hash重新核验无变化。原终态收集/后续队列不重建。

上一轮重新核验了原driver实际ps存活，属于已核验等待；本轮新增可测试组件和两个原容器
进展证据。学术完整RL曲线、TEST和跨任务泛化仍未完成，目标保持进行中。
