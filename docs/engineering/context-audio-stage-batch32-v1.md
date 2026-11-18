# 简单任务：原MINDS失败batch32逐层复现

2026-11-17 UTC。范围严格限于简单单音频原生意图判决的工程一致性，不开发或评测复杂任务。

## 为什么执行

完整效率对照已有声学批处理+KV16的吞吐收益，但MINDS259条面板存在概率差异，不能据效率提升放行正式SLURP训练。此前两个固定样本的单条和两条批次，在HF、原声学路径与批处理路径均exact，仍不足以覆盖原service32运行。

本次从原运行的真实宏批账本读取第一macro的`rollout-0001`。不重新抽样，也不根据预测挑选其中成员。按原服务规则稳定排序，保留整组32条邻居。原始输入tensor身份逐条匹配冻结缓存回执。

账本：`operations/minds-acoustic-kv-parity-v1/failure-evidence/macro-00000000-input.json`。
SHA256：`1a5cfad8e2a5bd24484c8526697eaea5595044e68fe537e59910c1cc1304ef38`。

选择封包完成后，用原冻结预测交叉确认该组32条中31条曾出现HF/macro logit差异。核对材料保存在`minds-audio-stage-batch32-v1/original-cohort-differences.json`；没有读取目标标签。该统计只是确认重放覆盖失败条件，不是任务指标。

## 实验与验收

沿用原MINDS声学KV16父包、共同Stage1、BF16、service32、KV16、原候选14及实际输入。隔离175物理7，固定镜像、共享模型/缓存只读，本地热状态。复用现有有限诊断生命周期、排他租约和准确容器句柄。

三个独立进程依次执行HF、原单录音声学worker、声学批处理worker。各路径完成同组32条单条请求；两个推理路径还执行整组32条请求，使用新的音频缓存UUID，保证实际声学计算。只读钩子保存中间tensor，独立收集后重新比较，不改权重和阈值。

比较声学encoder输入、18层encoder、ln_post、最终声学投影和最终logit/prob。记录首个差异层；若声学全部exact而概率不exact，下一步定位语言层/LoRA及概率head。若捕获期间的调度、顺序或数量与原失败不同，应明确记录不能视为已复现。

成功执行诊断不代表正式准入。必须修复后重新通过完整259条一致性面板、其他候选宽度、源训练数值和恢复验收，再启动全量训练。

## 固定入口

共享模板：`configs/portable/operations/context-audio-stage-probe.json`。
本地宏：`.local/config/minds-audio-stage-batch32-v1.json`。
操作模块：`tools.reports.context_audio_stage_operations`；源码与封包身份独立保存。已尝试包不重复launch。

新增固定汇总：

```bash
PYTHONPATH=src:. python -B -m tools.reports.summarize_context_audio_stages \
  --report /YOUR_LOCAL_OPERATIONS/RUN/COLLECTED/evaluation/report.json \
  --output /YOUR_LOCAL_OPERATIONS/RUN/stage-summary.json
```

占位路径从团队本地宏解析，不能照抄他人主机地址。汇总只消费独立比较报告，不负责替代容器和传输审核；不触发模型分析、训练重启或目标调参。

常规训练监听保持900秒；本诊断由同一个驱动等待准确容器终态，未设置逐分钟的模型分析任务。

## 后续主线

解决数值阻塞后锁定统一底座和配置，先全量clean SLURP简单任务GRPO；从共同Stage1比较源域指标、概率质量及完整曲线，再封存源选中/固定终点策略，运行FSC/MINDS跨数据集评测。效率研究每臂最多100步；正式效果按预登记完整预算。不把同类任务跨数据集结果宣称为跨任务结果。
