# FSC学术参考比较的可移植配置

原比较器`tools.reports.compare_fsc_context_reference`及正在等待的冻结队列保持不变。
新增宏模板将机器路径和来源身份集中到本地绑定，继续使用已有`tools.run_operations`
的materialize能力。没有新增评分实现或模型执行分支。

共享模板为[fsc-reference-comparison.json](../../configs/portable/operations/fsc-reference-comparison.json)，
团队绑定示例为[fsc-comparison-macros.example.json](../../configs/fsc-comparison-macros.example.json)。
填好路径与已验证pin映射后执行：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/prepare-fsc-comparison.json \
  --local .local/config/macros.json
```

输出为本地FSC_COMPARISON_CONFIG_OUTPUT下的comparison.json及provenance。示例中的
空pin映射仅表示字段类型，不可直接用于实际比较；不得删减来源证明以适配不同机器。
新团队需为本机资产和固定源码建立真实身份绑定。生成成功只证明配置解析及持久化成功，
实际比较仍由原比较器核验完整迁移终态、运行证明、参考模型、音频和预测。

本次用当前arithmetic-v2已冻结spec导出独立验收宏，解析后JSON值与原配置完全相等，
包含1102项比较pins和4项参考pins；源配置字节未改，真实本地存储检查与重入通过。
未重验这些pin指向的模型/预测内容，未运行评分或推理，也未替换队列消费的配置。
验收记录位于[validation.json](../../configs/releases/fsc-comparison-config-v1/validation.json)。

比较口径沿用[原协议](fsc-external-reference-comparison-v1.md)：FSC监督训练的SpeechBrain
官方权重，与SLURP训练后冻结模型在同3793条音频上的native top1作能力定位。
目标数据不用于checkpoint选择，不称同预算算法胜负；无外部概率输出时不计算其NLL。
SLURP HuBERT资产缺口仍未解决，本配置迁移不替代学术基线复测。
