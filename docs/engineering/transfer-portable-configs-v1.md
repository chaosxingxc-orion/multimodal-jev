# FSC、SLURP clean TEST、MINDS配置入口

三条已冻结评测路线现在提供同一宏解析器可消费的共享模板。模板只生成原driver格式的
JSON，不运行准备、评分、SSH或推理，也不改动当前队列。

| 模板 | 原执行入口 | 数据与结论范围 |
| --- | --- | --- |
| [fsc-transfer.json](../../configs/portable/evaluation/fsc-transfer.json) | tools.reports.run_fsc_arithmetic_experiment | FSC官方TEST3793录音，SLURP选择策略的配置迁移 |
| [slurp-clean-test.json](../../configs/portable/evaluation/slurp-clean-test.json) | tools.reports.run_slurp_context_test_experiment | clean TEST12991录音，冻结initial/selected比较 |
| [minds-transfer.json](../../configs/portable/evaluation/minds-transfer.json) | tools.reports.run_minds_context_experiment | 上游TRAIN8168录音迁移诊断，不称官方TEST |

先参考[本地绑定示例](../../configs/transfer-macros.example.json)填写路径、scoring contract和
真实pin映射，再执行：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/prepare-transfer-configs.json \
  --local .local/config/macros.json
```

三个步骤独立生成配置；共同的TRANSFER_SOURCE_PACKAGE绑定FSC源策略发布包供TEST和
MINDS使用。只绑定文件路径不意味着该未来包已存在，实际运行仍须完整源域选择和资产
审核。示例空pins和占位hash不能用于派发，不能通过删减pin绕过验证。

配置值等价验收使用原arithmetic-v2/v1 spec，三个解析结果逐字段相等，分别包含1073、
1081、1091项pins；原配置字节不变。经过真实本地持久盘检查，统一编排成功且重入复用。
证据见[validation.json](../../configs/releases/transfer-configs-v1/validation.json)。

本次没有重验模型/音频/预测或重新评分，不将配置等价当作跨机器可运行证明。原driver
仍限制既定remote prefix、镜像、设备和工具链；仅替换宏不能解除这些约束。需要跨主机
运行时，须先为新版本运行环境进行独立有限验收，再发布新任务。当前冻结队列不迁移。

FSC官方参考比较另有[可移植配置入口](fsc-comparison-portable-config-v1.md)。各目标域
只在源策略冻结后使用；缺失结果保持缺失，不通过目标数据挑checkpoint、模板或阈值。
