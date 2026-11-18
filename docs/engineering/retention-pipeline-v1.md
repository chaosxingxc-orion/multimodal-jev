# 保留清单接入统一编排

`local-operations-v4`新增`retention_plan`，复用已有产物保留工具。它只生成清单，
没有删除或训练派发分支；v3收集→轨迹、v2报告依赖绑定继续可用。

先依据[保留协议](artifact-retention-v1.md)发布封闭实验库存，登记所有运行/排队任务依赖，
将结论以相同hash归档到实验目录之外。使用
[宏示例](../../configs/retention-pipeline-macros.example.json)填写本地绑定，然后执行：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/retention-pipeline.json \
  --local .local/config/macros.json
```

workflow校验子发布schema、apply=false、deleted_paths为空、deleted_files=0，以及唯一
输出plan.json的准确路径与hash。它继承完整来源绑定，并额外绑定每个待清理文件的hash，
因此候选文件变化或消失后，旧workflow拒绝复用。该清单不是未来删除的永久许可；
独立apply会重新核验库存、归档、保护项和文件身份，删除前按原工具再次检查。

需要执行已授权清理时，为RETENTION_OUTPUT设置新的独立版本，保持同一个实验协作锁，
使用原入口：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.retain_experiment_artifacts \
  --config configs/portable/operations/artifact-retention.json \
  --local .local/config/macros.json --apply
```

该协议依赖真实的closed/evaluation_complete声明及完整依赖登记，无法发现未登记任务。
当前主实验仍在完整selection评测中，不生成其封闭声明，不删除其checkpoint。
实际主实验库存生产尚未接入自动终态链。本次只验收合成封闭实验的plan编排。

验收包括正常完成及重入、候选/归档/输出变化拒绝、拒绝apply发布和额外输出、v4门槛，
同时回归原收集/轨迹/采样/缓存/保留路径。结果及真实本地存储验收位置登记在
[版本记录](../../configs/releases/local-operations-v4/validation.json)。
