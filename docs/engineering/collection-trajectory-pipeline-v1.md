# 收集到轨迹导出的固定编排

tools.run_operations的local-operations-v3支持training_collection步骤，及轨迹步骤的
collection_from引用。共享流程为configs/portable/operations/collect-trajectory-pipeline.json。
v1/v2旧流程不接受远程收集步骤；现有训练、评测和collector保持原owner，不自动迁入。

```bash
PYTHONPATH=src:. python -m tools.run_operations \
  --config configs/portable/operations/collect-trajectory-pipeline.json \
  --local .local/config/macros.json
```

在[收集宏样例](../../configs/training-collection-macros.example.json)基础上集中配置
WORKSPACE、TRAJECTORY_OUTPUT、COLLECT_TRAJECTORY_PIPELINE_OUTPUT。
输出目录各自独立，运行期间冻结宏快照。adopt模式核验已有终态归档，collect模式调用
已有准确句柄收集器；不会启动或重启训练。不要对仍由其他owner收集的任务使用collect。

轨迹步骤必须显式依赖training_collection步骤。编排先核验上游COMPLETE、来源、完整
声明产物集合及transport身份，再为下游生成专用宏快照，注入TRAJECTORY_COLLECTION、
TRAJECTORY_TRANSPORT、TRAJECTORY_TRANSPORT_SHA256。无需手工复制hash，也不修改
全局本地绑定；即使全局存在旧transport宏，下游也使用已验证上游的身份。

成功重入验证来源和产物，不重复远程收集或导出；失败保留证据并阻断下游，无自动重试。
编排hash核验改为流式读取，避免将大型tar一次载入内存。仍会多次校验大型归档，
目前优先保证来源完整性，不声称已优化大型全量归档的I/O效率。

44项联合测试通过，覆盖真实编排调用、动态绑定、重复执行、产物变动和依赖要求。
旧两步prefix实际归档在本地以adopt方式完整串接，导出的256次query与此前导出逐字节
相同；再次运行already_complete。没有新增SSH收集、NPU推理或训练。

旧local-operations-v2的编排源码已按其登记hash存档至
configs/releases/local-operations-v2/source/tools/run_operations.py；当前增量见
configs/releases/local-operations-v3/SHA256SUMS。旧锁保留历史身份，不在新源码上冒充通过。
这是收集/轨迹组合的验收，研究中间产物保留清理仍未实现。
