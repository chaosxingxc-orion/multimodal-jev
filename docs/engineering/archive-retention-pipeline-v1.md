# 结论归档与保留清单的配置化编排

统一入口`tools.run_operations`的`local-operations-v5`支持`retention_inventory`和
`retention_plan.inventory_from`。生产者先提供符合[库存协议](retention-inventory-v1.md)
的封闭台账；编排负责完成结论归档、库存发布与保留计划，不调用模型，不删除文件。

使用[本地宏示例](../../configs/archive-retention-macros.example.json)填写资产身份与路径：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/archive-retention-pipeline.json \
  --local .local/config/macros.json
```

两个步骤的关系是：

1. inventory核验生产者登记的任务终态、审核、依赖和文件hash，复制结论并发布库存。
2. retention必须直接依赖inventory。编排先校验上游COMPLETE及全部声明输出，随后将
   root、inventory/hash、pin_manifests、protected_paths和lock_path写入独立的子步骤宏快照。
3. retention调用原工具的plan模式，继续执行其原有max_files/max_bytes门槛。
   上游核验或归档失败则停止下游，失败目录不自动重试。

全局本地宏不修改。上游生成的绑定覆盖本地陈旧的相应值；RETENTION_OUTPUT仍由本地
配置指定。上游输出集合严格限定为inventory、retention-inputs和登记结论归档，拒绝缺项、
额外输出或越界路径。发布后重入核验来源、生成文件和候选身份。

来源和输出hash用于检测已登记内容的变化，不提供对恶意同时改写全部证明文件的安全保证。
封闭及依赖完整性仍由实验生产者负责。工具不发现遗漏任务，也不把passed字段替代独立
训练/评测审核。它只消费已登记审核证明，复用原保留规划逻辑检查可消费性。

当前主实验没有发布该封闭台账，尚有selection及后继评测，未迁移冻结driver。此流程
不接管现有任务、不生成主实验closed声明、不删除其checkpoint。后续接入需要在一个
新版本生产者中完成全任务及依赖登记，而不能只把训练退出状态翻译为实验封闭。

99项联合测试通过；真实本地持久盘验收采用合成封闭实验，完成两步及重复调用，
7个登记文件保留，生成2个候选，保持选中step2/最新step3，全局宏不变，研究文件删除0。
证据及范围见[版本验收](../../configs/releases/local-operations-v5/validation.json)。
