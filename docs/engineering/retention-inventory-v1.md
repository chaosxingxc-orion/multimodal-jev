# 从实验收尾台账发布保留库存

`tools.publish_retention_inventory`读取固定生产者台账，核验登记任务的终态及审核身份，
复制结论到实验目录外，生成原保留工具可消费的inventory。无目录扫描、模型调用、
远程操作或删除。共享入口是[宏模板](../../configs/portable/operations/retention-inventory.json)，
本机信息集中在[本地宏示例](../../configs/retention-inventory-macros.example.json)对应的绑定中。

```bash
PYTHONPATH=src:tools/jev:. python -m tools.publish_retention_inventory \
  --config configs/portable/operations/retention-inventory.json \
  --local .local/config/macros.json
```

生产者台账schema为`experiment-closure-ledger-v1`，字段如下：

| 字段 | 契约 |
| --- | --- |
| root | 与配置相同的实验绝对根路径 |
| closed / dependencies_registered | 都必须为true，由完整实验收尾生产者负责真实声明 |
| terminal / terminal_sha256 | 实验根内相对路径及hash，JSON包含整数exit_code=0 |
| required_jobs | 所有必需训练/评测任务的非空、无重复ID列表 |
| jobs | ID→`{receipt, sha256, container_id}`，与required_jobs集合完全相同 |
| required_audits | 所有必需审核的非空、无重复ID列表 |
| audits | ID→`{path, sha256}`，与required_audits集合完全相同 |
| files | `{path, sha256, role, step}`列表，沿用保留协议，必须登记final/config/conclusion |
| selected_checkpoint_steps | 必须保留的选中checkpoint |
| pin_manifests | 依赖清单绝对路径→SHA256映射，传给保留工具继续保护 |
| protected_paths | 其他保护路径列表 |

job收据须含匹配的container_id，以及state.Status=exited、整数ExitCode=0、OOMKilled=false；
不同job不能重用同一container_id。审核JSON须passed=true且hash匹配。该步骤检查既有
审核证明，不重新运行学术指标或逐张量审核。生产者必须将正确任务和审核绑定进台账，
登记范围必须包括仍会消费模型的TEST/transfer等后继任务。

所有参与方使用同一个RETENTION_LOCK。工具在此锁及发布目录锁内验证来源与归档，
只有COMPLETE.json完成才是有效发布。失败目录即使已有部分归档/JSON，也不能作为
可用库存；失败不自动重试，需保留现场并使用新版本输出。

成功输出：

- `inventory.json`：原`closed-experiment-inventory-v1`协议。
- `conclusions/*.archive`：结论原始字节，流式复制、fsync及SHA256核验，不重新总结内容。
- `retention-inputs.json`：root、inventory/hash、pin清单、保护路径及协作锁绑定，供后续
  配置工具使用；不改全局宏，不包含删除命令。
- `COMPLETE.json`：输入、输出hash、候选数和保留step摘要。

库存发布前复用原retention.prepare检查文件角色、step、归档和保护逻辑。后续清理还需
自行设置输出目录及max_files/max_bytes限制；这里计算候选数不等于授权无限量删除。
每次成功重入核验所有原输入与输出。实际清理后原候选可能已不存在，此时不能复用旧
库存生产回执，应以对应清理回执追踪，而不是重新发布伪造的原始库存。

边界：该工具不发现遗漏任务、不验证结论科学正确、不自动编写台账。目前主实验driver
没有发布上述台账，也未迁移到统一协作锁，因此**不能接到当前GRPO收尾链执行清理**。
完整selection结束也不代表跨任务研究已封闭。下一步是在新版本生产者中登记完整任务
与依赖，再接库存→保留编排；旧冻结driver保持不变。

验收：43项库存、保留、保留编排、缓存测试通过，覆盖未完成任务、失败审核、缺失任务、
依赖保护、源变化、路径越界、锁冲突、归档篡改及重复调用。真实本地存储检查下的合成
实验完成归档→库存→原保留plan，7文件全部保留，2候选，保留step2和3，研究文件删除0。
证据见[版本记录](../../configs/releases/retention-inventory-v1/validation.json)。
