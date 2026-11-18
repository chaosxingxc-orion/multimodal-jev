# 日常编排增量版本 v4

本版增加retention_plan，将既有封闭实验保留工具接入同一个DAG；编排不提供apply。
更新团队入口说明，原采样、收集→轨迹、报告和配置入口保持可用。

72项相关CPU测试通过；真实本地持久存储检查下，合成库存清单完成并成功重入，
所有库存文件保留，研究文件删除0。证据见[validation.json](validation.json)。

`SHA256SUMS`只覆盖本次登记的入口、模板、文档与相关测试，不是全依赖锁，
不认证NPU部署、远程训练队列、实际主实验清理或任务收益。

旧v3编排源码原hash为34d7544100ff2dd9fc13fe60639cf73880ce7b98fe3f86e4ed48a16da95ce804，
保存在`../local-operations-v3/source/tools/run_operations.py`。更早v1/v2源码也各自存档。
本次改写前的configs/README.md和tools/README.md按v1清单核验并存于v1/source相应路径。
旧清单及旧运行收据不改写，旧收据不能在新源码下强行复用。

本次未更改任何冻结训练包、当前队列或900秒监控。真实主实验的封闭库存自动生产及
完整运行/排队依赖登记尚未接入，不把这次合成验收视为已完成研究存储清理。
