# 工具

日常任务使用固定工具和版本化配置，不由模型逐轮挑文件、抽样或拼接命令。
共享配置及团队启动方式见[配置入口](../configs/README.md)。

统一编排入口为`python -m tools.run_operations --config … --local …`。
已登记配置生成、全量日程、缓存清单、训练信号统计、概率错误分层、TRAIN抽样、
终态收集、轨迹导出及封闭实验保留清单；
按配置依赖顺序运行，失败停止下游，成功重入校验后复用。没有任意shell或模型调用入口。

`tools.configure`负责本地宏解析；`tools.cache_cleanup`提供`--apply`缓存删除。
`tools.retain_experiment_artifacts`提供显式库存、结论归档和依赖保护下的`--apply`
产物删除；DAG仅调用其plan模式，不触发删除。清理不扫描未登记文件、不发现未登记任务。
实际主实验封闭库存自动生产、完整依赖登记和远程训练队列迁移仍未完成；不能将入口
完成称为全工程清理完成。旧collector、900秒监听和冻结实验driver继续负责当前主任务。

保留流程及验收见[编排说明](../docs/engineering/retention-pipeline-v1.md)，当前增量锁见
[local-operations-v4](../configs/releases/local-operations-v4/README.md)。旧发布收据绑定原源码；
更新共享入口后使用新输出版本，不能强行复用旧收据。

新功能优先复用上述入口和现有包内算法，避免按step、checkpoint或数据集复制脚本。
现有冻结依赖保持不动；源码变更通过新的输出版本验收并记录身份。
