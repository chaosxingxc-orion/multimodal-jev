# 日常编排增量版本 v5

增加retention_inventory和inventory_from，将原库存发布器与原保留plan串接，继承
生成资产身份、依赖保护和同一实验协作锁。旧v1–v4配置仍可解析，不修改冻结训练包。

99项相关CPU测试通过，实际本地存储下合成完整流程及重入通过，研究文件删除0。
详见[验证记录](validation.json)及[使用说明](../../../docs/engineering/archive-retention-pipeline-v1.md)。

SHA256SUMS是本次登记文件的增量锁，不是完整传递依赖或NPU部署认证。旧v4编排及
两个入口README已按原清单hash存于v4/source相应路径；旧清单、收据和训练依赖不改写。
新源码使用新版本输出目录，不强行复用旧源码绑定的workflow。

主实验封闭生产者与完整依赖登记尚未接入；本次不执行主实验清理，不启动训练或监控。
