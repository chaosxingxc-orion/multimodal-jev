# 常驻行为概率消费自动队列

2026-11-12 UTC。新增无NPU设备的CPU消费队列，等待真实resident suite完整验收，
然后绑定initial策略到原Q第一步训练配置，调用实际音频/processor/actor prepare
对照探针。固定12个来源日程覆盖89条来源、96次使用，不训练、不加载语音模型。
这检验初始策略消费；仍不证明动态导出后连续actor更新闭环。

宿主先重新认证resident实际容器退出、utility前驱与完整概率证据，再生成
显式绑定配置。消费容器以只读方式挂载共享数据，并按固定哈希复制挂载三份
/home外部凭证。新增未知外部依赖即拒绝，不静默跳过pins。CPU命令无device或
privileged，线程数固定2，可与后续NPU评测独立运行，不改变FSC工作量。
创建只尝试一次，部分/模糊状态保留；实际容器退出0且完整89来源报告通过后
才记录消费成功。不存在resident成功凭证时不能用模拟数据派发。

19项本地及固定镜像测试通过，包括失败前驱禁止绑定/派发、已有创建记录禁止
重试、无设备命令、不同路径相同验收代码及状态谱系认证。固定CPU容器实际
退出0，测试使用替身；与未来真实消费结果分开。

实际控制器intent-resident-behavior-consumer-controller-v1，PID3688013，完整argv
确认存活，等待span-resident-policy-suite-controller-v2。冻结配置与源码在共享
同名目录。它没有派发消费容器或NPU任务。

12:30资源定位补充：精确捕获Q7阻塞容器0d846415…，它运行与已审计完成监听器
完全相同的Python脚本SHA256，镜像和host配置亦匹配，但监听目录是
`/var/tmp/s1-bl-candidate-ops-media-v1/runtime`，不在当前允许的omni-local-state
目录范围内。因此旧规则仍保守等待，不能将之前归档脚本豁免当作已解决所有
资源阻塞。没有停止该容器，也未更改冻结控制器；下一步应对该确切路径进行
单独审计并迁移等待队列。Q7 suite此时仍未创建。

历史真实音频控制容器96372bc8…最近确认运行在12:25；等待通知超时未触发
重启，结果仍待实际完成。共享梯度CPU容器12:30实际running，Q/P控制器实际
存活，已接受Q6，仍无新的学习成绩。

[执行证据](intent-resident-behavior-consumer-queue-v1-evidence.json)。
