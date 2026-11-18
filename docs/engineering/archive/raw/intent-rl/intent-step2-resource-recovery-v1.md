# 第二步验收资源阻塞与有界训练接续

2026-11-12 UTC。第二步训练成功退出后，控制器`intent-finite-continuation-queue-v2`已冻结suite配置，但在首次docker create之前因privileged容器`355f1843843681785182ffb2acbc3dbc7f4a44b506bf6a5c018eeab408c240f3`触发设备占用检查而退出1。`intent-finite-d12-pilot-v1`随后观察前序失败，退出1，未派发第3步。失败终态见`intent-step2-resource-failure-terminals-v1.json`；原训练、模型与optimizer不重跑、不修改。

实际资源核查时原阻塞容器已被外部移除，另一个privileged容器仍存在。没有停止他人容器或绕过设备检查。新增`queue_intent_prepared_acceptance.py`跟踪已冻结、尚无create-attempt/launch/run的step2 suite；已知容器占用时每600秒等待，在资源检查通过后只首次创建验收容器。未知错误、创建/启动不确定仍退出，不自动重试。两个针对资源等待/未知错误的测试通过。

新验收控制器`intent-step2-acceptance-resource-queue-v1`，PID3547966，启动身份及存活已核验。新D12控制器`intent-finite-d12-pilot-controller-v2`，PID3548079，跟踪该确切验收队列。旧D12已确认终态失败且PID消失，第3–12步均无config/launch/create-attempt/run后才接续。新控制器沿用原计划的所有训练/suite目录、12步预算、完整日程和协议哈希`529260ed0137baedc74d4c1d5ab2e77c05eaafce13e9981aeb6112bf16bf3ac5`；只更换依赖跟踪与创建前资源等待逻辑。15项相关控制测试通过。

两个新控制器均不代表第二步验收通过；等待阶段不会持有卡启动锁。若资源状态在检查与首次创建之间发生竞争，仍会由原launch检查拒绝并保留失败，不冒险继续。原source snapshots保持不变，新控制器源码独立冻结。Q/P仍仅准备，不派发，与D连续实验争卡无关。
