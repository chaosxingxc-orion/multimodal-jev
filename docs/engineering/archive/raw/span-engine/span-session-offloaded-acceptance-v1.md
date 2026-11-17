# 常驻actor的卸载验收协同边界

2026-11-12 UTC。`SpanTrainingSession.accept_offloaded`串接所有rank的卸载、独立验收、
恢复与策略接受。每个阶段都进行collective状态检查；验收期间会话不允许更新或导出
accepted cursor。任何rank发生阶段异常，所有能参与collective的rank都进入failed，
保留旧的已接受policy身份，不自动重试。最终仍使用完整验收回执摘要共识，不能只
凭相同policy名和步数前进。

独立验收回调仍负责实际子进程退出、全候选概率验证和设备所有权；此接口不发放
设备租约，也不绕过当前资源检查。一个进程硬退出或卡在FSDP内部collective时，
仍需后端超时和外部监督器处理，Python层的异常共识不能保证恢复整个进程组。

本地会话测试16项通过。固定镜像下实际CPU容器
`61760d58db1f4c55eccea636e9a172f51e9c3e3928eb675d0490ea12c6a082f8`
退出0，无NPU device挂载。四个真实Gloo rank各自运行原生verl actor8步，比较
常驻、step4恢复、每步卸载验收三条分支。各步参数/Adam/梯度与常驻路径完全一致，
对独立全局加权参考的最大误差2.9802322387695312e-8。保留混合候选、不同请求权重
与完全缺标rank。人为注入rank2恢复失败，四个rank均拒绝新策略；不同rank回执
证据不一致的反例同样通过。

该CPU探针的模型移动回调为no-op，状态检查与协同是真实执行，但不是NPU/FSDP
搬运、BF16或音频/vLLM常驻证据。已排队的真实FSDP卸载探针不修改；真实语音常驻
训练仍需在其通过后实现资源交接及独立验收，不能把本轮称为完整verl/Ray trainer。
机器证据见`span-finite-session-offload-dp4-cpu-v1-verified.json`；冻结源manifest、
实际镜像、Docker退出及四rank逐步结果均已重新核验。
