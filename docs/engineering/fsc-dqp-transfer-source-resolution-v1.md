# FSC迁移检查点身份绑定

新增`tools/jev/resolve_fsc_transfer_sources.py`，将已冻结的七策略选择规则绑定到实际通过验收的源域adapter。工具不读取FSC音频、标签或预测，不加载模型，也不派发任务。

必须先验证精确D、Q/P控制器配置和启动回执，两者成功终态及完整12步进度；再逐一核验每个源域gate实际Docker退出0、完整178输入概率、训练arm/update、优化器恢复/导出验收、同一base/runtime，以及磁盘adapter内容。输出供正式评分入口使用的`checkpoints`和完整来源证据；重复的adapter内容允许保留，以呈现无效更新，不能替换成表现更好的点。

19项绑定和评分入口测试通过，包括完整七策略模拟绑定、权重内容变更拒绝、未完成源域不解析检查点、错臂/错步/错顺序/重复gate/目标反馈拒绝。外部Docker和数值gate在模拟成功测试中替身化；没有声称已绑定真实未来检查点。

待执行配置在`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-dqp-transfer-source-resolution-v1/config.json`，SHA256 `b7cda79f335a72a8267f849e830b858b4ef86818849d1b71066aea075d7735f4`。固定现有D控制器v3与Q/P控制器v1；两者尚未完成，配置仅准备，不执行解析，也不写虚假成功回执。实际选择回执生成后，再固定目标评测协议和所有目标gate中的协议哈希。
