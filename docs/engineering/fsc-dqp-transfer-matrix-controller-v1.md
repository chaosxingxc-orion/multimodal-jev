# FSC完整迁移矩阵执行队列

2026-11-12 06:00 UTC。后台控制器`fsc-dqp-transfer-matrix-controller-v1`已启动，PID3565864，实际argv与启动收据一致，首次状态`waiting_training_queue`，精确等待低学习率控制器PID3563851。完整资源顺序为D12→Q12→P12→低学习率D12→FSC七策略评测。等待不占NPU，每600秒核验一次，前序失败或进程消失无终态则记录错误并停止，不重新派发已有任务。

配置SHA256 `5c23928a10b554a469206579d4c07c248d141ff002b949c84ce0f703dc890e73`，源码清单SHA256 `52894e172a486dd01ee8117bf36a7405a9e9b24d1c73f1bc9bb9fd0ce18a39d5`。共享根目录`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-dqp-transfer-matrix-controller-v1`。启动收据见同名前缀`-launch.json`。

低学习率臂只决定资源先后，不进入FSC策略选择，也不根据该臂成绩改变既定七策略名单。前序完整终态、12步进度及实际末点suite通过后，调用冻结的matrix builder；builder重新认证D/Q/P实际checkpoint，再生成七套完整目标gate。每个成员首次创建前验证协议/config/source/输入pins、资源和物理卡锁，只使用NPU4 TP1。逐个执行完整HF/vLLM校验，通过后只保存后端身份与概率一致性证据，尚不读取标签。

所有七策略执行并验收后调用正式评分入口；该入口再次核验整矩阵再提取目标标签，输出31命令Final、组件、配置不变性、配对变化及说话人区间。任一成员失败即保留现场停止，不跳过、不换checkpoint、不产出部分矩阵成绩。输出在`run/comparison.json`；控制器终态0仅代表流程完成，是否有正迁移仍由指标决定。

31项队列/准备/绑定/评分相关测试通过。新增完整七成员顺序测试，验证每个launch→exit→accept后才进入下一成员，全部完成才score；模拟Q-step6后端失败时不派发后续成员、不评分。外部执行在该测试中模拟，不能当作实际七策略推理完成。当前真实矩阵根尚未创建，模型推理尚未开始。
