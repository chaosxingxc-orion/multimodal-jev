# 低学习率D12完整对照队列

2026-11-12 UTC。将既有`intent-d12-lr2e5-control-prepared-v1`首步及协议扩展为完整12步训练/独立验收计划，保持原始adapter、全新Adam、seed317、每步8来源、同一96次来源使用日程（89独立来源）。唯一训练超参数变化为lr1e-4→2e-5，不更改奖励、过程权重、动量、LoRA或数据。12个核心计算文件逐一与首步控制及已验收D第二步一致；训练导出权限差异的历史说明仍见`intent-lr-control-compute-source-audit-v1.json`。

完整准备根目录`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-d12-lr2e5-full-chain-prepared-v1`，配置SHA256 `98fa712cc960baa9419ea72578e82e1f0916a868cfd30e66d565412c02d4a117`；协议保持SHA256 `e2b8cacdd048f573affd1d0482db7460ddf3d0609654336e5439d6f2a7126ca2`，没有重新挑来源。全部24个训练/suite根目录在启动等待控制器后核验仍未claimed。

控制器`intent-d12-lr2e5-controller-v1`已于05:45 UTC启动，PID3563851，实际argv核验一致，首次状态`waiting_exact_QP12`，跟随Q/P控制器PID3558893。配置SHA256 `24fc8448f2ed9c5529874857d7dcb32457f1ce1f5c90ec71e77eeb711082209f`，源码清单SHA256 `c51017f17a507c872cfb4c4747fca3191b4573d0d206602768005020f2f74843`。等待期间不占NPU；每600秒核验，要求Q/P成功终态、P12实际Docker退出0和独立验收。前序失败或进程消失无终态则停止，不重复派发或重启已有训练。

首步复用已验收的初始化策略，后续每步精确恢复自身前序FP32 master/Adam，使用新验收行为概率，完成独立178输入HF/vLLM与导出验收再继续。不能从原lr1e-4的D末点接续此对照。24项LR对照、匹配控制器及续跑相关测试通过，实际12步配置与谱系校验也通过；尚无本对照训练结果。

该对照在观察原D前两步后提出，属于探索性单因素对照，不是盲法超参数研究。第三步的Adam方向诊断进一步说明较小lr只能缩小给定状态下的更新步幅，不能保证改变动量方向；本实验不预设一定有效，也不把此对照加入已冻结的FSC七策略矩阵。后续仍需独立开发评测和多个种子支持任务收益结论。
