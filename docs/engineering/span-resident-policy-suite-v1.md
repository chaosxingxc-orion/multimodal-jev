# 常驻 vLLM 多策略独立验收入口

2026-11-12 UTC。新增可执行入口`run_span_resident_policy_suite.py`，接收冻结
的多策略配置，重新生成HF参考，然后在同一vLLM引擎中依次切换LoRA并输出
完整行为概率。它补充常驻rollout的实际调用路径，不是在线优化器循环。

## 执行语义

1. 固定各策略、基座、runtime profile、同一完整输入人口及源码hash。
2. 每个策略在独立HF子进程推理；检查真实子进程返回码和terminal。
3. 所有HF子进程退出后才创建一个常驻vLLM，依次处理各策略。
4. 常驻组件检查实际张量、策略与worker身份；入口再次从两侧完整概率
   独立重算候选/联合parity，不信任结果中的passed标记。
5. 全策略完成并成功关闭引擎后，才发布各策略READY和run终态。

没有HF/vLLM同时占卡假设，也没有actor共驻假设。子策略的vLLM terminal
明确表示引擎内阶段完成，不伪装成独立容器退出。外层宿主仍须检查suite
真实Docker exit0；训练还需单独认证FP32 master/Adam、导出和来源游标。
各策略产物布局兼容行为概率读取字段，但尚无原单策略`launch.json`/源码
身份适配，不能直接绕过现有`verify_gate`或作为已接受训练策略使用。

## 实际准备与验证

固定镜像无设备CPU容器实际退出0，54项测试通过：概率漂移、错误请求/策略、
重复/缺失人口、NaN、HF先退出、只创建一个引擎、HF/rollout/shutdown失败均
不发布READY，以及已有常驻组件与A/B/A独立验收测试。
[CPU证据](span-resident-policy-suite-cpu-v1-verified.json)。

准备工具从已有已验收初始策略和D12引用生成真实计划：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-resident-policy-suite-prepared-v1/`。
两参考实际容器均exit0；每策略仍是89音频、178输入。prepared容器计划保留
固定镜像、物理NPU4、原BF16 profile及完整源码/资产挂载；不变更训练因素。
当前只准备和验证，**尚未派发NPU，也未插入已冻结的训练队列**。
真实计划另在固定镜像无设备容器内完成全部策略、模型、输入和挂载依赖验证，
实际退出0，两策略各178请求。[计划预检证据](span-resident-policy-suite-plan-v1-verified.json)。

下一阶段需将动态新导出绑定到该验收路径，并在宿主实际完成与训练状态
核验后更新session的已接受策略。当前批量入口只能验收已存在导出，不能
以“有两个策略”声称已实现两个在线RL更新。

后续已补[宿主独立验收入口](span-resident-suite-host-acceptance-v1.md)，检查真实
Docker完成、重建任务人口和完整概率。该入口仍待真实常驻suite执行；尚未
将其凭证接入训练消费，不改变上文证据边界。
