# 常驻多策略suite的宿主独立验收

2026-11-12 UTC。新增`verify_span_resident_policy_suite.py`，为已准备的
`span-resident-policy-suite-prepared-v1`提供宿主侧验收入口，无torch依赖。
它不伪造单策略launch，也不放宽原178请求的输入与候选概率gate。

## 实际检查范围

- 精确Docker ID、固定image、物理NPU4、原create argv及真实exited/0；
  容器仍运行时即使READY存在也不能验收。
- config/source manifest与launch/prepared身份一致，全部冻结源码与依赖pins有效。
- 全部登记策略按原顺序完成，每个子目录config等于原固定配置，基座/adapter
  内容身份重算，模型/profile/输入人口一致。
- 从已准入pilot重新构造D/Q完整178任务输入，分别核对HF及vLLM结果：
  request hash、处理后音频、特征摘要、Span位置、候选顺序及完整归一化概率。
- 独立重算每候选log-prob误差与最坏联合动作比；原0.02和[0.98,1.02]不放宽。
- worker实际slot、BF16 LoRA缓冲/head、profile、READY、各阶段终态及摘要
  全部相符；缺失输出pin或自报摘要与重算不符都拒绝。

输出凭证记录每策略的gate_root与policy_id、完整文件哈希及真实容器身份。
它只证明推理验收，不证明训练FP32 master/Adam、策略更新游标或任务收益。
后续训练消费仍需显式接入这一凭证，并核验对应训练状态，不能把子目录
直接当作原单策略`verify_gate`已接受的目录。

## 验证

25项CPU测试通过。包含完整双策略目录的宿主检查、运行中容器拒绝、
缺失pin、伪造摘要/profile/worker/终态；两后端一致地使用错误任务/音频
或旧策略也拒绝，多个Span各自达标但联合动作比越界同样拒绝。
文件系统和概率核验使用真实JSON/哈希/数值，Docker完成与模型身份在目录
测试中使用替身，因此不能称为真实常驻NPU验收。

另在已有真实初始策略与D12独立引擎结果上复算相同人口及数值检查。
仅为测试此验收算子，给旧输出显式补充resident slot字段；不会写回原结果，
不把它说成已有常驻引擎执行。真实多策略NPU计划仍未派发。
两端各178请求复算均通过，最大log-prob误差9.5367431640625e-7；
[真实历史输出回归凭证](../../../span-resident-host-reference-regression-v1-verified.json)。

调用入口要求root、pilot、manifest、image及新的output路径；已存在的验收
文件拒绝覆盖。无新增模型下载、无训练目标或测试集合变化。
