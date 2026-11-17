# 候选绑定数据的训练输入适配前置

2026-11-12 UTC。新增`intent_binding_inputs.py`，未改原intent actor入口、冻结
D/Q/P源码或已排队实验。它提供三个独立环节：

1. 固定manifest、完整三臂文件和编译实现身份，重新从已准入源数据编译每条视图，
   比较请求与外部监督；Q/P输入必须相同。原发布仍标candidate_only_not_actor_admitted。
2. 先按原来源哈希日程选择8条录音，再展开每条四个完整有序视图，形成32请求。
   每视图保留原来源权重的1/4；报告source IDs、view IDs和gate IDs，不能将32视图
   冒充32个独立来源。原12步日程对应96次来源使用、384次视图使用。
3. 在调用方真实processor编码与`bind_span_request`之后，要求HF/vLLM证据的
   gate ID、request ID、完整输入身份和策略身份精确匹配；重算身份摘要，检查全部
   Span候选支持，再复用现有全候选/联合概率比验收。不能把native gate传给重排视图。

实际三个arm、各356条视图已重新加载核对，12步来源日程保持匹配；见
`intent-binding-loader-source-audit-v1.json`。本地首次测试因无torch无法收集，
随后在固定镜像CPU容器
`fb456b02c301fb45efa663659501ae1e72005f55ac9c581b4ab3ab0866ce78a8`
执行24项输入/映射/调度测试，实际退出0，无NPU挂载；JUnit为24通过、0失败、0跳过。
源码、镜像、Docker退出和JUnit哈希已认证，见同名verified JSON。

边界：这是可复用适配函数，不是完整actor准入。调用方仍必须检查实际gate退出、
完整712输入人口和导出身份，再将选中请求编码并转为原生actor张量。原训练后的
验收器仍固定178原生输入，原resume也要求178；不能只替换数据路径便声称可以
训练。后续需专用版本的训练/恢复/更新后全视图验收入口以及正式预算协议。
目前没有派发绑定增强训练，也没有将其加入FSC模型选择或测试反馈回路。

后续进展：真实音频编码与训练张量入口、D/Q/P各32视图的四Gloo-rank传递已完成
CPU验证，见[重绑定张量传递](intent-binding-tensor-transport-v1.md)。该探针使用
合成策略概率，未升级原候选发布或授予真实训练准入；完整策略gate与恢复入口
仍待完成。
