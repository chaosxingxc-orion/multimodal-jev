# 全任务候选HF/vLLM验收准备

2026-11-12 UTC。已准备44个无标签真实音频请求的独立策略验收入口，尚未派发NPU。优先级保持四卡原生恢复及其独立验收在前，不因单卡暂时空闲抢跑。

旧`check_span_runtime_suite.py`将max_model_len固定1024，无法容纳已通过processor检查的最长1998-token意图请求。其矩形候选转换还假定每个Span候选数相同。未修改旧入口或运行快照，新增`check_slurp_native_span_policy.py`，固定TP1、物理卡4、max_model_len4096，先HF独立前向退出释放设备，再加载vLLM。

完整44请求来自`slurp-jev-native-inputs-cpu-v1`的4音频/11问题包；前向前重算音频hash、处理后波形、token IDs、声学特征、mask、Span位置与候选映射。同一BF16未合并LoRA及基座身份固定。采用既有时间任务SFT adapter仅作工程一致性验证，不将其解释为全60/55任务训练模型，也不声称冻结基座零训练成绩。

新增候选传输函数`unpack_candidate_logits`按每Span候选宽度独立做FP32 log_softmax：真实候选必须有限，补齐列只能是负无穷。完整支持可包括60-way和二分类混合请求。既有`compare_span_policies`保留所有候选logprob误差0.02及完整联合动作ratio [0.98,1.02]；六个Span的累计误差超阈值仍拒绝，不通过抽样或减少候选绕过。

vLLM仍使用BF16原生LoRA、BF16 head、完整无缓存prefill及已定义的HF顺序配置；验证实际worker展开token完全相同。任何一项概率不通过均保存结果并退出非0，不写READY。

准备目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-jev-native-policy-prepared-v1`。启动参数及资产hash见同名JSON。固定镜像CPU容器`9417fb1312a7baff8e29b48573e4b111d2de6db1555a2c48f183e4e328dbc5e2`退出0，40项接口/评分/候选传输测试全部通过，无NPU映射。该CPU测试没有运行新HF/vLLM NPU入口，不能据此宣布后端兼容或训推一致。

剩余：四卡优先任务收尾后核对设备占用，执行此准备包；通过后仍只准入这些精确工程请求。正式全量训练需要已审核监督、更广输入覆盖及连续verl生命周期，草案定义和4条输入不替代该要求。
