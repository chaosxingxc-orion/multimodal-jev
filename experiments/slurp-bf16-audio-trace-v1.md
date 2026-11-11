# BF16 音频塔分层诊断

沿用同一BF16 adapter、同一音频和已登记的113个语言RMSNorm运算顺序干预；保持原生音频塔、LoRA、attention。音频侧新增只读hook：三层conv2d、conv_out、每个encoder layer输出、ln_post、projector两层；第一层另记LayerNorm、Q/K/V、attention输出和FC1/FC2。packed QKV按HF各输出宽度拆分。原始张量只存共享实验目录，不写入输入、不作为推理替代。

实验 `slurp-vllm-spans-bf16-audio-trace-v1` 使用物理NPU4、固定镜像、900秒截止、单个docker-wait句柄。待终态后比较上轮norm-v1的logits，先验证观测不改变输出，再定位音频侧最早不一致点。不会凭trace差异直接归因LoRA，也不会因argmax相同放行RL。训练/奖励/数据标签保持不变。

终态0：新增音频hook后的候选logits与norm-v1完全相同，max logprob差仍0.101054，gate失败。全部96个可比点由原始pt独立重算。声学路径最早在conv2d1输出已有max0.0009765625、RMS5.6872e-7、约1.7361e-6比例元素不相等；conv_out RMS0.000119378；第一层attention输出RMS0.000740143；第17层输出RMS0.0129962；最终audio projection RMS0.000235308。不能比较不同幅值层的RMS直接推断其因果贡献。

此结果把下一步缩小到卷积入口身份/权重及计算复现：需要精确记录HF/vLLM实际conv2d1输入（形状、stride、padding、数值）和权重，排除前处理/布局与载入差异；再用相同输入、相同进程重复执行卷积及布局对照。当前只证明输出存在差异，尚不能断言来自NPU非确定性、卷积算法或padding。音频第一层attention差异包含上游误差，不作为attention单独不一致的证明。无新增模型下载、训练或RL更新。

后续入口trace已增加conv2d1实际输入/weight/bias及stride采集，尚未派发验证，不回写已完成音频trace快照。临时共享存储清理期间保留最新音频trace供定位使用。
