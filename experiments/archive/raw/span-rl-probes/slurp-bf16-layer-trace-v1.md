# BF16 跨后端分层定位

前置证据：BF16 LoRA SFT20次更新及48条保存重载完成，HF重载最大候选logprob差0；同checkpoint vLLM单条最大差0.140574，一项argmax翻转。worker确认LoRA缓冲及head投影BF16，仍未通过概率gate。不执行RL更新。

本探针固定同一音频、BF16 adapter、特征、问题和位置，首先重新运行HF并要求结果与保存reference在1e-5内一致，然后在原生vLLM路径记录以下只读张量：音频投影、28层input_layernorm和post_attention_layernorm输出、第一层Q/K/V投影和最终norm。保持完整token序列；仅移除HF的singleton batch维。vLLM融合残差返回tuple时只读取其归一化输出，不把独立residual与HF层输出直接比较。packed QKV按HF实测输出宽度拆分比较。

hook不修改输入或返回值，不注入HF中间特征，不生成token。adapter请求使用独立multimodal UUID，确保先前base请求的音频缓存不会遮蔽音频塔trace。原始张量只保存在共享实验目录，不提交仓库。报告shape/dtype/max/mean/RMS/不相等比例；这些用于定位，不取代候选概率gate。以最早可比点的差异确定下一步，不预先假定全部偏差来自LoRA或音频。

运行：`slurp-vllm-spans-bf16-trace-v1`，固定镜像、物理NPU4，TP1/eager、无chunked/prefix cache、900秒硬截止。启动凭证见 `docs/engineering/slurp-vllm-spans-bf16-trace-v1-launch.json`。单个docker-wait句柄等终态；不增加高频状态轮询。完成后先比较无hook历史输出，判定观测是否改变结果，再解释分层差异。

主机测试覆盖：hook复制防止后续原位操作污染记录、残差tuple语义、packed QKV正确分段以及只改变K时能够准确定位到K。该测试不代表NPU分层路径已通过。

终态0：hook输出logits与无hook的BF16探针完全相同。61个可比点（音频投影、56个语言norm、Q/K/V和final norm）已记录。音频投影max0.001953125、RMS0.000235308；第一层input norm max0.03125，最终norm max5.375/RMS0.161725。全候选概率gate仍失败。独立从pt张量重算全部统计（浮点reduction用rtol1e-6/atol1e-8比较）并保存hash。

关键切片：第一层input norm中，音频之前104个text token的max0.03125、RMS0.000611011、约25.6%元素不相等；这些位置尚未进入本层attention，不能由当前音频条件引起。因此存在语言输入/归一化独立差异，而非仅音频误差传播。尚未比较原始embedding/weight，不能单凭这点断言norm是唯一根因。

后续独立干预 `slurp-vllm-spans-bf16-norm-v1`：额外记录第一层norm之前embedding与norm weight；将语言所有RMSNorm（包括Q/K norm）替换为HF显式运算顺序，BF16执行/残差相加、FP32方差计算、归一化后先舍入BF16再乘weight。保持原生音频塔/attention/LoRA/LM head，不替换特征、不改概率门槛。此为机制诊断，只有实际概率gate和覆盖通过后才能讨论作为正式对齐方案。

norm-v1终态0：实际替换113个语言RMSNorm；第一层norm权重完全相同，568个文本token的pre-norm embedding及干预后input norm与HF均逐位相等（max=0）。对比未干预文本norm的max0.03125，这确认了归一化运算实现是一个独立差异来源。文本第一层Q/K/V仍分别有max0.00048828125/0.001953125/0.001953125，不能视为完全相同；K无LoRA，因此不把全部投影残差归因于LoRA。

候选logprob最大误差由0.140574降至0.101054，argmax改变从1项降至0项；完整动作比率[0.844176,1.153852]仍不通过[0.98,1.02]。音频投影max仍0.001953125、RMS0.000235308；final norm RMS从0.161725降至0.151799。单条机制对照，不代表全局收益，不准入RL，不改变默认服务路径。两次任务均已结束，当前没有本轮残留NPU作业。

下一步按剩余证据推进：在当前已对齐的文本norm输入上核对packed QKV/分支运算；在音频塔增加conv及逐层输出定位，区分卷积padding/矩阵打包/attention实现。隔离attention本身需同一Q/K/V输入的诊断，并保留与真实端到端路径的区别。只有全部候选gate通过后再验证多样本与版本同步，最后接入verl更新。不能以argmax一致替代PPO行为概率一致。

本轮代码测试64项通过；分层统计与完整动作gate从本地原始张量独立重算。证据见 `docs/engineering/slurp-bf16-layer-trace-v1-verified.json` 和 `docs/engineering/slurp-bf16-norm-v1-verified.json`。原始trace仍仅在共享实验目录，未提交仓库。
