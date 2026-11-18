# Omni 专家权重转换的数值诊断

上一轮本地 checkpoint header 与 HF meta Thinker 的检查仅证明转换后的参数形状
覆盖完整。本轮实际调用固定 Transformers 的 WeightConverter，验证数值与专家编号。
脚本为 `tools/jev/probe_omni_expert_conversion.py`，不修改生产加载器或镜像。

## 输入、转换和判定

使用 128 个专家，与本地模型相同；hidden=8、intermediate=12，权重为 seed=17 的
FP32 合成张量。384 个独立 gate/up/down 源张量按逆专家编号插入，并打乱投影类型
的插入顺序，覆盖 9/10、99/100 等字符串排序容易出错的编号边界。

从实际 `get_checkpoint_conversion_mapping('qwen3_omni_moe_thinker')` 获取两条规则，
使用加载器同一 `dot_natural_key` 排序，再调用真实 rename_source_key、add_tensor、
convert。gate/up 使用 MergeModulelist(dim=0) 和 Concatenate(dim=1)，
down 使用 MergeModulelist(dim=0)。逐专家检查三个投影与原始值严格相等。
这验证转换器与明确的排序调用，不等于执行完整 from_pretrained 加载管线。

将两个打包张量严格加载到真实 Qwen3OmniMoeThinkerTextExperts，执行 128 token、
每 token 两个固定专家的 CPU 前向。所有专家均被覆盖，路由权重归一化。
独立参考计算逐 token、逐专家使用原始 gate/up/down 和 SiLU，按路由权重求和。
预设 atol=1e-6、rtol=1e-5。

两个区分性负例分别交换专家 2/10 的 gate/up 权重，以及交换 gate 与 up 两半。
每个负例必须产生大于 1e-4 的输出误差，避免输入过于退化导致错误转换也能通过。
负例仅修改探针内合成权重，不是优化器更新。

## 实际结果

固定容器退出 0；384 个源张量转为两个打包张量，128 个专家的全部投影逐值相等。
CPU 专家前向与独立参考最大误差为 0。专家顺序负例最大误差 0.0196906，
gate/up 交换负例最大误差 0.0463829，均超过预设区分阈值。
转换和模型源码哈希与上一轮固定镜像审计一致。

## 边界和后续

本轮没有读取真实 checkpoint payload，没有音频 encoder、完整 Thinker 前向、
自动设备映射、TP/EP、NPU 或梯度。不能由合成专家验证宣称真实 30B actor
加载正确或 RL 已可训练。后续实际加载必须核验缺失/额外键、专家转换及设备分布，
再用真实音频和已验收采样 token 比较 HF 与 vLLM 的概率。

容器使用已有固定镜像，2 CPU、6 GiB、无网络和 NPU；实际结果与源码哈希
见 `omni-expert-conversion-probe-v1-result.json`。主任务基线配置保持原样。
