# vLLM pooling 原生未合并 LoRA 诊断

启动前固定：使用 balanced-v2 的非零 rank4/alpha8 q/v adapter，同一本地基座；adapter safetensors 实际为 FP32（112 tensors），vLLM 显式 lora_dtype=float32，基座 BF16。物理卡4、TP1 eager、无 chunked/prefix cache、无生成、无参数更新。

取该训练记录第一条录音及完整输入定义，候选概率参照同一 adapter 的 HF after.json。vLLM 顺序执行 base → adapter → base：验证 adapter 对输出有非零影响，卸载请求后恢复 base（logits 最大差≤1e-5）；HF/vLLM adapter 候选 logprob 最大差≤0.02 单独记录，未达标不准入 RL。此为单条接口诊断，不是更新热同步、多卡或完整概率 gate 验收。

adapter/config、音频和 HF reference 文件全部哈希锁定，保留 checkpoint 和 source 快照。当前基座 pooling 差 0.0633 已失败，不能期望只开 LoRA 就修复；本探针回答 native LoRA 加载/映射/请求隔离能否运行。

v1 启动退出1：vLLM LoRAConfig 字符串枚举不接受 'float32'，只接受 auto/float16/bfloat16 或 torch.dtype。修复为 API 接受的 torch.float32 类型，保留 FP32 适配器目标，不自动降精度；v2 重新验证真实 backend 是否支持。
