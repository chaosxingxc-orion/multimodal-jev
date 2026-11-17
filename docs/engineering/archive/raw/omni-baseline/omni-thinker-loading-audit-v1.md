# 本地 Omni Thinker 权重加载与内存核查

本次读取本地 15 个 safetensors 的头部和固定镜像内真实 Transformers meta 模型，
不读取权重 payload、不加载新模型、不占用 NPU。模型 revision 仍为
`26291f793822fb6be9555850f06dfe95f2d7e695`；配置和 index 哈希与已验收资产一致。
源码工具：`tools/jev/audit_omni_thinker_loading.py`。

## 实际权重规模

| 分支 | 参数量 | checkpoint GiB |
| --- | ---: | ---: |
| Thinker audio_tower | 647,927,168 | 1.207 |
| Thinker model | 30,221,219,840 | 56.291 |
| Thinker visual | 538,631,408 | 1.003 |
| Thinker lm_head | 311,427,072 | 0.580 |
| Talker | 3,324,596,480 | 6.193 |
| code2wav | 216,016,577 | 0.402 |

Thinker 共 31,719,205,488 个参数，约 59.082 GiB。构造完整 Thinker 仍包含 visual，
即使任务只用音频；不能在内存预算中无依据扣除。首个真实 actor 前向应直接加载
Thinker，不实例化语音输出分支。权重大小是下限，不含转换临时张量、激活、
完整词表 logits、运行时 workspace 或通信开销。

vocab=152064，单条 4096 token 的 BF16 完整 logits 约 1.160 GiB；
FP32 约 2.320 GiB。现有 HF thinker 和 verl padded 路径先生成完整 logits，
不能沿用旧 ASR 的 response-only projection 内存假设。

## 参数布局差异

checkpoint Thinker 有 19,743 个张量，当前 HF meta Thinker 有 1,407 个参数张量。
直接去除 thinker 前缀后比较，出现 96 个缺失键、18,432 个额外键，
双方共有键的形状全部匹配。差异完全落在 48 层、每层 128 个 MoE 专家：

- checkpoint 每个专家独立保存 gate_proj、up_proj、down_proj。
- HF 每层保存 experts.gate_up_proj `[128,1536,2048]`，
  experts.down_proj `[128,2048,768]`。
- 固定版本 `conversion_mapping.py` 将 qwen3_omni_moe_thinker 映射到 qwen2_moe
  转换规则：先沿专家维 stack，再将 gate/up 沿 dim=1 concatenate；down 只 stack。

因此不能直接用未转换 state_dict 并忽略 missing/unexpected keys。
v2 审计用所有本地 header 重现上述形状变换，与实际 meta 参数集合完整比较；
这仍不证明 from_pretrained 已运行、专家值/顺序正确、分片设备加载或 vLLM 导出成立。
转换源码与逐轮记录保存在共享实验目录和验收 JSON 中。

## 下一步的真实前向边界

需要在已有采样队列完成后，绑定其原始 token、音频和 policy 身份，加载同一
checkpoint 的 Thinker，显式检查转换后的加载报告，先做冻结权重的单／双音频
teacher-forcing。保持 padded、SP=1、temperature=1，不启用未经验证的 fused
输出或 remove-padding。记录实际设备分布、峰值内存和逐 token 概率差异。
候选设备映射必须以空闲 NPU 的实际容量和实测为依据；本次没有生成可直接派发的
NPU 配置，没有声称两卡训练或模型加载成功。
