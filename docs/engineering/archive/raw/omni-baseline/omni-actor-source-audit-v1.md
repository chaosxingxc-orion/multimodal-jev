# Qwen3-Omni / verl actor 接口核查

## 证据与边界

固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
Transformers 5.14.1、verl 0.8.0.dev0、vLLM 0.30.0+empty。
从已退出的 CPU 数值探针容器读取五个源码文件，并逐字节重新核验。
vLLM 为 editable 安装，实际路径是 `/vllm-workspace/vllm/vllm/`，不是 site-packages。
精确路径、SHA256、容器与探针结果见 [receipt](../../../omni-actor-source-audit-v1.json)。

`tools/jev/probe_omni_actor_source.py` 从固定源码 AST 提取方法，去除装饰器，
使用真实 CPU torch、合成张量与模拟 audio tower。无模型权重、无真实音频处理、
无完整模型前向、梯度或 NPU 执行。它验证方法内的张量契约，不证明完整集成成立。
首个容器退出 1：torch 自动加载 NPU 后端时缺少宿主驱动库。
保留该容器和日志；第二个容器设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 后退出 0。

## 已确认的适配要求

| 接口 | 固定源码行为 | actor 要求 |
| --- | --- | --- |
| 音频 mask | Omni processor 输出 `feature_attention_mask` | 不沿用 ASR 的 `input_features_mask` |
| 多音频 | 每个样本可有多个音频；verl 按样本顺序沿第 0 维拼接每个字段 | 保留每个样本的音频数量、顺序和 mask，不能按每样本一段音频索引 |
| 特征宽度 | verl `extract_multi_modal_inputs` 直接 `torch.cat(..., dim=0)` | 同一 batch 的 features 和 mask 需统一最后一维宽度，再按微批切分 |
| 音频长度 | HF `get_audio_features` 从 mask 求长度并去除 padding | 不能照搬 vLLM 的 lengths-only 输入；本版 HF 会将显式 lengths 清空，然后对 None 调用 sum |
| 位置 | 无图像/视频 grid 时，HF 三个 RoPE 轴均为 attention mask 的累计位置 | 当前纯音频任务不能仅因旧适配使用二维位置就判为错误；需后续完整前向等价验证 |
| 响应对齐 | verl 非 remove-padding 路径使用 `logits[:, -R-1:-1]` 对应 R 个响应 token | 需左填充 prompt、右填充 response，并只在 response mask 内计 loss |
| 显存 | HF thinker 对完整 hidden states 执行 lm_head；verl 随后才截响应区域 | 不能声称原 ASR 的 response-only logits 优化已在 Omni 生效 |
| MoE loss | HF 仅在启用 router logits 且传 labels 时向内建 loss 加辅助项 | PPO 自定义 loss 是否使用 router 辅助项必须显式定义，不能假定自动加入 |
| 权重映射 | vLLM 支持 thinker 前缀及独立 model/lm_head 前缀，忽略 talker/code2wav | 优先审查 thinker actor；静态映射不证明 LoRA 导出、热加载或 logprob 一致 |

纯音频位置还需区分旋转位置与因果 mask：HF 文本模型把二维位置扩成四轴，
四轴输入的首轴用于因果 mask，其余三轴用于旋转。原生三轴输入没有显式 text 轴。
本次仅证明旋转位置数值；尚未验证两种完整注意力路径相同。
HF 在 padding 处填 1，旧 ASR 构造器填 0；这些位置不参与有效响应 loss，
但在验证前不把该差异概括为完整等价。视频/图像和 remove-padding 路径不在结论范围。

## 实际探针结果

- 双音频样本加单音频样本，含左右 padding：三轴位置等于累计位置，delta 为 0。
- 合成特征宽度 5 与 7 直接交给 verl 提取方法：出现预期的拼接维度错误。
- 将第一样本两段音频补至宽度 7 后，得到三段音频，mask 长度依次为 `[5, 5, 7]`。
- HF 方法按 mask 展平特征后向模拟 tower 传入 `[4, 17]` 特征和上述长度。
- 仅传 `audio_feature_lengths`：复现 `'NoneType' object has no attribute 'sum'`。

这些输入是为暴露接口问题构造的张量，不是模型质量样本或有效 encoder 特征。

## 接入顺序

1. 新增 Omni 专用 batch 适配，承接已校验的 SFC 采样身份、原始响应 token 和 old logprob。
   继续使用 waveform + 任务定义；保留 supplied history 和有序多音频。
2. 使用真实 processor 对混合单/双音频样本验证单样本与 batch 的 token、音频掩码和位置；
   动态微批不得让音频顺序或 GRPO 同 prompt 分组漂移。
3. 首次完整 actor 前向采用 padded、sequence parallel 1 路径，核对响应 logprob 与
   冻结 vLLM rollout，并测量完整词表 logits 的实际峰值内存。
4. 前向成立后才验证 LoRA 参数选择、adapter 导出/加载、零优势不更新、非零梯度和 checkpoint。
   remove-padding、sequence parallel、多卡训练分别验证，不从 CPU 探针推导支持。

当前执行优先级仍为冻结推理及离线教师提示词优化。本核查没有启动权重训练，
也没有改变正在运行的正式评测配置。
