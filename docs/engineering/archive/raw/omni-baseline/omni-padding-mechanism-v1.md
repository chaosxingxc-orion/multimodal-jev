# Omni 左补齐机制检查

首条完整预训练诊断显示，文本左补齐使 false token log-prob 变化0.14458656，
前四个条件完全相同。完整TRAIN6诊断仍在运行，不能从单条认定根因。

从该诊断的精确容器读取实际安装的 Transformers 源码，没有修改容器：
`modeling_qwen3_omni_moe.py` SHA256
`a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340`；
`masking_utils.py` SHA256
`5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2`。

## 源码确定的路径

- Thinker 收到显式 position_ids 时跳过 get_rope_index 和 rope_deltas 自动计算。
  原参考与此次五条件均显式传二维位置；不能把缓存的 rope_deltas 当作已确认原因。
- TextModel 将二维位置扩展为四轴：一轴供 mask，三轴供 rotary。
  左补齐位置虽设为1，有效 token 的位置仍是0到L-1。
- 无padding的 SDPA mask 可能返回 None，走 is_causal；有padding则生成显式布尔mask。
- SparseMoeBlock 将完整 batch×sequence 展平，包括 padding 位置；router/expert
  不接收 attention mask。每个token独立top-k，没有发现容量淘汰或跨token竞争逻辑。
  padding 可改变路由输入及每个专家线性运算的矩阵形状，但这不证明有效token路由改变。
- get_audio_features 根据 feature_attention_mask 先选出有效帧。文本补齐不是直接给
  audio tower 增加输入帧，但尚不能据此跳过真实模型激活检查。

## 实际 mask、rotary 与 SDPA 检查

固定镜像运行 `probe_omni_padding_mask.py`，无模型/数据挂载，使用真实 HF 函数与
torch SDPA，合成Q/K/V，随机种子17。长度包括7+3左补齐与1076+154左补齐；
后者对应首条1074 prompt+2响应及154 padding。head_dim=8、三头，
不是完整预训练模型维度或已保存的真实Q/K/V。

四个条件（两长度×FP32/BF16）均确认：有效位置、rotary输出和因果边完全一致；
有效query不见padding key。将padding K设为100、V设为-100后有效输出逐位相同。

| 有效长度 | dtype | 无padding隐式/显式mask差异 | 无padding/左补齐SDPA差异 |
| --- | --- | ---: | ---: |
| 7 | FP32 | 0 | 1.19209e-7 |
| 7 | BF16 | 0 | 0 |
| 1076 | FP32 | 0 | 2.38419e-7 |
| 1076 | BF16 | 0 | 0.0009765625 |

因此不能把“mask路径不同”直接等同“有效因果关系错误”；同长度的隐式/显式路径
在这组张量上相同，增加masked位置后可出现浮点差异。该合成例并不能解释真实模型
最终0.14459的误差大小，也没有检查真实MoE路由翻转或逐层误差传播。

容器 `3ec639f9e388adc8358474a2e437986013c4c087a820a384a9305c29a17c48ea`
退出0，源码 `c844629`，2 CPU/8 GiB/300秒上限，无网络/设备。
已核对源文件哈希与原运行容器一致、源码快照与终态，见
[运行结果](../../../omni-padding-mask-v1-verified.json)。

下一步仍等待完整TRAIN6五条件结果，再固定针对实际预训练激活的诊断：比较相同有效
位置上的输入、attention输出、router top-k和MoE输出，区分首个产生差异的层。
暂不自动派发另一份大模型前向，不改原0.1门槛，不以本检查替代actual-verl复验。

## 逐层记录工具验证

新增 `omni_activation_trace.PaddingTrace`，用只读 forward hooks 捕获每层有效token的
输入、attention输出、router logits/weights/top-k indices、MLP输出和层输出。
参考张量逐事件单独保存，比较时按attention mask对齐有效token，一次只加载一个
参考；核验SHA、dtype和形状。对路由分别报告顺序变化与专家集合变化，避免把
top-k顺序交换误称为更换专家。默认参考tensor payload预算8 GiB（不含序列化开销）。
只允许eval/inference单行调用，异常时也清除hooks；不改返回值、不保留计算图。
它不独立认证模型/输入身份，后续大模型入口必须绑定源码、模型哈希、相同response
和完整输入清单。当前类内参考manifest保存在内存，跨进程验收须另存并核对报告。

5项本地测试覆盖有效token对齐、参考文件篡改、预算/前向/不完整调用失败、hooks
清理及训练/autograd拒绝。固定镜像另运行真实两层随机Omni BF16 SDPA/MoE：
14个记录点完整；开启hooks前后原始及补齐输出均逐位相同，模型状态未变，hooks
全部清除。身份比较全部相等；更改有效token的反例在layer-00.input即被检测。
此小型随机模型的补齐比较14点均相同，因此只证明工具有效，未复现真实预训练偏差。

源码 `cfa412c`，容器
`0659d7fdae4cd3b320c31db0f02f186faaa8b982aaca9b761d675f94196729b0`
退出0，2 CPU/8 GiB/300秒上限，无网络/设备及预训练模型/数据挂载。
参考张量payload总计2640字节，14个文件SHA与报告全部核对；见
[固定镜像结果](../../../omni-activation-trace-v1-verified.json)。尚未在预训练Omni上运行这些hooks。
