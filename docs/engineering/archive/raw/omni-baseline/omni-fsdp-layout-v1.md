# 固定镜像 Omni / verl 分片结构核查

结论：后续NPU概率验证需要显式Thinker加载和显式解码层FSDP包裹配置，不能
直接照搬旧worker的AutoModel选择和默认wrap policy。尚未验证NPU执行、
FSDP初始化/HCCL、权重加载或前后向，未派发训练。

固定镜像真实meta模型探针已完成，源码92a626f，容器
`c3ae5d96b9b46fd201b8ddfe68412f77c33ef0cbe2d41b4ededcbc5d200189e1`，
退出0无OOM，探针4.03秒，2 CPU/8 GiB，无设备或网络。仅挂载原config.json，
不挂载checkpoint权重。见[独立核验](../../../omni-fsdp-layout-v4-verified.json)。

## 实际发现

1. 固定镜像已有NGA `nga_vision2seq_shim`，导入verl时将旧名称
   AutoModelForVision2Seq映射到AutoModelForImageTextToText。首次探针在
   导入verl之前访问旧名而失败，这是探针导入顺序问题，不能推导worker
   本身导入失败。旧worker会先经过verl包初始化。
2. shim生效后，完整原始Qwen3OmniMoeConfig仍未命中所检查的
   AutoModelForVision2Seq/ImageTextToText、AutoModelForCausalLM或AutoModel
   映射。原config无auto_map，旧fsdp_workers._build_model_optimizer默认
   选择最终落到AutoModel，不能认为它会自动加载Thinker。显式
   Qwen3OmniMoeThinkerForConditionalGeneration是已验证CPU路径的类，
   完整NPU加载仍待实现和检验。
3. 真实Thinker运行时的_no_split_modules是合并后的集合，包含不存在的
   Qwen3OmniMoeDecoderLayer。固定verl get_fsdp_wrap_policy(model,{})
   因此直接报错。顶层声明的两个音频/视觉模块不是完整运行时集合。
4. 显式选择Qwen3OmniMoeThinkerTextDecoderLayer、Qwen3OmniMoeAudioEncoderLayer、
   Qwen3OmniMoeVisionBlock后，原verl wrap policy可构造并遍历。对应108个
   不重叠参数单元，其中48个文本解码层。根单元768,270,080参数、BF16
   1.4310 GiB；每个文本解码层623,120,640参数、BF16约1.1607 GiB。
5. 外部按原checkpoint头部张量形状和模块前缀重建108个单元的参数计数，
   全部与meta报告相符，总计31,719,205,488参数。单元大小不等于峰值显存，
   仍有常驻分片、all-gather临时存储、激活和算子工作区。

## 剩余缺口

meta配置选择grouped_mm专家实现。固定Transformers dispatcher对NPU没有
专门能力核查，非CUDA路径主要检查torch API是否存在；不能据此推导NPU
kernel可运行。固定verl npu_patch.py含Qwen3-MoE/VL补丁，但没有发现Omni
类的专用替换，不能将相近模型的补丁视为Omni已适配。

旧actor把微批搬到当前设备再调用actor_module，适合进一步验证每rank一张
NPU的FSDP路径。把HF模块按层放到两张卡不能称为verl分布式actor或训练支持。
显式层包裹只是容量验证候选，原BF16专家、attention、RMSNorm等实际NPU
算子，以及FSDP collectives、完整156条真实响应概率仍须验证。

下一步准备单个真实解码层的无梯度NPU算子探针，消费已保存且有哈希的TRAIN
状态、原权重和原实现，记录支持/失败及峰值内存；不替换kernel或改变精度来
掩盖失败。这不替代完整模型、实际verl actor或训练验收。NPU4–7遵守既有
冻结基线和采样队列，本轮没有新增NPU派发。

## 失败与更正

- [v1](../../../omni-fsdp-layout-v1-failed.json)：探针在verl导入前访问已移除旧类名。
- [v2](../../../omni-fsdp-layout-v2-failed.json)：默认wrap policy找不到旧层类。
- [v3](../../../omni-fsdp-layout-v3-failed.json)：运行时集合未转JSON列表，报告序列化失败。
- v4保留默认失败信息，完成显式方案的结构核查。未修改安装包、模型、镜像或
  运行中的实验，所有探针均未加载模型权重或执行推理。

v4映射列表第一项使用alias实际类名，与第三项同为AutoModelForImageTextToText。
原查询顺序和shim已在独立核验明确记录，不把重复项当成额外支持证据。

## 后续发现：BF16 整模型 cast 改变位置编码

[buffer 精度探针](omni-buffer-precision-v1.md)确认：旧 worker 的
`actor_module.to(torch_dtype)` 在显式 BF16 加载方案中会舍入四个 FP32
位置编码 buffer，随后 FSDP cast 回 FP32 仍无法恢复。原 FP32 RoPE 与九份
历史真实预训练 prefix 张量完全一致，往返版本均不一致。完整分片实现必须
单独保留并验收非持久化 buffer，不能仅以参数哈希和 108 单元计数验收。
该探针仅使用 meta 参数、真实 buffer 和 rotary-only CPU 调用，未验证
NPU/FSDP 通信、完整模型加载或任务影响。

[完整CPU加载预检](omni-fsdp-loading-v1.md)随后通过：BF16参数meta构造保留
FP32 buffer；实际加载19,743个原始Thinker张量全部元素精确匹配，四个buffer
与认证参考一致。峰值主机RSS118.326GiB；这仍不证明FSDP/HCCL初始化或NPU前向。
