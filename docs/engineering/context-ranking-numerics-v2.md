# 排序奖励概率的数值表示修复

macro-training-v5 的全量数据/日程预检已在固定镜像通过，两个NPU actor加载成功，
两个推理实例完成reference和behavior各一个128-query macro。首次optimizer更新前，
双信用组构造拒绝FP32排序概率饱和；无checkpoint、无训练收益，不是OOM/core dump。
失败journal与运行文件20项按原哈希收集在operations/macro-training-v5/failure-collected。

使用全量日程首批真实标签和actual logits复算：128条、60候选，在正负候选排序对中
有2个FP32 sigmoid结果为1，最大绝对logit差17.0625。单候选实际概率都严格在(0,1)，
因此存在性概率本身正常；此前只测固定小幅logits遗漏了此范围。

context_dual_credit.py新增数值协议dual-credit-cpu-pair-fp64-v2：
- 排序只对小型CPU logits张量做FP64差值/sigmoid/logprob及匹配的FP64随机采样。
- rollout采样、actor当前项和独立审计使用相同定义，保留候选排序、种子、G16、GRPO奖励与分母。
- 进入固定verl surrogate前，current/old logprob同时转FP32，loss、Adam、master保持FP32。
- 音频主干和LoRA/head执行仍BF16。存在性沿用真实NPU FP32 sigmoid，未改成CPU概率。
- 不clamp概率、不丢样本、不修改标签；超出新表示能力仍明确拒绝。

旧冻结包仍含v1源码和采样轨迹，不能跨数值版本直接恢复Adam或混合比较。
本次新实验从同一Stage1起点重跑，算法仍dual_credit_grpo_floor_clipped_v1，
runtime源码hash区分数值协议。独立group audit同步更新。

固定诊断入口tools/reports/diagnose_context_macro_rank_saturation.py，宏模板
configs/portable/operations/context-macro-rank-saturation.json：实际失败128条均可重算新credit，
排序概率一致且梯度有限。CPU sigmoid与实际NPU存在性最大binary-logprob差
2.86102294921875e-6；此为不同backend，不能据此放宽NPU actor/vLLM阈值或宣称训推通过。

51项信用/独立审计/session测试通过。另固定镜像CPU容器
c122a31fa37a7e558f34f2dd82040b6b537ea48c6948b3dbbcdb1db871d358d3
在运行DP2全局128/micro8,16,32/native verl/新进程恢复，以及大logit差FP32 loss梯度验证。
准确wait42669待终态，未声明通过。下一步收集CPU证据，再派发v6两步NPU验证。

固定镜像CPU已正常exit0、非OOM，独立收集4tests全部通过、0skip。报告
`docs/engineering/context-macro-dp2-rank64-fixed-image-cpu-v1.json`。
wait42669已消费终态。此证据限CPU真实verl，不能替代BF16 FSDP/NPU对齐。

已派发v6两步：driver1805957/start339597680，operations/macro-training-v6，
本地宏.local/config/macro-training-v6.json，远端/data/xingchao/jev-macro-train-v6。
manifest6794abab2d758339007fa152ff3d4e578df1c2954948422ebe4e40fbba0a1d85。
尚无NPU更新/恢复结果；完整100step效率实验及迁移评估仍待。

历史审计兼容：context_dual_group_audit按credit绑定的arithmetic显式区分FP32 v1与
FP64 v2，分别重建概率和随机采样；禁止仅改协议标签重解释旧轨迹。39相关tests通过。
新finish脚本finalize_context_macro_training.py精确pidfd等待原driver，再独立收集/audit、
生成完整步骤耗时；失败保留终态，不自动重启/派发。2实际进程身份测试通过。
v6 finalization-driver1807046/start339623331已启动，等待driver1805957。
此为固定脚本，无模型调用、无轮询，不改变普通训练观察900秒规则。
