# 冻结 Omni 学习集采样运行器

`tools/jev/sample_sfc_omni_learning.py` 独立于原 greedy 质量基线，读取已固定的
TRAIN78 processor 协议及学习划分。覆盖 6 个回合、4 个完整会话、78 个决策，
每个决策 K=2，目标完整产出 156 条逐 token 采样记录。没有训练、教师 API 或 TEST 目标读取。

## 输入与执行

配置严格包含 baseline_config、baseline_config_sha256、partition、partition_sha256、
group_size=2、seed=17、max_tokens=1024。baseline_config 只用于复用已核查的资产、
processor 版本与输入人口；采样配置单独生成，原 greedy 运行快照保持不变。

运行器复现完整 TRAIN 划分，要求四个工程会话全部在 learning 中，校验本地
processor 资产，并逐决策比较真实展开输入与原预检的 prompt/token 哈希。
实例化真正的 vLLM SamplingParams 后检查温度、截断、惩罚、停止和概率字段，
拒绝额外 grammar、allowed-token、logit bias 或自定义参数。
显式 `watermarking=False`，避免配置的引擎水印改变行为分布。

实际推理设置 BF16、TP2、context4096、max_num_seqs=1、max output1024、
prefix caching off、最多双音频，`generation_config='vllm'`、`raw_logprobs`。
真实执行前后检查全部模型资产；policy 哈希绑定仓库、revision 和模型文件哈希列表。
模型加载后再检查生效的 probability mode 和 generation config。

每次 dispatch 前保存请求意图，返回后先保存原始 token、文本和概率，再执行严格转换。
格式失败输出照常保留；不修复、去重或按奖励筛选。输出目录必须全新，不能恢复时
重放状态不明请求；精确容器监督和资源锁由外层调度器负责。
每次请求完成更新 progress，异常写 terminal；外层超时/强杀仍须以 Docker 终态为准。

## 已完成预检

源码 `9918fcd` 在固定镜像 CPU2 / 12GiB / 无网络 / 无 NPU 设备容器中执行
`--preflight-only`，退出 0，用时 49.13 秒。完整 78 决策通过，实际 SamplingParams
与指定参数一致；框架将 None 停止列表正规化为空列表，校验兼容这一等价表示。
见 [独立验收](../../../omni-sampling-train78-preflight-v1-result.json)。

64 项相关测试通过，包含学习边界、配置和输入文件漂移、响应身份、概率缺失与错位。
输入选择测试使用隔离假数据；真实完整人口和 processor 兼容性由上述 CPU 容器覆盖。

## 尚未派发

没有启动新的 NPU 采样，也没有修改现有 NPU4–7 队列或启动权重训练。
当前工具的实际 generation 分支尚未实跑。还需：

1. 独立验收器检查 78 个 intent/raw/result 的精确对应、全部 156 条记录、容器身份及模型哈希；
   校验失败时不能只依据 result.status 接受结果。
2. 接入现有资源锁与不可变调度快照，明确等待原质量基线结束，不抢占正式运行。
3. 保存完整实际引擎配置；确认 Ascend sampler、K=2 和逐 token 返回概率的运行行为。
4. 在 frozen thinker 上比较有效响应位置的 HF/vLLM 概率，并单独测量内存。

本轮证明 processor 与采样参数接口兼容，不证明实际模型采样、概率一致性或 RL 效果。
