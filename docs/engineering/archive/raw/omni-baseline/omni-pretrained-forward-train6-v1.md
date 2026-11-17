# 冻结完整 Thinker 的真实音频 CPU 前向

本探针将前期合成/静态检查推进到真实 checkpoint 的加载与音频前向。
源码快照为 `96c5b08`，实际派发记录见
[launch](../../../omni-pretrained-forward-train6-v1-launched.json)。
固定镜像不变，无 NPU 设备、无网络，CPU 16 核、内存 160 GiB，timeout 14400 秒，
终止宽限 30 秒，不自动重试。派发前宿主报告 available 约 1.8 TiB。
仅允许现有本地 Qwen3-Omni，不下载模型，不更新权重。

## 输入与加载

复用 TRAIN78 采样配置的输入验证：六条 TRAIN 回合、四个完整原始会话，
确认属于固定 learning 分区，并核对 CPU processor 的真实展开 token/hash。
输入准备覆盖全部 78 个决策，模型前向只取每条回合的首个声明决策，共六次。
这是显式工程探针，不是缩减后的正式任务评测；不读取目标文件。
固定人工响应为 tokenizer 编码的 `false` 加 EOS，不能当成采样或正确答案。

加载前逐文件核对官方资产哈希。直接调用固定 Transformers 的
Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained，BF16、device_map=cpu、
SDPA。保留原始 loading-info；missing/mismatched/error 任一非空就失败，
unexpected 仅允许原 checkpoint 的 talker/code2wav 分支。
确认参数全部在 CPU、无 meta 残留、requires_grad=false。
实际参数和 checkpoint 布局由既有完整 header/meta 检查提供前置依据，
但是否实际加载成功以本轮运行结果为准。

## 前向与结果

真实波形经过同一 processor，保留有序多音频、函数定义和 supplied history。
输入为 prompt + 固定响应，使用显式二维累计位置、feature_attention_mask、
use_cache=false。逐条前向，不截断输入；上下文必须在 4096 内。
从完整 logits 的正确 causal offset 取固定响应概率，要求有限且非正。
记录逐条音频数、响应 token/logprob、耗时和进程峰值 RSS。
完成后再核对原 checkpoint 文件哈希，保存 result 和 terminal。

本轮通过最多证明该版本可加载完整 pretrained Thinker 并对这些真实输入完成
有限数值前向，不证明输出正确、音频依赖、HF/vLLM 概率一致、verl actor 集成、
反向传播或 NPU 训练。CPU BF16 与 NPU BF16 也不能直接视为数值等价。

## 观察与后续

以 launch 中精确 Docker ID 查询状态。宿主超时或客户端观察中断不触发重放。
容器 timeout 可能早于脚本写 terminal，应保留 Docker 退出状态、日志和各阶段
progress，不将缺少 result 解读为完成或恢复依据。当前文件仅记录派发协议。
后续验收须重新核对容器资源、源码/配置、loading-info、全部六次记录和终态。
若通过，再设计相同 token 下 actor 微批和 vLLM 概率的比较，禁止直接据此启动训练。

## v1 失败与 v2 修订

v1 已确认退出 1：from_pretrained 返回后，loading-info 含 set，原 JSON 写入
失败，尚未执行音频前向。失败容器、日志和 terminal 已保存；不将其算作加载验收。
v2 源码 `42b018c` 只将加载报告中的 set 转为排序列表、tuple 转为列表，
保留所有缺失/额外/形状错误字段，未知对象类型仍报错。模型、输入和资源预算不变。
v2 是故障原因确认后的新显式尝试，不是自动重放；精确容器见 v2 launch 记录。

新增 `verify_omni_pretrained_forward.py` 独立检查成功容器、完整源码和配置、
原始学习样本绑定、加载报告、六份前向记录、响应 token 和有限概率。
验收端须使用固定 tokenizer，命令要求 launch/intent 的已登记 SHA256。
16 项单元测试验证记录缺失、错绑、非有限概率及加载报告序列化；
这些测试不替代实际运行结果，真实验收待 v2 完成后执行。

## 后台独立验收观察

`watch_omni_forward_acceptance.py` 已固定为独立脚本并后台运行，
精确 PID/start ticks、脚本与 plan 哈希见 v2-observer-launched.json。
它只观察 v2 原容器，最长 15000 秒；瞬时 SSH 错误继续查询同一 ID。
原容器失败则记录终态，不派发验收，更不重新加载/推理。
成功后捕获 Docker inspection 并绑定哈希，按先前 prepare 的不可变源码和
argv 创建一次 CPU verifier，写入 fsync intent 与精确 ID 后仅启动一次。
验收后重新检查原容器终态，保存 acceptance 哈希和 observer 终态。
观察超时或派发状态不明均保留现场供核对，不做自动重试。
本轮 37 项 observer/forward-acceptance 测试通过，包含上述失败分支；
真实模型和验收结果仍以后台运行终态为准。

## v2 完成并独立验收

完整 v2 容器 `e8f68cb9ebd5a607722e676eab977d08cf750fd2d86342b5001a3f3d3cc44be3`
已退出 0。独立验收容器
`2f02ce68d53381c18eb28f92ca87c94d82cb9d280d29dbad44e153c603722d56`
同样退出 0，观察器复查原容器终态后正常结束。完整凭证见
[验收记录](../../../omni-pretrained-forward-train6-v2-accepted.json)。

实际加载 31,719,205,488 个 Thinker 参数；六条输入全部完成，音频数依次为
1/1/2/1/1/2。总耗时 1831.10 秒，峰值 RSS 119,766,804 KiB（约 114.22 GiB）。
固定响应为 false + EOS，所有逐 token 概率有限且非正；没有读取目标或更新权重。
模型加载前后文件哈希检查完成，Thinker 无缺失、形状不匹配或加载错误。
这关闭了完整预训练音频 CPU 前向的验证缺口；任务质量、真实 verl wrapper、
HF/vLLM 概率一致性及 NPU 训练仍需各自验证。
