# 统一基线收敛与当前上下文缺口

2026-11-13（Asia/Shanghai）。执行顺序仍为任务定义→工程/配置冻结→训练。
本轮没有模型forward、NPU训练、权重下载或历史快照修改。

## 决定与实际状态

维护一套共享底座，登记GRPO、RLOO-clipped、PPO-critic三个算法配置，各自验收。
GRPO先接入，RLOO作为同底座优势尺度参照，critic不阻塞已通过验收的无critic配置。
三者目前均未集成验收；不能把JSON设计登记称为工程冻结。
现有`span_finite_credit.py`仍为可加奖励的有限候选精确边缘PPO，clip=.2、dualclip=3，
没有critic或GRPO标准化；最近NPU两步仅为CE。它保留为历史定制诊断，不冒充标准算法。

固定运行候选沿用Qwen3-ASR-0.6B、本地权重、BF16 LoRA q/v r4 alpha8 dropout0、
FP32 master/Adam，175卡4–7 FSDP DP4及vLLM TP1。镜像和verl commit见算法v2配置。
任务输入、优势估计、reward、CE、KL、采样预算、恢复和评测需要共同版本化。

## 当前输入实测

旧context-v3使用60意图完整定义，但只供应历史槽位短定义及部分边界字段。
它的max7412不能证明当前槽位v3、政策v3、边界v4也满足8192。

设计probe现在显式供应类型definition/thinking_requirements/role_rules/scope、
所选政策字段、边界scope及未决约束。相同政策值共享一次，各类型override可无损展开；
历史争议、来源样本及支持计数不进入actor。未加入逐样本监督mask。
这只是任务可行性探针，不是已冻结生产renderer；没有改变训练入口。

相同固定镜像、真实processor、相同15面板、learning中位数/最长两条音频，三臂测量：

| 辅助Q预算 | 请求数 | token范围 | 超过8192 |
|---|---:|---:|---:|
| 0（Direct） | 30 | 5243–14071 | 28 |
| 2 | 30 | 5322–14231 | 28 |
| 8 | 30 | 5559–14710 | 28 |

90次均完成processor及Span定位，但84次超限。同一音频/面板三臂的知识完全相同，
所以Direct也承担共同定义长度。容器无NPU设备、exit0、未OOM；exit0是测量完成，
不是上下文通过。source pins、请求唯一性、长度统计、Span位置和三臂知识hash已另行复算。
没有模型推理、全人口上下文验收、语义正确性认证或NPU资源结论。

证据：`docs/research/panel-context-probe-v4-verified.json`；共享原始配置、快照和记录：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/panel-context-probe-v4-full-slot-contract`。

当前8192候选明确标记失败，selected_context_limit=null。下一步先定稿语义表示、
可无损共享的公共规则及固定菜单，再预注册上下文预算；16384只是一种待评估选项，
不能由两条音频的CPU长度自动认定它可用于全量NPU训练。禁止截断/删掉必要语义来通过。

## 固定verl优势标准化核验

直接获取固定commit `e9405d70b77d56a23e72bb2e40a17b906856e913` 的上游源码，
core_algos SHA与原锁定值`fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24`一致。
来源、函数行号及hash在`docs/research/rl-source-normalization-audit-v1.json`。

`compute_gae_advantage_return`先计算未标准化return，随后对advantage调用masked_whiten。
一轮任务raw A=R−V，但函数返回的并非raw A。此前配置未说明这一步，现对齐为
`(A−mean_valid(A))/sqrt(unbiased_var_valid(A)+1e−8)`，critic仍用raw return。
设计要求在完整行为批的真实query×G动作上一次计算，padding不参与；之后再分rank、
切microbatch、执行两轮更新，不逐rank/逐Q重复标准化。该分布式语义仍待工程验收。
GRPO沿用组内sample std+1e−6，RLOO仍是leave-one-out均值基线，无std标准化。
没有调用这些函数开展新训练；本轮是源级审计及设计纠偏。

## Step-wise与后续交付范围

当前一次forward并行填Span，语义节点不能直接当作环境时间步套GAE。
按相同起点依次检验Final-only、同输入的固定过程reward、另立ID的节点条件信用。
Q-input与Q-credit保持CE、KL、动作成员和采样预算相同；过程缺标mask，不补false。
可加reward节点信用可精确求期望，交互reward必须说明因子与局部surrogate，不能称为
标准联合PPO的等价替换。GRPO归一化可放大低预算过程差异，reward上限不是梯度上限。

接下来仅围绕三个发布包收敛：

1. 任务包：完成定义冲突、可靠语义正负监督、固定Q/Final关系、输入表示、指标及FSC冻结迁移；
   原生clean意图真值与新增语义Q监督分层，输入歧义可mask，不强求零未决。
2. 工程包：任务冻结后，接入GRPO/RLOO并验收驻留FSDP→保存重载→vLLM概率、
   全量输入、更新/恢复、old/ref同步与实际资源，PPO critic单独验收。
3. 实验包：固定LR、batch、更新数、seed、评测点及资源上限，Stage1学习定义消费，
   Stage2从固定起点训练RL+衰减CE；固定配置运行，缺陷修复发布新版本。

本轮变更仅涉及设计probe、配置及文档；三门槛仍未通过。不能承诺开始训练的具体时刻。

验证：25项相关CPU测试通过；六份JSON及字段/引用一致性检查通过，git diff --check通过。
这些检查不替代NPU生命周期或任务效果验收。
