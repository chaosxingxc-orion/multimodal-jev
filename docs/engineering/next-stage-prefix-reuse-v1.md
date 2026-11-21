# 下一阶段：稳定定义前置与缓存复用

2026-11-19 用户指定下一阶段训练调整提示词，以最大化前缀复用。
本版本只面向现有简单意图成员判决，不增加复杂任务，不改变gold或候选集合。

输入顺序：

```text
system: 任务规约、输出约束、版本、完整有序候选定义与问句
user:   本条可变历史（若有）、本条音频
assistant: decision_id_1 <|jev_decision|> ... decision_id_K <|jev_decision|>
```

定义区不放音频路径、样本ID、时间戳、标签或随机候选顺序。所有marker仍在音频之后，
因此每个判决能看到完整音频。历史是可变内容，不混进共享前缀。各数据集的本体不同，
其prefix分别缓存；不能为了提高命中率把不同任务合并成同一个key。

旧布局先音频后定义，实际SLURP样本在音频前只有36token，音频后有2764token。
新布局将稳定定义前置，理论上显著增加跨音频可复用区间；具体token数和完整块命中率
须通过实际processor和NPU统计，不能把字符数当token收益。旧模型没有按新顺序训练，
不能把旧checkpoint直接换模板后的分数混入本轮曲线。

## 实现与启动接入

共享编译器 `encode_context_request(..., prompt_layout='jev-stable-prefix-audio-readout-v1')`。
缓存准备器 `CachedContextPreparer(..., prompt_layout=...)` 透传该配置并计入缓存key。
不传参数仍使用历史布局，保证正在运行的冻结配置不变。
设计配置：`configs/experiments/context-prefix-reuse-next-stage-v1.json`。

下轮Stage1、Stage2 actor、rollout、reference生成与评测必须共同传入该布局字段，
不能只修改推理prompt。Stage1仍为InfoNCE；Stage2仍为GRPO，reference由采用新布局
训练后的Stage1生成。旧processor输入cache、旧reference概率不能冒充新输入缓存。
当前已经完成共享编译器与预处理缓存接口，尚未发布全套训练配置或新训练任务。

训练期间语言骨干LoRA在变化，旧policy KV不可跨optimizer update使用，也不能把
需要梯度的共享prefix简单detach后用于训练。冻结策略rollout/评测可按policy身份复用；
冻结声学或预处理缓存独立管理。实际policy/marker/head/runtime身份变化必须隔离。

## 缓存实验与边界

固定镜像候选v0.0.99，两条真实音频，四次执行：首次无命中；重复及反序请求命中2816token，
每条仅重算77/78token，并恢复head所需的前缀隐藏状态。冷/热logit和prob差异为0。
两条请求首次约4.23秒，热请求约0.17–0.19秒；启动约175秒。
这是重复输入诊断，绝非9226条独立音频全量提速，也不包含启动成本的10倍验收。

原生缓存候选与冻结对齐runtime对比，两条最大prob误差0.01150/0.00949，
logit误差0.609375/0.71875；阈值翻转和Top1变化均0，但未达到原精度准入要求。
所以当前不能发布为已对齐可复用基座，正式TEST和迁移仍暂停。
证据：`/srv/multimodal-jev-xingchao/operations/evaluation-cache-v0.0.99/collected/result.json`
与 `frozen-runtime-comparison.json`。

CPU24项测试通过：共享前缀跨音频稳定、历史隔离、任务变化失效、marker因果顺序、
缓存布局隔离、历史协议回归、前缀隐藏状态恢复与错误命中拒绝。
后续必须实测固定镜像processor编码、跨不同音频命中、policy切换失效、缓存淘汰，
并完成缓存/非缓存与actor的数值一致性。用户随后决定效率优化收口，不再以10倍作为前置；
实际收益仍如实报告。下一轮错误诊断和目标配置见[末点诊断与实验设计](stage2-endpoint-errors-next-round-v2.md)。
