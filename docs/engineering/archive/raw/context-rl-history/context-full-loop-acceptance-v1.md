# 全日程训练循环与跨进程恢复验收

2026-11-14，沿用固定镜像52ba94e1、Qwen3-ASR-0.6B、BF16执行、FP32 master/Adam、
DP4/global128/micro8×4。正式人口仍为40513学习录音、两轮634更新，日程manifest
d9cd5832aa160df46a4c10a2051ff12c352d2a2609e2cffd6901267b28a814f6。

`tools/jev/train_context_instruction_npu.py`现在使用完整日程loader、纯输入缓存、
单后台线程整批ahead1预取、原生verl actor及联合提交的四rank checkpoint。
前台消费时绑定策略和监督，实际processor形状必须重现固定rank分发。
`context_training_checkpoint.py`在改动参数前验证实验身份、拓扑、全部master/Adam；
四rank共同通过校验后恢复。空的未拥有分片允许没有Adam状态，非空分片不允许缺失。

## 实际派发与连续链

第一段容器d7d64c970a6d338383ce5b037368499fd1406f93131cd8bd6fa2748789b13ece
已启动于175卡4–7，执行全日程第1–2更新。它是同一全人口日程的工程前缀，
不是缩小人口后的研究训练，也不是完整Stage1已经完成。

本地驱动`.local/decision-objective-research/continue_context_full_schedule_acceptance_v1.py`
PID277491准确等待该容器终态。通过独立收集后才启动新进程，从第1步checkpoint恢复，
重做第2步，独立比较原连续执行的第2步结果：

- 四rank所有FP32 master和Adam状态逐张量精确一致；
- 完整128个BF16执行张量精确一致；
- 实际输入、监督、权重及rank分发对应固定日程；
- 每个checkpoint的四分片和执行导出符合联合commit文件的hash。

首段和恢复段均持评测共享资源锁，旧评测driver保留。任一失败记录终态并停链，
不因等待时间长而重启原容器。驱动状态和日志位于同目录
`context-full-schedule-acceptance-v1-status.json`及`*-driver.log`。
恢复入口是`launch_context_full_schedule_prefix_npu_v1.py --resume-replay`，
收集入口同名前缀的`collect_...py --resume-replay`，首次派发不带该参数。

## 已完成证据与未完成边界

本地checkpoint、日程、缓存批次和behavior概率检查21tests通过。首次未设置PYTHONPATH
的测试收集失败，显式`PYTHONPATH=src`后运行通过；没有把收集错误当算法失败。
先前独立DPv3更新/进程内重放通过，并且DP更新策略256音频×60概率HF/vLLM精确，
见`context-native-actor-dp-npu-v3.json`与`context-engine-dp-updated-npu7-v1.json`。

本次全循环及新进程恢复在实际终态收集前仍未验收。两步保存频率为每步，
只用于恢复验证，不能作为正式训练吞吐测量。完整Stage1、selection、生产token准入、
实际NPU RL rollout生命周期及Stage2增益曲线、FSC冻结迁移均未完成。

## 后续实际终态

首段d7d64c97及新进程恢复段5fd0e43f均exit0并独立收集，驱动已进入
`accepted_bounded_loop_and_fresh_process_replay`。证据分别为
`context-full-schedule-prefix-npu-v1.json`和`context-full-schedule-resume-replay-npu-v1.json`。
新进程从step1重放step2，四rank master/Adam与连续执行精确一致，128完整BF16
执行张量亦精确一致；正式训练未自动派发。

首段真实冷缓存计时（各rank最大值）：

| 更新 | 准备秒 | actor秒 | 完整秒 | 后台输入服务秒 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64.34 | 27.14 | 93.81 | 63.74 |
| 2 | 27.63 | 15.61 | 44.82 | 57.23 |

两批cache hits均0；第二批后台服务与第一步计算重叠，但仍前台等待27.17秒。
两步包含热身和每步checkpoint，不是稳定吞吐。结果支持训练前准备输入，而不能
据CPU热缓存短测声称端到端已提速。

新增`local_audio_mirror.py`按内容hash发布并核验本地副本，
`CachedContextPreparer`可用该副本读取音频，同时保留原request.audio和缓存键，
每次消费复核本地bytes；缺失/损坏失败，不静默回退Ceph。镜像代表已固定的数据版本，
源文件后续变化不会替换已发布的副本。新训练入口可验证完整mirror manifest后启用，
此可选分支尚未经过真实NPU吞吐验证，已完成恢复验收的远程快照不变。

全学习音频镜像b4d97c6d已派发固定CPU镜像，不占NPU；目标为40513条、只读Ceph，
写`/data/xingchao/multimodal-jev/simple-final-v1/context-learning-audio-mirror-v1/run`。
collector `collect_context_learning_audio_mirror_v1.py`会在终态后逐副本hash并独立重建
完整学习人口。镜像完成仍不等于processor缓存完成，后续还需固定processor冷构建成本。
