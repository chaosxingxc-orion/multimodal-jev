# 固定参考概率缓存与真实切换结果

2026-11-14。Stage2候选代码保持独立，context-prob-v1冻结训练文件未改变。

## 固定参考缓存

`tools/jev/context_reference_cache.py`按固定策略ID、base和contract分命名空间；
entry按完整输入tensor身份与marker位置确定。缓存只保存原始vLLM概率与请求身份，
不保存监督、动作或可变化的behavior。每次读核验内容hash和bind_rollout身份/概率范围。
原子link避免替换已有文件；重复发布必须完全相同，写失败清理临时文件并保留原条目。
文件flush/fsync和目录同步均保持，输出必须为本地存储。

CachedReferenceEngine是固定参考facade：命中直接读取，未命中时只能调用所绑定的
固定engine；无engine时缺条目直接失败，不静默用当前策略替代reference。
未来可在Stage1终态固定后一次构建全学习人口的reference，再关闭参考模型，
只保留当前策略rollout engine；当前尚未派发此全人口推理，也未验共卡容量。

8项CPU测试通过：恢复复用、不匹配输入、旧策略、禁止标签/动作、文件损坏、
NaN、重复写与不同参考策略隔离。测试中本地存储准入被替换，实际存储检查另列。

实际历史记录验收：使用已独立收集的context-engine-rl-updated-npu7-v1的256条
vLLM输出及历史纯输入cache，逐条重新核验所有输入tensor hash/positions，
写入真实/home ext4、恢复新cache对象，无engine读取256条全部精确。
证据context-reference-records-cpu-v1.json（记录每个文件hash与原vLLM文件hash）。
参考policy为历史诊断69bad9a9，不是正式Stage1终态；不得用于正式Stage2参考。
这证明实际记录的存取身份与恢复，不是新的模型推理或连续RL实验。

## A/B/A真实切换已独立通过

v2 f6dc19cb在175远程13:41:03 UTC exit0/nonOOM，driver324510已完成独立收集。
证据context-resident-switch-npu-v2.json：16音频×3轮、60概率全0误差，
源step1/2的全部LoRA/head/marker张量与重导出精确，A/B输出确有变化。
两个切换0.79518/0.79487秒，CPU注册LoRA112张量/缩放和活动slot核验通过，
A→B→A未重启引擎。该时间不包含每轮策略导出和actor更新成本。

尚未闭合：同actor当前策略真实rollout→更新→导出→切换→下一次更新、
该循环的新进程恢复、正式行为概率数值契约、稳定吞吐及完整选择集曲线。
不能将本次两策略静态切换推论为任意后续训练策略已经一致或有收益。
