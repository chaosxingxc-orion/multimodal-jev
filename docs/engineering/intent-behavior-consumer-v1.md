# 行为概率进入真实actor输入的消费探针

2026-11-12 UTC。新增probe_intent_behavior_consumption_cpu.py，以本地真实processor
重新编码音频，分别调用原行为配置和候选行为配置的intent_actor_inputs.prepare。
逐来源要求音频特征、输入token、Span位置、候选、policy/request身份、监督、
mask及权重一致；行为分布可有已登记后端容差内差异，仍检查全候选及联合比率。

探针重放固定12个来源日程位置，96来源使用覆盖89条独立学习来源；不执行
12次更新、不加载语音模型权重、不调用优化器。resident模式要求显式新行为
绑定，恢复配置同时通过原Adam/参数状态凭证认证。历史控制模式要求两次读取
同一无resident绑定的真实配置，不能将其结果称为resident消费成功。

11项张量对照测试在固定镜像无设备CPU容器通过，实际退出0；覆盖音频、位置、
候选、policy、监督、行为概率、来源选择改变时拒绝。容器e9fbc24d…。
历史控制使用Q6训练配置（其输入行为为Q5已接受策略），不是Q6更新后的行为。

第一次真实控制探针a057a5fa…实际退出1：三份/home下的固定凭证未挂载，
在输入pins检查阶段失败。保留原失败日志；核对本地哈希后，将三份文件复制
到新实验assets，并按原绝对路径只读挂载。第二次容器96372bc8…在12:25 UTC
实际running，尚未输出完成结果。docker wait的60秒观察超时不是失败或重启
依据，原容器继续。第二次未修改探针或训练配置。

行为绑定工具同时修复跨冻结目录的验收脚本路径比较：只允许脚本内容哈希
完全一致时归一化自身文件路径；保留原凭证及全部输出pins，实验内容仍逐项
相同。新增“不同路径相同代码成功、修改代码即使更新pin仍拒绝”测试；本地
绑定和恢复26项测试通过。该修改不改变已冻结NPU训练或推理队列。

当前仍缺真实resident suite完成后的actor消费结果以及动态策略在线训练闭环。
共享根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-behavior-consumer-historical-cpu-v2/`。
[当前证据](intent-behavior-consumer-v1-evidence.json)。本文件记录已执行范围，不是
真实音频消费成功或强化学习收益声明。
