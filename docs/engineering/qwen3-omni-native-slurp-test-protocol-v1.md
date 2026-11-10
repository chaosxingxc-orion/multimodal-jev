# 冻结 Qwen3-Omni 原生 SLURP 完整 TEST

固定官方 real TEST 的13078条录音、2974个语义组，不抽样、不读目标调提示。
使用已完成 TRAIN16 探针的原生复合输出指令、taxonomy、BF16、seed17、
greedy、4096上下文、256输出token和同一官方Qwen3-Omni权重。
输入为音频和完整taxonomy，输出scenario/action/entities。无ASR串联。

原生复合任务与FSC-Jev教师提示优化独立；本次不替换为单value指令，
不把本次TEST反馈传入教师。官方评分以录音为单位，报告joint intent、
entity exact F1、word/char F1和SLU-F1；保留格式失败、重复实体和输出截断。
严格解析，不修复Markdown围栏。全部输出与容器、应用终态成功后才评分；
不完整结果不报告完整TEST成绩。权重前后全文件校验，禁止静默截短音频。

服务器175使用授权的NPU6/7，TP2，沿用固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
FSC选择队列使用4/5，通信端口分开。单次容器最长86400秒，观察86700秒，
由脱离会话的监督进程管理，不自动重试。模型不下载、不训练。

该实验是冻结通用基座的完整原生任务基线，不能替代SpeechBrain任务训练
权重的官方复测，也不能据其成绩宣称Jev跨任务提升。
