# Stage1 每卡64 / micro16 容量验收

沿用固定镜像、全量TRAIN计划、原始初始化、BF16、LoRA和新前缀缓存，DP4卡4–7，每卡有效batch64，micro16，累积4，global256；只派发两步验收，没有恢复长训练。runtime源文件与已通过76项测试的v120固定镜像CPU证据一致。

实测第一次更新前4rank均NPU OOM，语言骨干RMSNorm前向：已分配57.25–57.83GiB、预留59.28–59.39GiB，剩余1.06–8.74MiB，申请178/356MiB失败。完成更新0，无loss/吞吐成绩。验收进程已退出，现场确认卡4–7释放。

当前无激活重计算的基座：micro8已正式跑通，micro16和32均失败；micro64未NPU验证。不能从LoRA参数量推断激活容量。下一候选应优先验证可配置语言层激活重计算，而非继续盲目扩大microbatch。旧正式任务继续暂停，step32 checkpoint保留。

证据：/srv/multimodal-jev-xingchao/operations/context-stage1-local64-micro16-v0.0.121/capacity-summary.json
