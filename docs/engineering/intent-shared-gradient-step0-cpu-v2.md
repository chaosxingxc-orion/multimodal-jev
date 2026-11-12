# 共享梯度诊断的内存预算修订

2026-11-12 UTC。v1实际退出137且Docker OOMKilled=true后，检查175主机
约1.9TiB可用内存；v2容器内存上限改为128GiB，CPU仍4核，无NPU设备
映射。不是因等待超时重启，旧失败证据保留。

保持相同step0模型、BF16前向/参数梯度、FP32跨请求累加、FP64几何汇总、
日程batch1/6共16条来源以及原始过程/Final权重。新增前向与两次反向阶段
进度，逐条完成后释放图引用；没有改用更短录音或缩小任务范围。

实际容器`0076de75b8234b7ea7457b15be71533f0c1b281ddd3e1058520204c9e702b0e6`
启动检查为running、镜像匹配、memory limit137438953472字节、Devices空。
结果尚待验收，不能声称共享梯度协同/冲突已测出。下一次普通检查不早于
09:30 UTC，保持原容器，不因观察超时重建。

配置/源码/资源观察及v1实际终态见共享目录
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-shared-gradient-step0-cpu-v2`；
启动记录见同名launch JSON。CPU结果仍不能替代NPU FSDP梯度一致性验证。
