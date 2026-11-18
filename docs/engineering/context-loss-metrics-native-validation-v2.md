# 混合损失分项：真实 verl 固定镜像 CPU 验收

分项候选实现未改动。固定镜像52ba94e1中CPU容器3ba3490e准确docker wait返回0，
独立inspect确认exit0/nonOOM，16项测试全部通过，无跳过。361文件封包身份和收集文件hash重验。
证据：`context-loss-metrics-native-fixed-image-cpu-v2.json`。

## 本次实际覆盖

使用当前GRPO arithmetic-v2封包的src、固定原生verl core_algos SHA，以及实际
registered_loss函数体，比较原函数和新增分项函数。新测试不替代verl或损失函数：
原生GRPO/RLOO优势、原生clipped surrogate、当前ranking/binaryNLL/reference KL都真实执行。

合成8×60 logits，G16，7真实query及1个padding；刻意保留一组恒reward。
按4rank×2microsteps模拟分区累积和跨rank平均，检验分项汇总还原全局loss。
覆盖instruction/task、FP32/BF16 logits；前后loss、参数梯度、一次AdamW更新和全部
Adam状态逐元素精确相同。padding梯度为零，恒reward的真实query仍有辅助项/KL梯度。

测试在CPU单进程运行，未运行真实DDP通信、完整DataParallelPPOActor.update_policy、
Qwen音频前向、FSDP或NPU；不是模型训练收益或吞吐测量。生产包/队列保持。

## 首次失败与修正

v1容器56a4d6ad准确wait1、exit1/nonOOM：16项中12项在Adam默认foreach能力探测触发
NPU查询时失败。没有设备的CPU容器无法调用该接口。保留完整v1收据与日志，不认定loss
数学失败或运行中训练发生故障。

v2只将两份测试中的AdamW设置为foreach=False、fused=False，与正式配置相同；
未改镜像、未挂设备、未放宽数值断言、未改候选实现。实际原生验收随后通过。
此前8项本地验收对应的旧测试文件仍保存在v1封包中，原报告hash保持历史意义。

## 后续

当前运行不热替换日志。若完整简单任务selection审查支持新优化实验，将此日志候选
与选定的损失/算法配置一起版本化，完成受影响的真实actor更新验收后再发布。
不因CPU验证通过放行复杂训练。简单完整曲线、TEST和迁移收益仍待原队列完成。
