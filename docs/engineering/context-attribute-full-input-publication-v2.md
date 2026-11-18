# 全量语音Jev候选输入与日程发布

原CPU容器edf4509f成功退出，40513录音、81026个Direct/Q-input请求已完整处理并收集。
原v1审核在音频帧数ceil/floor差异处失败，现场保留；预先封存的v2按实际单音频processor
规则核算帧数，并逐条重建int32 mask、匹配字节hash。全量v2审核通过。

原finalizer792402于Unix1794731378.142096完成并退出。发布路径：
`.local/decision-objective-research/context-attribute-full-input-publication-v2/PUBLICATION.json`。
发布SHA256：`e68c639211776ffaaae65e2f60a82764db5647b76d0eae7a9cd0d121efbfe657`。

独立复读13个文件并逐项核对哈希、完整文件集合、原日程slots和卡间索引：
Direct、Q-input、Q-credit三组均为634步、每步128 slots、DP4每卡32 slots。
每轮40513条录音各出现一次，共81026次真实曝光；两轮共126个padding slots，全部权重0。
三组完整batches精确相同，Q-input与Q-credit长度身份相同，三组checkpoint契约身份各异。

这固定了可比较的全量输入和数据曝光；未新增音频模型前向、NPU验收或训练。
发布保持training_dispatch_allowed=false。简单任务完整选择集确认收益及复杂任务自身
有限NPU验收仍是训练前提；本次没有修改当前GRPO/RLOO及其后继评测队列。

证据：
- [完整发布独立核对](context-attribute-full-input-publication-independent-v2.json)
- [实际输入长度统计](context-attribute-full-input-geometry-v2.json)
- [原CPU任务终态与收集](../research/context-attribute-population-fixed-image-cpu-v1.json)
