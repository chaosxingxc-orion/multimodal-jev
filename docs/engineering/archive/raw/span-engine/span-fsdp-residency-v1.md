# FSDP常驻训练的卸载与恢复准备

2026-11-12 UTC。直接读取固定镜像中的`/opt/verl/verl/utils/fsdp_utils.py`，SHA256为`5dff5af2125de68d3cb15deb519f30a47e603a36e8f75d6a250a12c9e9941840`，与已有源码审计一致。FSDP1接口按handle搬移flat_param并重新绑定_local_shard；不是普通model.to。优化器默认加载函数会把所有张量状态搬到指定设备，包括step，因此不能假定保持非capturable Adam原来的逐张量设备布局。

新增`SpanResidency`边界，调用方注入固定版本FSDP模型搬移函数，优化器逐张量记录原始device后恢复。卸载前要求梯度清空，并同步设备；卸载和恢复后核对参数对象ID、优化器参数引用、dtype、逐张量值与设备位置。状态改变后失败不可自动重试。为验证精确性，本版本保留CPU参数/状态副本，增加主机内存占用；它是验证实现，尚非大模型生产内存优化方案。

固定镜像CPU容器`f4ea91fe727c5e51e915a289c01cdfc2ea7480b6c11e501ddfc8884c775834cc`实际退出0，7项测试通过：恢复后下一步与未中断分支一致、未清梯度拒绝、参数/动量/dtype/引用损坏拒绝、重复状态转换拒绝。CPU测试的模型搬移回调为空，因此没有验证真实设备搬移或FSDP内部布局；见`span-residency-cpu-v1-verified.json`。

已扩展原有四卡FSDP小模型探针，新增显式`--residency-roundtrip`选项，默认历史行为不变。新路径在第2步恢复后执行CPU卸载和设备恢复，再比较第3步输出、FP32 master、Adam及BF16导出与未中断分支，记录卸载前后allocated内存。准备产物位于`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-fsdp-residency-npu4-prepared-v1`，源码与镜像已固定，挂载pins通过；尚未创建NPU容器，也未新增后台队列。

探针同时保留参考分支和HCCL上下文，内存数字不代表完整训练卸载后的空闲容量。模型buffer、通信分配和vLLM所需空间仍需真实音频实验核验；此边界不授予并发使用设备的权限。现有四卡队列继续执行，不能仅因CPU测试通过就宣称常驻训推已打通。
