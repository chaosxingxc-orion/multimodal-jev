# 原始 Omni 首层 NPU 兼容性探针

CPU加载预检已通过；2026-11-11修复队列空闲识别后，NPU探针已派发，结果待验收。工具
`tools/jev/probe_omni_npu_layer.py`使用已保存的实际TRAIN6 actor首层入口，
保留原顺序[2,0,5,1,3,4]和六次micro1、三次micro2，共九次调用。
无音频重放、生成、标签读取、优化器或权重更新。

## 已验证部分

固定源码dbb9c0f，固定镜像，容器
`58edfb4a9aeb925b8649a77f76fb8368e95bef0975ffbeb2131557472bd2c7bd`，
2 CPU/8 GiB、600秒加30秒，无设备或网络。实际退出0无OOM，10.78秒。
见[启动记录](../../../omni-npu-layer-preflight-v1-launched.json)和
[独立预检核验](../../../omni-npu-layer-preflight-v1-verified.json)。

原config与第一权重shard在加载前后均重算哈希；实际加载623,120,640个
BF16参数，所有128专家的gate/up/down共384份张量与原checkpoint逐元素
核对，strict加载，无缺失/meta参数。转换代码与此前已核验的首层诊断加载体
一致。独立协调器核查固定源码、实际容器命令/资源/挂载以及全部九份prefix
张量哈希、形状、有效长度和有限值，未重复模型前向。

此预检执行了权重物化和输入校验，但没有执行CPU或NPU层前向，也没有验证
NPU可运行性、峰值HBM、梯度、FSDP或策略概率一致性。

## 待执行的NPU协议

- 保持原HF BF16 SDPA、原保存attention mask、position ids与cos/sin。
  不启用此前compact attention或FP32专家候选；默认grouped_mm路径不替换。
- 一张授权且空闲的物理NPU映射为逻辑npu:0；运行器拒绝多于或少于一张可见卡。
  协调器须独立核查实际设备映射及原队列完成情况，不能只依赖可见卡计数。
- 使用全部九份原始padded首层状态，形状为[1或2,1230,2048]。这是固定输入
  兼容性探针，不以缩短序列或换成随机输入绕开真实输入的算子问题。
- 对每次前向同步NPU，要求完整输出有限，保存输出与token mask的张量哈希、
  耗时、累计峰值allocated/reserved内存。算子异常直接保留终态及进度，不自动
  重试、不启用替代kernel或改精度。成功执行与数值保真验收明确分开。
- 预检和NPU执行均使用独立不可变源码与输出目录；部署前重新核对实际源码
  与预检一致。首次NPU容器采用有界运行时间，并保留实际容器ID与失败日志。

## 资源与后续

NPU4–7当前由冻结FSC/SLURP及其后续队列占用。本轮仅完成CPU预检，没有创建
NPU容器或新监督进程。后续应接在既有资源队列之后或确认既有4/5队列终态后
利用其释放资源，不中断基线、不越过已登记任务。

首层成功也仅支持原始算子和该层局部容量结论。之后仍需完整Thinker分片加载、
FSDP/HCCL实际初始化、真实156响应的完整音频前向与逐token概率比较。原raw
waveform探针不能直接作为vllm_hop协议下的on-policy证据；最终actor需使用
[已验证波形边界](../complex-tasks-deferred/sfc-omni-waveform-parity-v1.md)及真实rollout响应。

## 后续：已排队的一次NPU执行

`omni-npu-layer-v1`监督进程现已启动，身份与实际进程核验见
[排队记录](../../../omni-npu-layer-v1-queued.json)。协调器源码403d019，运行器继续
使用已预检的dbb9c0f快照。12项测试通过，覆盖原监督进程必须存活、原队列
和两个精确容器的成功终态、命令/应用终态变化、设备占用，以及模糊create
结果保留意图且不重复派发。

顺序为原FSC TEST parent → p1全部结束 → 驱动与Docker均确认NPU4/5空闲 →
仅物理NPU4的首层探针。使用npu45-diagnostic-resource.lock；现有FSC队列
没有共享该锁，其释放依靠精确队列/容器依赖检查保证。锁不对不遵守锁的
第三方任务提供原子排他保证。观察原队列最多350000秒，探针容器600秒加
30秒退出宽限，终态观察900秒。瞬时SSH失败继续观察同一对象；已有派发意图
或不明create/start结果不重放。失败保存证据；队列成功仅标记执行完成，
不自动发布数值/容量验收，不启动完整模型或训练。

启动后独立核对协调器PID/start/argv、122份队列源码和122份运行源码的Git
身份，确认phase=waiting_fsc_test_queue、尚无probe launch.json。真实NPU
兼容性与容量结论仍待实际执行和独立审计。

## 完成后的独立输出审计

`tools/jev/audit_omni_npu_layer.py`已准备；仅在原监督与应用成功终止后读取
输出，重新核对固定队列、CPU预检、原容器、启动配置和Git运行源码。
NPU identity须与已验收CPU预检完全一致，仅允许`preflight_only`从true变false。
逐一认证全部九份输入/输出文件哈希、旁路报告、BF16形状、有限值和原始mask。

按原顺序配对六条样本的micro1和micro2结果，分别报告有效token和完整padded
张量的相等性、不同元素数、最大/平均绝对差及RMS差；不隐藏padding差异。
容量汇总要求九次累计allocated/reserved峰值单调、allocated不超过reserved，
并保留前向加保存耗时。PyTorch allocator峰值不是整卡HBM占用，不能据此
推断完整模型可容纳。没有设置数值通过阈值，也不自动批准双卡派发。

宿主32项相关测试通过（2.91秒），包括真实形状九组合成张量的完整审计编排、
padding区别、文件损坏、遗漏/重复/错序、非有限值、mask漂移及累计峰值异常。
Docker/Git依赖在编排测试中显式模拟；这不是实际NPU执行或数值保真结果。

原单层任务完成后，在工程根目录使用有torch和SSH访问权限的宿主Python：

```bash
PYTHONPATH=src /usr/bin/python3 tools/jev/audit_omni_npu_layer.py \
  --root /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-npu-layer-v1 \
  --prefix /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-actor-prefix-train6-v1/run \
  --output /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-npu-layer-v1/diagnostic-audit.json
```

该工具不加载模型权重、不执行前向、不读取任务标签、不派发容器，已有审计
文件不会覆盖。跨CPU/NPU保真及完整actor概率验证仍须独立完成。

## 2026-11-11 队列修复及实际派发

FSC完整队列成功后，原监督进程停留在waiting_npu45_free。驱动对两卡返回
`No process in device.`，旧判断仅接受空字符串，因此误判忙碌。修复只接受
空字符串或该完整明确消息；包含PID、额外文字或错误的输出仍拒绝。相关16项
测试通过，新增原依赖截止时间已过则禁止派发的检查。

在核对原PID/start/argv、尚无容器派发意图后，以SIGTERM停止旧监督进程；
原文件、源码和记录全部保留。新协调器快照44cf24f写在queue-recovery-v1，
没有修改原preparation、queue-plan或探针运行源码。新进程继续使用原资源锁，
依赖等待截止限制为原started_unix+350000秒，不延长等待，也不自动重试。

实际容器`8deccddc76a19ae113362748c36e24a0b83fa26ea4a9303758607bd6b6623d1d`
已创建并启动，独立检查原镜像、命令、挂载、资源与物理NPU4映射通过。
见[交接与启动记录](../../../omni-npu-layer-queue-recovery-v1.json)。这只证明已派发；
九次前向、数值差异和容量仍须终态后独立审计，完整双卡初始化尚未派发。
