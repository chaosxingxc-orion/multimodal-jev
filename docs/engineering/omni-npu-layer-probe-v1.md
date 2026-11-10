# 原始 Omni 首层 NPU 兼容性探针

CPU加载预检已通过；NPU前向尚未派发。工具
`tools/jev/probe_omni_npu_layer.py`使用已保存的实际TRAIN6 actor首层入口，
保留原顺序[2,0,5,1,3,4]和六次micro1、三次micro2，共九次调用。
无音频重放、生成、标签读取、优化器或权重更新。

## 已验证部分

固定源码dbb9c0f，固定镜像，容器
`58edfb4a9aeb925b8649a77f76fb8368e95bef0975ffbeb2131557472bd2c7bd`，
2 CPU/8 GiB、600秒加30秒，无设备或网络。实际退出0无OOM，10.78秒。
见[启动记录](omni-npu-layer-preflight-v1-launched.json)和
[独立预检核验](omni-npu-layer-preflight-v1-verified.json)。

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
[已验证波形边界](sfc-omni-waveform-parity-v1.md)及真实rollout响应。
