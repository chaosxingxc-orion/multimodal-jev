# 固定 verl 的 Ascend 概率计算路径

从已停止的固定镜像容器复制 `/opt/verl/verl/utils/torch_functional.py` 和
`device.py`，未启动进程或占用NPU。源码哈希及控制流验证见
[核查记录](verl-npu-logprob-source-audit-v1.json)。

`logprobs_from_logits`先按flash-attn导入可用性选择其cross entropy，否则按
torch_npu算子可用性选择NPU路径，最后才使用通用实现；不检查输入tensor的
device。这与此前CPU原生actor误入NPU算子并失败的证据相符。已执行原函数
AST的三种分派条件，使用仅提供shape/reshape/view的哨兵对象及记录调用的
叶函数；没有执行数值kernel。该验证不能证明实际运行时导入标志的取值。

NPU wrapper将logits及labels展平，直接调用
`torch_npu.npu_cross_entropy_loss(..., reduction="none")`，返回负loss并恢复
批次形状，没有显式FP32转换。固定padded actor在完整模型logits中截取
response位置，再在设备autocast上下文调用此dispatcher。实际算子累加精度、
autocast转换和返回dtype不能由这个wrapper推断，需要NPU执行时记录。

已有双rank CPU验证显式将低精度logits转FP32后做log-softmax/gather，属于
另一条概率计算路径。早期 `npu-audio-runtime-v1` 的9.54e-7交叉熵误差来自
ASR模型loss与项目 `response_log_probs` 的比较，并未直接调用verl NPU
dispatcher，不能当成该路径或完整Omni的已通过证据。

后续真实Omni概率运行应增加以下观测，仍保留原始dispatcher作为被测路径：

- 固定实际actor及torch_functional源码身份，记录两个availability标志及
  实际函数身份；不得启用CPU monkey patch或静默替换kernel。
- 记录进入dispatcher的logits/labels形状、device和dtype，以及返回dtype。
- 使用同一份response logits对比NPU输出与独立FP32 log-softmax/gather，
  不重新执行另一遍模型来混入MoE/微批变化；参考计算计入资源预算。
- 与vLLM raw logprob比较是另一个检查：同logits概率归一化通过，并不代表
  actor与rollout的模型前向一致。原始old logprob不得被参考值替换。

此处尚未指定正式NPU数值阈值，也未派发独立算子探针。需要将观测接入
完整真实响应运行器并预先固定容差；全模型FSDP初始化及单层NPU前置依赖
继续遵循原队列。
