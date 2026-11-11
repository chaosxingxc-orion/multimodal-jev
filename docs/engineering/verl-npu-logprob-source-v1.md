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

## 已实现的无梯度观测接口

`omni_logprob_observer.observe_native_logprobs`在隔离进程内临时包装actor
导入的原生dispatcher，核对源码SHA、函数身份和实际availability标志，要求
选择Ascend路径。拒绝嵌套observer、CPU替代函数、训练梯度模式和非NPU输入。
退出或异常时恢复原绑定，不改安装包文件；调用方须保证没有并发actor线程。

每次调用先从原始logits逐batch/sequence分块复制到CPU，再以FP32计算
log-softmax/gather参考；随后原函数只调用一次，原输出对象直接返回。
默认每块16个词表行，不在设备上reshape复制完整非连续response张量。
保留全部位置的token IDs、原生/参考logprob、形状、device、dtype与差异。
这包括padding，不能直接把全位置最大误差当作有效响应验收；运行器还须
将调用序号绑定到真实microbatch身份，再用认证response mask筛选。

合法数值差异只记录，不裁剪、不替换原概率、不调整容差；无效数值或原生
异常直接失败，不重试。CPU拷贝、同步及参考计算增加开销，不用于报告未插桩
吞吐。该接口未接入正在运行的冻结TEST，也未执行实际NPU kernel。

宿主测试覆盖非连续2D/3D FP16/BF16/FP32输入、独立cross entropy参考、
近确定token舍入差异、输出对象保持、单次调用、异常恢复、错误分派及数值
拒绝。与CPU概率适配和响应导出测试合计53通过、1项因宿主缺verl跳过
（2.65秒）；新observer测试全部通过。尚未在固定镜像或NPU实际执行。

后续固定镜像验证完成：源码9d4af5b，实际容器
`411e380e9b4bbd4543f7992a54475eb24e4c8a67e566d0b5cc0035f39b6a335f`，
2CPU/8GiB、无网络/设备，30项通过、0跳过，12.96秒，退出0且无OOM。
新增实际verl绑定检查确认原函数源码SHA、flash不可用/NPU算子可用的标志，
CPU输入在kernel调用前拒绝，正常/异常退出均恢复原函数。数值用例仍是
CPU合成张量，没有执行NPU cross entropy、音频或模型前向。逐文件源码、
实际容器配置和JUnit核验见
[固定镜像验收](omni-logprob-observer-tests-v1-verified.json)。

## 微批观测绑定补充

`omni_probability_capture.recompute_native_observed` 在一次 no-grad actor 调用周围
使用原生 observer，恢复函数绑定后才导出结果。`bind_observations` 要求静态连续
微批每批恰好一次归一化调用，逐项核对完整 padded labels、调用顺序、shape、
原生 NPU 设备，以及有效 token 上观测值与 actor 返回值的精确一致性。
导出原生与同 logits FP32 参考两份完整响应记录，附各微批的响应身份、原记录
哈希和 mask；padding 留在原始观测中，但不进入有效 token 比较。

此绑定不设置数值通过阈值，不修复概率、不重试，也不证明运行环境或音频身份。
具有相同 tokens/概率的不同音频不能仅靠输出区分，调用方仍须认证实际输入及
隔离的同一次 forward。完整 NPU runner、真实采样与多 rank 集成仍待验证。
本地 CPU 合成测试覆盖不同微批大小、缺失/额外调用、错序、错 token、错 mask、
返回概率不一致与绑定恢复顺序；不能用这些测试宣称原生 NPU 数值已经通过。

随后在固定镜像、无设备/无网络的 2 CPU/8 GiB 容器执行相关测试：64 项通过、
零跳过，含真实 verl 包绑定和 DataProto 检查。核验134份源码与JUnit、容器
退出0且无OOM；见[验收记录](omni-observation-binding-tests-v1-verified.json)。
微批绑定测试仍使用合成概率，尚未执行真实 Omni NPU 观测。

`omni_probability_shards.recompute_observed_shard` 复用原有完整名单与 rank 分配
校验，每 rank 仅调用一次 observed recomputation，返回原生和 FP32 参考两份
分片 envelope，以及原始观测证据。两份概率各自用 `merge_shards` 要求全部
rank/响应后恢复完整顺序。合并不认证原始 observer 内容，真实运行身份和
隔离进程仍由 runner 负责。此接口不启动分布式进程、不重采样，也不设置阈值。
固定镜像相关86项测试通过、零跳过，含真实DataProto重排后分配与两份概率
合并；[验收记录](omni-observed-shards-tests-v1-verified.json)保留容器和源码身份。
此处 observed forward 是合成替身，不是双 rank NPU 运行证据。

观测器同时固定 `dp_actor.py` SHA256
`207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce`，
在替换函数绑定前核验，并将其写入报告、在微批绑定时复核。该版本静态路径
使用 `data.split(micro_batch_size)`，依次调用 `_forward_micro_batch` 后按行
concat；padded/unfused 分支每批调用一次 `logprobs_from_logits`。
源码变化必须重新核查顺序假设，不能仅因归一化函数未变就接受升级后的 actor。
磁盘哈希并不认证任意运行时 monkey patch；隔离进程与实际 actor 身份仍由
完整 runner 认证。NPU完整前向还依赖单层和FSDP初始化实际验收以及采样完成。
