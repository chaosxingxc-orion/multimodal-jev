# 直接actor调用后的FSDP根模块存储释放

固定verl的`FSDPWorker.compute_log_prob`在调用底层actor并搬回结果后，对
world_size>1的FSDP1执行`actor_module._handle.reshard(True)`。项目直接
调用`DataParallelPPOActor.compute_log_prob`会绕过这个步骤。

已在固定镜像实际双rank CPU/Gloo验证：确定性BF16两层小模型，root有18个
参数，子单元30个参数，各rank使用不同输入。两次无梯度前向都与未分片
参考逐元素一致；每次结束后root保留36字节完整参数存储及18字节本地分片，
子单元完整存储已释放。显式reshard后root完整存储变为0、本地分片不变。
这是缓存存储保留，不表示本地参数shape仍为完整参数，也不代表每次调用
都会无界增长。v1容器6.63秒、退出0、无OOM，见[v1核验](fsdp-forward-storage-cpu-v1-verified.json)。

`omni_actor_forward.compute_log_probs`现于actor返回后、读取概率前补齐
FSDP1根handle的reshard；仅适用于实际FSDP及其process group多于一个rank，
无handle时跳过。未修改模型数值、响应token或old logprob；未实现FSDP2。
失败前向不会被重试，也未增加异常后的通信操作。

v2在真实双rank FSDP上通过项目适配器执行第三次前向，两个rank均验证概率
与参考完全相同、root完整存储为0。使用合成actor包装和合成模型，未调用
真正verl actor或加载Omni，不能替代NPU概率验证。容器退出0、无OOM，
4CPU/8GiB、无网络/设备，127份快照与Git逐文件核对，见
[v2核验](fsdp-forward-storage-cpu-v2-verified.json)。宿主相关测试36通过、
1项因缺verl跳过。v2首次准备因有未提交验收文件在派发前被拒绝；提交后
完成一次实际create/start，没有失败容器或自动重试。

完整Omni meta布局的root单元为1.431GiB，因此这项生命周期差异值得在真实
NPU actor中保留检查；此数值是结构估算，不是本次实测HBM。两卡初始化
清单只执行初始化/完整性验证，不执行actor前向，原快照不受本修订影响。
后续概率运行须使用包含该修订的新源码快照，仍等待单层NPU与双卡初始化
验收、完整真实采样及预登记数值门槛。
