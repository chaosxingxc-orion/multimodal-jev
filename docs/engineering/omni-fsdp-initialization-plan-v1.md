# 两卡 Omni FSDP 初始化探针：已准备，未派发

目标是在完整 CPU 加载已通过的基础上，验证两 rank 的真实 FSDP1 初始化、
权重同步和非持久化 buffer。当前代码已准备，尚无 HCCL/FSDP 初始化或
NPU 内存测量结果；不执行前向、优化器、采样或训练。

## 初始化回调

固定镜像 PyTorch `_materialize_with_param_init_fn` 遍历所有尚未嵌套包裹的
子模块，逐个调用回调。verl 默认 `init_fn` 对非零 rank 执行
`to_empty(recurse=False)`，会同时清空实 buffer，随后依赖 rank0 广播恢复。
这本身不证明默认路径错误，但无法保留我们准备阶段已认证的 buffer。

`MetaParameterMaterializer` 对每个模块仅处理本层状态：

- 只为冻结 BF16 meta 参数分配目标设备存储，不初始化随机权重；这些值在
  `sync_module_states` 完成之前不能参与前向。
- 对真实 FP32 buffer 仅迁移设备，保留精度和值。拒绝 meta、非有限或
  非 FP32 buffer；不调用全模型 `.to(dtype)`。
- 保留仍存活的共享参数引用；跨已释放 FSDP 单元的别名直接拒绝。回调弱引用
  新分配参数，避免 FSDP 替换参数后仍保留整个未分片分配。
- 不对已经物化的同设备参数重新执行 empty，避免二次回调破坏同步后的值。

固定镜像 19 项测试通过、0 跳过，3.459 秒，实际调用 PyTorch 的模块遍历
函数；覆盖 buffer 精确保留、别名、非递归行为和分配释放。容器
`6eac0634e9e4fa588bfbfc6b5ee83fefefee944e7e7bafce0bcf927faeaf787b`，
源码 `e7af067242c85a2b942a844bdc88d8455e000642`，2 CPU/8 GiB，无网络/设备，
退出 0、无 OOM。外部核对127份源码、容器配置和JUnit，见
[固定镜像测试核验](omni-meta-materializer-tests-v1-verified.json)。这是CPU契约
测试，不包含 FSDP 实例或任何通信。

## 完整探针流程

`tools/jev/probe_omni_fsdp_initialization.py` 由单机双进程 torchrun 启动，
要求 world size=2、rank=local rank、恰好两张可见 NPU。物理设备映射由外部
启动器验证，程序内 logical 0/1 不能独立证明物理4/5授权。

1. 认证完整 CPU preflight acceptance、模型 config、产物；rank0 重新校验
   全部模型文件 SHA。实际 HF 源码必须匹配固定版本。
2. rank0 实际加载完整 CPU Thinker并记录全部参数原始字节哈希；rank1 仅
   构造 BF16 meta 参数和真实 FP32 buffer。两边核验全部 buffer。
3. 使用实际 verl wrap policy 的三个显式层类，FSDP1 FULL_SHARD，
   BF16 param、FP32 reduce/buffer、sync_module_states=True、
   use_orig_params=False、limit_all_gathers=True、无forward prefetch。
4. 要求108个唯一FSDP单元。两rank按相同顺序逐单元执行
   `summon_full_params(recurse=False,writeback=False,rank0_only=False)`，
   排除嵌套FSDP子单元，只哈希本单元负责的参数。全部键、形状、dtype、
   元素原始字节必须与rank0加载后的参考完全一致。
5. 核对全部31,719,205,488个参数和四个buffer、无梯度/无训练状态；保存
   每rank报告、原始参数哈希、初始化后allocated/reserved及累计峰值。
   异常终止且不重试，不把单rank通过当作双rank通过。

逐单元聚合是为了避免将全部59.1GiB权重同时聚合到单张64GiB设备。实际
是否满足容量仍待运行；FSDP启动、跨卡通信、torch-npu支持不能由静态代码
证明。宿主另有2项测试检查单元参数不重叠覆盖和字节哈希的区分性，合计
21项相关测试通过；完整探针未在固定镜像实际执行。

## 派发前约束及后续

先等待原FSC评测及已排队单层探针完成，并核验其实际结果。两卡探针暂未
入队，不能抢占NPU4/5或重跑已有评测。派发时固定源码快照、完整镜像SHA、
参数、精确设备映射、资源锁、容器ID及1800秒通信超时，保留一次性启动记录。
单进程CPU加载已测得118.326GiB RSS，双rank初始化不能沿用128GiB容器上限；
准备按16 CPU/256GiB总容器容量验证，最终启动前核查实际服务器可用资源。
HCCL通信所需IPC、网络和设备映射须单独核查，不能照搬无设备CPU容器配置。

已生成具体准备清单：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/omni-fsdp-initialization-v1/preparation.json`，
SHA256 `24ee2d400d01058931aa49e2e70da408d7a4ea1756d71b0261a0ff5383c6fceb`。
126份Python源码已逐文件与Git `98d751c0cb64c21d546a4212f9d98119cfb7054e`
核对。完整CPU加载验收SHA及原单层探针的准备/队列SHA均已绑定。

具体命令使用`docker create`后核查再单独start，保留镜像默认entrypoint；
NPU4/5加三个管理设备，host网络、private IPC/8GiB共享内存，16CPU、
256GiB内存及同值memory-swap上限。torchrun单机双进程、static rendezvous
`127.0.0.1:29581`、`max-restarts=0`，HCCL/Gloo使用bond0，HCCL起始端口
64400；总期限3600秒、TERM后30秒宽限、外部观察3900秒。
运行前必须获得同一`npu45-diagnostic-resource.lock`，重新检查设备实际进程、
至少320GiB主机可用内存、端口及排队任务，核验create后的实际Docker配置。

当前只读核查发现主机可用内存约1.85TiB，bond0地址及候选端口符合准备条件；
这不预留资源，也不能证明双rank HCCL通信成功。尚未创建容器或启动监督队列。
见[准备清单核验](omni-fsdp-initialization-prepared-v1.json)。

即便初始化通过，也只证明分片和同步完整；真实音频前向、完整采样概率、
参数更新、导出/恢复及任务提升/跨任务泛化仍是独立验证项。当前权重训练
仍遵循冻结基线→教师提示优化→native/Jev比较的执行优先级。

## 实际双rank CPU/Gloo逐单元恢复验证

已新增 `probe_fsdp_unit_gather.py`，在固定镜像执行真实两进程 FSDP1
FULL_SHARD，使用三单元、99个确定性BF16参数、三个FP32非持久化buffer。
各单元参数数目21/39/39均为奇数，因此两rank的本地分片包含padding。
两边从实际CPU参数构造，明确 `sync_module_states=False`；这条CPU路径
不能验证NPU/HCCL、meta-rank广播或原始Omni模型。

使用与完整Omni探针完全相同的 `owned_parameters`、`logical_path` 和
`tensor_identity`，实际逐单元执行 `summon_full_params(recurse=False)`。
全部参数恢复后哈希与原值相同，退出上下文后本地分片不变；跨rank比较
一致，FP32 buffer不变。外部独立重建所有7个参数张量和3个buffer，以及
两rank各自的三份padding分片，全部哈希匹配。

成功运行v3容器 `4b908b84b3a8a332e3dfaa426a8b0414a8bfc8abea901071116ced6a19e65741`，
源码 `5e5411760a2f04b555f7e6c9f4efe3c3e7f8971b`，固定镜像4CPU/8GiB，
无网络/设备，退出0、无OOM。rank0/1探针耗时约0.157/0.034秒，不含
torchrun及导入启动时间。见[外部核验](fsdp-unit-gather-cpu-v3-verified.json)。

两次失败原样保留，分别修正后才启动新版本，无自动重试：

- [v1](fsdp-unit-gather-cpu-v1-failed.json)：`--standalone`在无网络容器中
  公布无法解析的容器hostname，只有launcher、没有rank worker。核实进程
  后手动停止，退出143。v2改显式static rendezvous `127.0.0.1:29571`，
  `GLOO_SOCKET_IFNAME=lo`，不开放外部网络。
- [v2](fsdp-unit-gather-cpu-v2-failed.json)：两个rank真实执行FSDP后，
  在恢复分片时失败，报storage不可resize。原hash函数的`.cpu().contiguous()`
  在CPU上可能仍指向FSDP原storage；`.numpy()`使原storage变成不可resize。
  修复为先独立CPU复制再转换NumPy，不改FSDP实现或参数数值。新增回归测试
  检查hash后原storage仍可resize，宿主相关测试现为22项通过。

后续完整NPU探针也采用修正后的hash函数。CPU双rank通过不允许跳过原
单层NPU探针及其外部核验，也没有新NPU运行入队。

## 初始化完成后的独立验收入口

`tools/jev/audit_omni_fsdp_initialization.py`已准备，不派发任务。它要求固定
preparation、实际成功退出且未重启/OOM的原容器、两个rank成功终态；核对
NPU4/5及管理设备、16CPU/256GiB/8GiB shm、host网络/private IPC、默认
entrypoint、命令、挂载与显式环境，以及全部快照文件与Git来源。

输出检查绑定已接受的CPU preflight和布局，核对1407份原始参数身份、两rank
全部参数哈希及四个FP32 buffer；从完整参数名称的最长模块前缀独立推导
108单元的归属和参数量，拒绝漏项、重复单元及总数相同但归属错误的报告。
同时检查零forward/零更新、初始和峰值allocator计数。保存哈希相等仍依赖
已认证运行器的测量，不是独立重新加载权重或真实前向验证；派发前置条件
审查也不由此结果代替。

宿主46项相关测试通过（1.90秒），含34项新验收测试；合成rank结果覆盖
缺rank、权重/buffer变化、错误单元归属、设备越界、额外设备、资源/命令
变化等拒绝路径。另读取实际CPU结构与布局验证108单元/1407张量/31719205488
参数的归属；该检查的rank哈希及内存数值为合成值，不是NPU执行。

实际任务成功完成后使用：

```bash
PYTHONPATH=src .venv/bin/python tools/jev/audit_omni_fsdp_initialization.py \
  --root /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-fsdp-initialization-v1 \
  --layout /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-fsdp-layout-v4/run/result.json \
  --output /mnt/cephfs/xingchao/experiments/multimodal-jev/omni-fsdp-initialization-v1/diagnostic-audit.json
```

目前完整初始化尚未创建容器，以上CLI尚未用于真实NPU验收。
