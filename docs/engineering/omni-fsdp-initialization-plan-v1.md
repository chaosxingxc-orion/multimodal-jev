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

即便初始化通过，也只证明分片和同步完整；真实音频前向、完整采样概率、
参数更新、导出/恢复及任务提升/跨任务泛化仍是独立验证项。当前权重训练
仍遵循冻结基线→教师提示优化→native/Jev比较的执行优先级。
