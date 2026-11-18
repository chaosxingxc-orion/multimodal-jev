# 双信号 RL 整合包：依赖修复与验收范围

本版本继承统一固定镜像和原训练器的更新、保存、激活、恢复流程，接入组件级
RLOO/GRPO-floor、批量实际 logits 传输及共享 reference 缓存。尚未准入 NPU 训练。

封包通过 `tools.reports.prepare_context_dual_training_candidate` 和
`configs/portable/operations/context-dual-training-prepare.json` 生成，部署路径由本地宏绑定。
v3 的 PREPARED.json SHA256 为
`03ed38f0046c75b2094be077fe582aedc1c3ae0bb81ebf6f759cdc3d0c0d2c2d`，398 个文件；
source manifest 为 `00cc1e05eef7577887827bf7c6a82051f931ccd86f620b91f0aed35be959c603`。
基座模型只外部引用，LoRA/head 保存白名单包含在包中。

## 实际发现与修复

固定镜像 CPU 容器 `d7dd6b711c2541c8cc1c2b297e8a36ca562b5480eefee6950b3f6e2335c51687`
对 v2 包执行 8 项检查，7 通过、1 失败。独立进程加载实际封包时发现继承的
`context_switch_rollout.py` 已被旧实验改为共享容器设备校验入口，与新批量 owner
依赖的原源码 SHA 不一致。先前工作目录中的模块测试没有覆盖这个继承差异。
失败证据保留在 `context-dual-package-fixed-image-cpu-v1.json`。

v3 封包显式复制锁定的原始 owner，并在封存前校验其 source transform。
共享容器的设备校验通过新版 dual owner 的显式 `shared_scope` 参数注入，
原校验器也锁定 SHA。service 仅在配置声明 process_device_scope 时选择该分支；
有限单卡推理 probe 继续使用独占设备校验。实际 CANN 设备数量及逻辑到物理映射
仍由原 validator 检查，不能仅凭环境变量声明设备隔离成功。

## 验证方式

`tests/test_context_dual_packaged_runtime.py` 检查完整包 hash、配置、源码 manifest
和本包 pins，然后使用 Python `-I -B` 隔离进程，只加入该包的源码路径。
解析并加载 trainer 中的 imports，实际派生 verl actor、service 和两种 scope 的
owner，并导入 vLLM worker；核验已加载项目模块没有回退到工作目录源码。
此检查不会调用 trainer 的 NPU loop 或构建真实 Qwen 引擎。

本地 30 项 trainer、设备约束和 LoRA guard 测试通过。修订版固定镜像检查通过
`context-dual-package-cpu-tests-v3.json` 调度，8 项全部通过、零跳过，退出码 0、非 OOM。
容器 `6116a543dbb43ffc98f209cb1e191b17e0882905926dc2193c10585b96ba66b8` 已准确 wait
并独立收集，报告为 `context-dual-package-fixed-image-cpu-v3.json`。
CPU 测试 v2 的本地宏误改了父目录，远程准备失败、没有创建容器；v3 使用新 tag
和正确父目录，保留此前记录，没有把失败尝试记成通过。

剩余放行条件是实际音频 serial/batch 对齐、DP4 连续两步及新进程重放、全 TRAIN
排序分布数值覆盖和整体吞吐验收。不凭 CPU 导入成功宣称新算法有效或启动全量三臂训练。

最新用户授权允许在效率收益得到实际验证后停止旧训练、切换新版，无需等待原队列
全部完成或再次请示。当前尚无新版 NPU 吞吐证据；切换前须处理原调度器，保留有效
LoRA/head 恢复点与已完成结果。旧 probe 操作脚本的整链等待条件仍待适配，不能将
这份授权记录当作已经具备自动切换功能。
