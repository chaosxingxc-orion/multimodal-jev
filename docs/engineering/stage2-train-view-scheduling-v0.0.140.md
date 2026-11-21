# Stage2 TRAIN 打分调度候选 v0.0.140

69 的 [v0.0.139](stage2-train-view-host69-v0.0.139.md) 每个完整批次用八张卡，但封包、训练集输入读取和推理分开串行。数值验收本身也先用单卡生成 HF 对照，再启动八个推理实例。本候选缩短可避免的等待，保留真实同机八卡数值验收、完整 TRAIN 审核和原采样规则。

- 控制端在父包就绪后并行准备验收包和全量打分包。全量包准备不调用模型。只有验收独立通过且全量包准备成功，才进入全量启动流程。
- 全量启动时执行一次完整独立准入。取得卡的租约后重新核对全包文件清单和哈希；目标端仍检验空闲卡、固定镜像、外部依赖和解包内容。用准确的容器句柄确认退出后释放卡，随后收集并独立审计结果。
- 打分端按原顺序预读下一批音频和特征，逐条保留哈希、缓存命中和张量检查。最多预读一个批次；模型输出顺序、去重、调度批大小和固定策略均沿用原合同。CPU 准备时间记为各工作项时长之和，和打分重叠，不能相加得到总耗时。单卡 HF 对照、启动和不足八组的尾批仍可能出现空闲。

沿用 `stage1-train-reference-chain.json` 的本地宏。v140 的新参数放在 `STAGE1_TRAIN_REFERENCE_CHAIN` 对象内：`package_preparation: {"cpu_workers": 2}`、`reference.input_prefetch: {"io_workers": 4, "ahead": 256}`、`reference_operations.single_admission_launch: true`、`reference_operations.admission_execution: {"io_workers": 4}`。这是八卡候选值；数据、设备、本地与远端目录仍由各次绑定提供。

本地 187 项定向检查通过，覆盖打分与预读确实重叠、样本映射和批次不变，以及获得租约后发现包变化即拒绝启动。冻结源码与运行时的隔离导入审核见 `/srv/multimodal-jev-xingchao/operations/stage2-train-view-host69-v0.0.140/runtime-audit.json`。本候选未派发 NPU，没有实际利用率或节省时长的结论。v139 已有独立八卡验收通过的收据，继续用其原包执行本轮打分。
