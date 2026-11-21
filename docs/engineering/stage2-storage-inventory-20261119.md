# Stage2 v0.0.93 产物体积根因

2026-11-19 现场通过固定镜像只读挂载已退出训练容器盘点，收集范围 14,642 文件，
33,961,994,190 bytes（31.63 GiB），不含 tar header。此前 18 GiB 是传输中的部分大小。
普通 SSH 用户无法遍历全部 policy/adapter 目录，必须使用与 collector 相同的只读可见范围，
否则会低估。未删除或修改待独立审核的冻结运行产物。

| 类别 | GiB | 原因 |
| --- | ---: | --- |
| policy execution | 8.150 | 634 步各保留约 13.8 MB LoRA/head |
| 临时 execution | 8.149 | 634 步另外保留组装权重 |
| policy processor | 6.751 | 每步 copytree processor，634 份 |
| 两 rank update JSON | 1.748 | 各 rank 保存包含 rollout 的完整记录 |
| optimizer checkpoint | 0.773 | 10 个预登记点×2 rank，含恢复状态 |
| native adapter | 0.689 | 634 份服务加载用 LoRA 格式 |
| batch records | 0.375 | 634 批数据记录 |
| pool journal | 0.300 | 请求和响应账本 |
| policy manifest | 0.065 | 每步身份清单 |
| 其余 | 约 4.63 | 包括逐步 groups 张量、激活记录和配置等 |

两个 rank 的 checkpoint-scope.json 均明确 base_model_saved=false；保存白名单为 LoRA/head。
每个完整 policy execution 13,803,531 bytes，每个 native adapter 权重 1,163,208 bytes；
每个预登记 checkpoint 的两rank恢复状态合计约83 MB。规模来自版本数量和重复封包，
不是单份完整基座。共享音频、特征和reference缓存位于运行目录外，不属于collector递归目录。

根因源码：`context_checkpoint_policy.export_checkpoint_policy` 对每一步 copytree processor；
trainer 每次更新同时写 execution、policy、native adapter；collector 按 layout 整目录递归，
没有将临时激活版本与持久评测/恢复版本分开。

目标契约：共享模型、processor、音频、特征、reference按内容身份引用；本地缓存跨任务复用。
临时训推权重仅保留仍被消费者引用的版本，完成ACK后由固定保留脚本清理；
预登记评测点及必要恢复点持久保留；轨迹每批保存一份，rank仅引用并记录自身指标。
收集器消费显式产物清单，将评测最小资产和完整审计归档分离；全量审计完成后，按清单
删除中间权重和冗余传输tar，保护后继评测依赖。未将上述目标描述为已经上线。

Ceph：共享数据/特征应归档复用，但175/181热写限制仍有效；通过已验收客户端归档封闭产物，
不直接把训练写目录搬到Ceph。后续实现发布新版本，不原地修改当前冻结审计契约。

证据：`/srv/multimodal-jev-xingchao/operations/context-stage2-composed-v0.0.93/storage-inventory.json`
与同目录 `storage-detail.json`。
