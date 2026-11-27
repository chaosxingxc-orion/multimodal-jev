# 台账执行配置

操作入口：[根目录 run_experiment.py](../../run_experiment.py)。每条可执行的台账记录对应一个
`JEV-*.json`，由 Git 提交固定。先登记协议、评测覆盖和资源，再将台账与执行配置一起提交。

- `protocol` 必须与台账的协议链接一致。`state_root` 指向调度机的本地持久盘。
- `stages` 按 data → train → evaluate 执行；每阶段可登记多项。带 `run` 的项以参数列表
  调用 Git 中已有的 `tools.*` 模块；无 `run` 的项只接收已有审核收据，适合接手外部运行。
  `inputs` 可锁定后端依赖，`config` 引用后端的 JSON 模板，驱动保存本次绑定。可指定
  `python` 使用审计环境；`dispatch_guards` 登记已有运行或发布目录，路径已存在时等待收据。
  后端继续负责冻结、资源租约、部署、恢复和独立审核。
- 复用既有后端模板时，`operation_bindings` 将其输入/输出宏绑定到本实验目录，禁止本地同名覆盖。某个输入文件的摘要须在上游产出后才能确定时，步骤可声明 `input_sha256_macros`（宏名 → 本步骤 `inputs` 下标），由入口把已读取的实际 SHA-256 注入后端配置；旧实验的摘要不得手填。
- `checks` 指向独立审核产物；用 SHA-256 或关键字段校验证据。路径用本地宏，`cite` 用于
  生成可共享的报告索引。终态与独立审核分列；进程退出码为 0 但缺少审核收据，依然不能进入下游。
- `panels` 登记同人口、候选与计分规则下的结果；第一臂是目标，第二臂是主对照。单列
  各数据集与判决方式；没有后端或审核输入的必跑项写成 `pending`，不得自动跳过。
- 本地私有设置使用 `.local/config/macros.json` 或 `--local`。当前 TPO 接续需要
  `TPO_EVIDENCE_ROOT`、原生评测两组互不重叠的设备列表（`TPO_EVAL_SLURP_DEVICES`、
  `TPO_EVAL_MINDS_DEVICES`）、`TPO_EVAL_PYTHON` 和 `EXPERIMENT_PACKAGE_STATE_ROOT`。
  阶段脚本从选中的 Git 提交执行，阶段产物写到宏绑定的持久目录。

新 TPO 实验用 `./run_experiment.py plan --experiment <ID>` 查看阶段分工，确认后用同一入口的
`run` 执行。封包项登记 `config` 并调用 `tools.reports.prepare_tpo_windows`，传同一模板与
`${EXPERIMENT_LOCAL_CONFIG}`。正式全量单阶段训练用 `tools.reports.run_full_joint_tpo` 接续，
参数仍是这份模板和本地宏，由驱动等待收集与独立审核。数据和训练分阶段登记；训练项的 `checks`
包含审核控制器的终态和 `formal-audit.json`。其他联合 TPO 使用现有容器后端及其审核收据。

181 的 TPO 布局：本地宏 `FULL_TPO_OFFLINE_REFERENCE_DEVICES` 绑定 8 张卡，必须等于
`FULL_TPO_TRAINING_DEVICES`（6 卡）和 `FULL_TPO_INFERENCE_DEVICES`（2 卡）的并集。
联合对照模板以 `TPO_OFFLINE_REFERENCE_DEVICES` 绑定离线卡，训练卡由父配置绑定。
离线阶段每卡一个实例，按训练 rank 与步号分片；训练阶段仍是 DP6 加两路 TP1。
`plan` 和 `status` 显示配置分工，运行日志落同一计划，生成的分片清单须通过完整性校验才能训练。
这是新封包的规则，已派发的 201 和 204 继续读取各自的冻结配置与收据。

新实验可将 `tools.reports.prepare_stage2_dataset --action release` 登记为难例视图的数据步骤，
并以 `stage2_dataset_release_audited` 为收据。训练配置须绑定能够消费该视图且已通过
验收的运行器；现有 full-source runner 的准入不覆盖难例视图。评测项按预登记的端点和数据集挂接。
201 的原生评测复用 197、198 已审核的对照评分。

通用 TPO 评测模板按[缓存复用规约](../../docs/engineering/evaluation-cache-reuse-v1.md)先复用已审核缓存，
再按内容身份准备缺失输入。评测分片申请设备之前完成 CPU 校验；报告记录产物身份。
