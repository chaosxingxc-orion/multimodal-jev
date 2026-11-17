# 旧 RLOO 的恢复数据与交接缺口

恢复工具固定镜像检查已完成：容器
`45bb484735fd958e6308335dea4d6f3a78c8ae0c4d34b42635a8a5278ab58f50`
准确 wait 返回0，独立 collector 核验 exit0、非OOM、无NPU设备，59项测试通过、零跳过。
快照共1,232文件；报告
[context-arithmetic-resume-tools-fixed-image-cpu-v2.json](../../../context-arithmetic-resume-tools-fixed-image-cpu-v2.json)。
配置 `configs/portable/operations/context-arithmetic-resume-tools-cpu-v2.json`，本地宏
`.local/config/arithmetic-resume-tools-cpu-v2.json`。范围包括有限恢复准入、挂载规划、
拒绝条件、保存的group/终态比对；仍非完整真实恢复归档组合、NPU更新或提速证明。
本次未改变冻结恢复包、原训练、服务或后继队列。

恢复后的独立比对入口已补充：`tools.reports.audit_context_arithmetic_resume`；模板
`configs/portable/operations/context-arithmetic-resume-audit.json`，本地宏配置
`.local/config/rloo-resume-audit-v1.json`。当前没有实际恢复 collection，尚未执行完整审计。
入口绑定启动包、Docker inspect、transport 与完整归档，再重审原连续两步参照。
逐步比较实际 batch、采样、优势、行为／参考输入、BF16 policy 和 processor；同时检查
四 rank restored 收据、每个 microbatch 的行为一致性、正常终态、末步 LoRA/head
master 导出和 Adam 合法性。策略文件序列化身份可以不同，但必须各自通过身份校验、
采样概率复核，且实际张量精确一致，不能仅忽略 policy ID 后宣布一致。

明确保留 `final_adam_equal_to_continuous=null`：旧 step66 没有 Adam 分片，只有新恢复
终点的 Adam 独立合法性检查，不能宣称连续／恢复的优化器末态精确相同。
新增14项合成 batch/group/终态反例测试，与原 following/operations 合计34项本地通过。
测试覆盖范围不包括完整真实归档组合、NPU恢复、固定镜像执行或吞吐；这些仍待验证。

有限恢复操作入口已补充：`tools.reports.context_arithmetic_resume_operations`，共享配置
`configs/portable/operations/context-arithmetic-resume-operations.json`。默认只在本地校验
封包身份、原目标、两步游标、推理配置、依赖挂载与设备命令，不调用 SSH。当前本地
配置 `.local/config/rloo-resume-operations-v1.json` 对真实 379 文件候选校验通过。

实际派发开关仍为 false。启动路径复用原设备 lease、准确容器 wait 和终态 collector，
要求卡4–7空闲；原运行目录、基座与共享数据只读，恢复输出独立。保存准确句柄后再
start，观察超时不重建容器，失败先收集产物再拒绝验收。没有自动停旧训练、抢占服务、
修改队列或全量续训功能。新入口及既有恢复准入／挂载约束共37项本地测试通过；本轮
仅验证本地准入、命令和拒绝条件，尚无真实 launcher 启动、终态收集或 NPU 恢复证明。

已补充连续运行对照：`rloo-step0064-following-snapshot-v1` 成功收集 54 个文件，
包含 step64 恢复状态，以及原任务完成 step65/66 的采样、策略发布、激活和四 rank
更新记录。固定入口 `tools.reports.audit_context_following_snapshot` 通过全量 40,513
训练人口和 634 步日程重新绑定这两步，复算 256 个请求的 G16 采样、reward、RLOO
advantage，并检查 BF16 policy、冻结 marker 和四 rank 保存的一致性记录。
报告为 [context-rloo-following-reference-v1.json](../../../context-rloo-following-reference-v1.json)。
共享配置 `configs/portable/operations/context-following-reference-audit.json` 的路径全部
来自本地宏；收集及记录校验相关 21 项本地测试通过。

实际最慢 rank 步时为 99.533/98.725 秒；rollout 为 65.728/64.134 秒，其中固定
reference 为 31.954/31.915 秒，actor 为 23.350/23.399 秒。reference 是 rollout
的子项，不能重复相加。这是两步旧运行的耗时分布，不是新版收益或代表性全程统计。
两批无组内 reward 差异的请求分别 91/128、104/128，仅作训练信号诊断，不是评测收益。

第一次本地审计将保存的 group 顺序误当原始 slot 顺序而拒绝；核对原 trainer 后改为
严格验证冻结日程的四 rank 索引拼接顺序，再次真实审计通过；原训练数据未修改。
step65/66 原任务未保存 master/Adam，不能宣称后续恢复可与不存在的原 Adam 精确
对照。现有参照仅支持采样和 BF16 policy 对比；实际 NPU 恢复、原生 adapter 张量
复验和新版端到端吞吐仍须独立执行。本轮没有停止训练，也未派发新 NPU 任务。

为支持确认效率收益后的受控切换，已在不停止训练的情况下收集原 RLOO 第 64 步
已提交检查点。来源为容器
`c7a9dbd1281556ba13773e41e55c102f2af7b3b1e21e6d577d8ab029dae1ee06`，
固定镜像 `52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
只绑定显式 step，不自动选择正在写入的最新目录。

入口 `tools.reports.collect_committed_context_checkpoint` 读取已提交 commit，核验
四 rank 分片、execution、联合 policy 与原生 adapter 的文件哈希；归档前再次核验
同一清单，传输后复验全部字节。调用只读、无网络、无 NPU 的 CPU reader 容器，
仅挂载原运行根目录；不继承训练容器设备或其他挂载，不复制基座和 reference cache。
共享模板为 `configs/portable/operations/context-committed-checkpoint-snapshot.json`，
机器地址和路径来自本地宏。原终态训练 collector 保持不变。

第一次宿主读取因 policy 目录为容器用户所有而 PermissionError，未修改原目录权限。
失败现场保留为 snapshot-v1。采用上述只读 reader 后 snapshot-v2 完成，18 文件共
123,863,044 字节；四份恢复分片各约 20.83 MB，仅 LoRA/head master 和 Adam。
本地实际收集目录：
`/srv/multimodal-jev-xingchao/collections/rloo-step0064-recovery-snapshot-v2`。

独立入口 `tools.reports.audit_committed_context_checkpoint` 复验 launch、归档、
配置、原 RLOO 目标和优化器契约，校验联合 policy 的所有权重与 processor 文件，
按 rank 顺序重组 master 后精确比较 BF16 execution，并将四份状态实际装入新建
CPU 参数及 Adam 优化器。真实结果：128 参数、6,878,209 元素的导出一致，四 rank
master/Adam 精确恢复。报告 `context-rloo-step0064-cpu-recovery-v1.json`。
原生 adapter 本次核验文件身份，尚未执行 safetensors 张量重载；报告明确区分。
收集器、master 导出和 checkpoint 恢复相关本地测试共 25 项通过。收集器测试包含
真实 stdlib 子进程 tar 往返，以及分片、游标、配置、符号链接、路径与 adapter
改变时拒绝；这些测试不声称执行了 NPU 训练。

这不是新进程 NPU 恢复证明，也不是新版提速证据。当前正式入口
`context_formal_runtime.validate` 固定 start0→634、禁止 resume_root；仅底层训练器
有恢复代码不足以证明正式任务可续跑。不能在原包中修改游标或放宽正式验证后重启。
下一步受控交接仍需要：

1. 独立封存从原检查点恢复的运行配置，保留相同目标、日程、参考策略和数值协议。
2. 新进程实际恢复与连续执行对照；CPU 还原不能代替此项。
3. 原驱动与后继队列交接，保留被中止运行的真实退出状态及已消耗成本。
4. 新版同负载端到端吞吐验证通过后，才执行已获授权的正式切换。

已准备有限恢复候选 `context-rloo-recovery-step0064-v1-package`，379 文件，
PREPARED SHA256 为
`bf436f9a7a312bbdd485a2fa33a4a83da4f5a8e18e56e2f4dad9bd4c3228d1d8`。
入口 `prepare_context_arithmetic_resume` 在封包时重新执行真实 CPU 检查点审核，
要求来源与原正式包 config/manifest 一致。共享配置为
`configs/portable/operations/context-arithmetic-resume-prepare.json`。

候选只复制源码和日程，零 `.pt`/`.safetensors`；Stage1、基座和 step64 的
LoRA/head/Adam 均引用原只读路径及哈希。训练器与 supervisor 分别绑定原源码
SHA256，只将 `context_arithmetic_formal_runtime` 校验导入替换为独立恢复校验。
反向替换后与原两文件字节相同，优化器/更新/保存/激活/恢复代码未改。

恢复校验先验证原正式发布，再严格对照允许变动的配置：start64→stop66、恢复根、
当前 behavior adapter、隔离输出与冷 reference cache。固定 Stage1 reference、
目标、全量日程、batch、精度和 checkpoint interval64 不变；step66 作为候选终点
会提交恢复状态。所有新旧依赖均要求完整 pins。完整续训不被这个入口准入，
`training_dispatch_allowed=false`，也未实现自动停训或队列交接。

18 项本地准入测试通过，包含改变算法/参考/batch/游标/派发标志/分片/原发布的拒绝。
这些测试使用合成正式发布，候选实际运行时的源路径/挂载验证和 NPU 恢复仍待。
同组 18 项在固定镜像中全过、零跳过，容器
`959de164aca2adbbce5313f7cf82de34a68c7c6a86dfae2298646387c6948a4e`
准确 wait0 并独立收集正常退出。报告
`context-arithmetic-resume-fixed-image-cpu-v1.json`；固定镜像包含 1224 个源码/测试文件，
没有挂载 NPU 或模型权重。

实际主机路径的 CPU 预检也已完成。使用同一个封存恢复包，映射到其计划 runtime
路径；原运行根、共享模型/数据、特征缓存和音频镜像均只读。容器没有设备挂载、
无网络，根文件系统只读，只有隔离的日志/临时目录可写。入口为
`tools.reports.check_arithmetic_resume_environment`，配置模板
`configs/portable/operations/context-arithmetic-resume-preflight.json`。不启动 supervisor，
不构建基座模型，原模型文件只用于已有依赖哈希校验。

容器 `256bca125e1fa5aad25830c98be70c18175d2b254cbdfdb22615c04785ee4193`
准确 wait0，实际 inspect 证明 exit0、非 OOM、零 NPU 设备及上述挂载约束。
433 项实际依赖通过原发布与恢复校验；step64 的原生 adapter 已通过
`verify_native_adapter` 重新读取 safetensors 并逐张量对照联合 policy；四份真实
master/Adam 在固定镜像中再次精确恢复。报告 `context-rloo-resume-preflight-v1.json`。
这补足前文“原生 adapter 仅 hash”和“实际 runtime 路径未检查”的缺口，不是
NPU FSDP 恢复、模型前向、optimizer 更新或性能验证。

本轮实际执行后的 runner 远程哈希及 inspect 环境逐项补验也通过，证据为
`context-rloo-resume-preflight-identity-v1.json`。后续入口已将 runner 传输哈希检查
放在 create 前，并在收集时验证环境；没有为这些检查重复执行恢复任务。
预检挂载/拒绝测试及原恢复准入共 25 项本地检查通过，报告不将它们视作 NPU 验收。
旧监听新收据为四 rank 第106步、最慢101.042秒，next_due1794800892.2863834。

资源核查中，175 0–3 的 `omni-npu175-instruct-tp4-v1-run01` 和 181 的两个
`saea-*-instruct-local-storage-v1-r3` 均确认为运行中的 Omni 服务，不是遗留进程；
181 B 服务本次实际启动时间为 2026-11-16 02:53:23 UTC。未停止任何服务。
原监听最新消费为第 97 步、最慢 rank 98.817 秒，下一监听截止
1794799992.2854755；该信息是观察时点状态，不是当前最新进度保证。
