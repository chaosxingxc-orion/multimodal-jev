# 双实例 macro batch 调度实现

`src/multimodal_jev/context_macro_rollout.py` 新增 MacroRolloutPool，封装两个独立的
score_many/switch 客户端，兼容已有批量 session 的接口。已补宏配置连接入口；
实际 NPU 设备启动与 DP2 actor 尚未接入。

一次 score_macro 接收完整 query 列表，按文本/音频长度估计成本形成 rollout 作业，
每个作业只归一个实例。两个线程分别独占一个客户端连接，允许两个实例同时执行；
同一实例内部串行接收作业并通过 score_many 交给其内部真正合批实现。调度层本身
不能证明服务端执行了 batched forward，也不能证明 NPU 数值一致性或加速。

每个结果重新绑定实际输入张量、候选 marker、policy_id 与概率/logits。结果恢复
原始顺序后才提交 COMPLETE 并返回。每个计划 slot 是一次独立曝光，重复抽到相同
输入可拥有相同内容 request_id，但 slot 不可重复消费；不把重复曝光当作漏项删除。
日志包括 INPUT、每个 rollout 结果及摘要、COMPLETE 或 FAILED；复用本地磁盘原子
提交/fsync。相同 batch_id 已有目录即拒绝重启，失败后池进入 failed 状态。

双实例 switch 并行执行，两边都确认实际 LoRA/head 激活后才更新池的可见策略。
部分激活失败禁止后续推理。一个实例异常后另一实例可能仍在完成原请求；线程等待
该原请求结束，不重派，客户端超时由既有 IPC 层处理。

10 项 CPU 测试覆盖真实线程并行屏障、整作业归属、返回顺序、重复提交拒绝、漏项、
乱序、概率矛盾、策略漂移、超时、部分策略激活和重复输入曝光。模型/客户端均为
fixture，不代表真实服务。首次默认 tmpfs 被 checkpoint 存储策略拦截；改到本地
`/srv/multimodal-jev-xingchao/tests/macro-rollout-v2` 后10项通过（2.12秒）。
测试 fixture 随后显式隔离存储准入，以支持默认 pytest tmpfs；生产检查未放宽。

## 双进程连接与 session 验证

`tools/reports/context_macro_client.py` 的 `load_pool(template, local, bundle=...)`
读取 `configs/portable/operations/context-macro-client.json` 和本地宏，连接已经启动的
两个独立服务。连接通过 BoundDualClient 校验 hello 中的 runtime、policy 和 contract。
镜像及实际源码准入仍由服务启动器负责，hello 不等于独立的远程可信证明。
socket、exchange、journal root、runtime identity 和 attempt id 均从本地宏获取。
连接并行建立；部分连接失败会关闭已建立连接并记录失败，禁止自动重试。

每次连接使用新的显式 attempt 目录；已有目录拒绝复用，内部 batch 序号从零开始。
这防止重启后悄悄覆盖旧批次，但尚不提供训练恢复。每个实例独占其传输目录。
连接入口只连接服务，不启动或停止 NPU 容器。

`tests/test_context_macro_process_integration.py` 使用两个实际 spawn 子进程、Unix
socket、文件 IPC、生产 `context_batched_service_candidate.serve_connection`，以及
128 条 query（127真实输入+1零权重padding）的 DualContextRolloutSession。
明确选择 `dual_credit_grpo_floor_clipped_v1`，不使用 session 默认的 RLOO。
模型输出和权重激活以 fixture 替代；存储准入仅在测试中隔离，原子写入/fsync仍实际执行。

通过项目宏模板连接，在每个策略首次前向使用进程屏障验证两个实例同时工作。
rollout32、内部模型batch8；两轮 prepare 与单实例参照逐项核对 request身份、
G16排序/存在性actions、rewards、advantages、old_logprob及固定reference。
第一次填充127条reference，第二次零miss；3个完整macro收据（reference、两次behavior）
全部覆盖原始slot；双实例均依次确认reference、初始policy、更新policy。
客户端exchange没有残留pt，两个worker正常退出。还有重复socket、exchange及已有attempt拒绝测试。

执行：`PYTHONPATH=src:tools/jev python -m pytest -q tests/test_context_macro_process_integration.py tests/test_context_macro_rollout.py`
结果14 passed（7.67秒）。本地 `.venv` 缺少torch，使用已有系统Python的torch2.10.0+cpu；
未安装新依赖。该结果不是实际vLLM/NPU前向、LoRA加载、actor更新或吞吐证据。
补充回归中 dual session 与原 bulk process 共17项通过；同次误在仓库根目录运行的
`test_context_dual_packaged_runtime.py` 两项在setup失败：它要求工作目录仅有一个
显式提供的封存包，当前仓库发现8个。该封存包专用检查未通过，本轮未修改历史包
或放宽其准入，也不把这两项算作通过。`git diff --check` 和新增Python编译通过。

待接入：持久候选定义清单与业务 request/slot 身份、持久消费游标及恢复对账、真实双NPU
实例启动、DP2 actor 全局归一化/恢复，以及真实音频和持续吞吐验证。
COMPLETE 仅表示完整推理结果可返回，不证明 optimizer 已消费，更不保证崩溃后
optimizer exactly-once；这需要后续训练游标与检查点的统一提交。
