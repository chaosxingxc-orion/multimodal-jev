# SFC-Jev 教师学习反馈准备观察器

`watch_sfc_teacher_feedback.py` 从固定源码快照运行，绑定已有 TRAIN78 监督进程的
PID/启动身份、queue plan、config、preparation 及 evaluation-launch 哈希。
观察期限 110,000 秒；成功后只调用现有 `prepare_sfc_jev_teacher_feedback.prepare`，
该入口重新验收原始 78 个决策及确切 Docker，重现完整 TRAIN 分区并重新评分。
输出六条学习输入、四个会话的反馈，最多 150,000 字节，固定每类错误代表数量，
没有选择集或测试集反馈。私有消息和目录保持受限权限，不提交到仓库。

准备结束后再通过 `sfc_teacher_wire.prepared_feedback` 检查下游实际调用契约，
确认 teacher TEMPLATE、parent 指令哈希、6/78/4 覆盖、消息哈希和字节上限。
只有两层都成功才写 `feedback_ready_no_api` 和 result。
失败保留准备意图和已有输出，不自动重放；暂时 SSH 观察失败只重查原依赖。

本观察器不读取 API 凭据、不生成 executor 配置、不发请求，不启动候选推理或训练。
成功后仍需按现有单次预算协议固定教师调用，再检查响应、候选语义及真实 CPU
上下文可行性。自动准备反馈不会将候选变为“已改进”。

同时修复真实候选衔接：wire stage 生成 `sfc-jev-teacher-wire-staging-v1`，原 CPU
loader 只接受 local staging。现在支持真实 wire 类型并重验持久化请求/SSE 来源；
连通测试覆盖 stage → load，另用篡改请求、重新哈希候选指令验证不能绕过来源。
