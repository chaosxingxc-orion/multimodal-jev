# SFC-Jev 教师候选的上下文可行性

候选提示不等于已获准派发的 actor。`sfc_prompt_candidates.py` 仅替换通用
INSTRUCTION，保留函数定义、环境、当前决策、固定输出约束、音频和官方历史。
新增的 CPU 路径用于核实替换后每条输入的真实音频展开、token 数和上下文预算。

## 固定输入与输出

输入为已有模型目录、原始完整 SFC-Jev release、候选 staging 目录、
`staged.json` 的 SHA256 和 `p1`/`p2` ID。加载时检查 receipt 中的 parent 提示哈希、
候选文件哈希、字符数和指令结构。检查开始与结束均核对候选身份。
不读取监督目标或参考转写。

支持本地结构暂存和真实教师 wire 暂存两种 receipt。真实教师候选加载时重新
验证原始请求、SSE、最终内容与用量的本地一致性，并逐字核对指令；不把
`wire_authenticated=true` 标签本身当作来源证明。候选或 receipt 即使重新计算
哈希，也不能覆盖原始响应。语义审查仍为 pending，CPU 检查不自动批准候选。

实际执行沿用全量 parent processor 审计：全部 8,895 bundles / 136,722 decisions，
逐条加载完整单/双音频、校验音频哈希、运行本地官方 HF processor，检查
展开的音频片段与完整帧覆盖、非有限特征、4096 上下文与 1024 输出预算。
不因为超长音频、输入溢出或候选文本长度变化而减少人口。溢出记录为
`context_overflow=true`、`all_inputs_fit=false`，不是静默截断。

身份文件、结果文件和独立验收均携带 `prompt` 对象，绑定 staging/指令/parent
哈希以及 `kind=staged-candidate-context-only`。记录各决策的 prompt/ID 哈希，
供将来实际候选引擎核对。该分支保持 `candidate_accepted=false`：
CPU 可行性不证明教师来源可信、语义审查通过或任务效果改善。

## 调用方式

沿用已固定镜像和独立源码快照、CPU 4 核 / 内存 16 GiB / 禁网 / 无 NPU，
模型、release、候选 staging 只读挂载。新建实验目录，不能覆盖 parent 审计。
现有完整 parent 审计及运行中的实验快照保持独立。
对于真实教师候选，还需只读挂载完整 teacher request root（包含实际请求、SSE、
配置及进程终态）和配置中固定哈希的 executor 文件。配置前应将 executor 按原字节
放入共享实验目录，固定该路径；不能向 175 容器传入仅本机工作区可见的路径。
验证只读取 executor 文件哈希，不执行教师代码或再次发送 API 请求。

```bash
python3 -B tools/jev/audit_sfc_jev_context.py \
  --model <local-qwen3-omni> --release <complete-sfc-jev-release> \
  --output <new-candidate-audit-root>/run \
  --staging <staged-candidates> --staging-hash <staged-json-sha256> \
  --candidate-id p1

python3 -B tools/jev/verify_sfc_jev_context.py \
  --root <new-candidate-audit-root> --output <new-acceptance.json> \
  --staging <staged-candidates> --staging-hash <staged-json-sha256> \
  --candidate-id p1
```

第二条命令要求独立启动记录和成功退出的确切 CPU Docker 身份。未传候选参数时，
验收器只接受 parent；冻结 parent shard runtime 也显式拒绝含候选身份的审计。
不得将候选结果冒充 parent，以绕开实际候选引擎验证。

## 当前边界与下一步

代码测试覆盖实际候选 adapter、合成完整三划分、单/双音频、正常预算和溢出、
候选绑定、parent/candidate 隔离；其中 processor 和 Docker 使用明确的测试替身。
尚无真实 SFC 教师响应，本轮没有启动真实候选 CPU 审计或候选模型推理。

兼容性复核：新版 verifier 已重新检查原 parent CPU 容器
`844aa68b45dcbe41f8ca76c96cf8dbb9d53ced940a422c8a002af52bc63b1073` 的实际退出状态及
完整三划分记录，仍确认 136,722 个决策全部满足预算。
原结果 SHA256 为 `42ffcdc9e5c89fdae87d2a9ad779f75ff4f89bed05d65940299453dabd772f3e`；
原 acceptance 文件没有覆盖。候选、parent、分片与合并相关 95 项测试通过。

下一步依次为：TRAIN78 验收 → learning 反馈 → 固定预算单次教师调用 →
候选语义审查及本 CPU 可行性检查 → 候选引擎输入一致性验证 →
完整 selection 比较并锁定候选 → 完整官方 TEST 比较。
CPU 检查可在语义审查前进行，但不能凭它批准模型推理。
