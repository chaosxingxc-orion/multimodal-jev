# 正式 RL 选择集评测候选入口

状态：候选工具，未发布正式 Stage2。原 Stage1 训练、评测及连续 RL 验收队列保持。
本页说明现有入口如何衔接，不代表远程 launcher、正式训练或全量评测已经完成。

## 输入与运行

`tools/jev/run_context_rl_selection.py --prepare --config SPEC --output CONFIG`
从完整正式训练收集重建11点评测计划。SPEC 包含：

- `template`：固定评测模板 JSON 路径；保留模型、音频面板、代码及其他运行 pins。
- `experiment`、`stage1_proof`、`rl_proof`：三个 JSON 路径。
- `stage1_root`、`rl_root`：独立收集根目录，容器中必须同路径只读可见。
- `run_id`、`algorithm`（grpo_v1/rloo_v1）、`step`（0、64…576、634）。

准备器要求正式634步已完成及全部预定策略存在，不能用工程两步替代；RL step0
绑定 Stage1 step634，其余绑定正式 task checkpoint。准备操作只写配置，不分配设备。
`prepare_context_rl_evaluation_package.py --spec SPEC --output PACKAGE --remote-root ROOT`
复用同一准备规约，复制当前点的策略、源码及证据到独立包，重写包内路径并逐文件
封存。ROOT必须是固定`/data/xingchao/multimodal-jev/simple-final-v1/context-rl-eval-*`。
模型和数据保留Ceph只读路径；已收集源目录不再要求远程同路径挂载。
外部模型/task pins保持，旧Stage1源码和诊断branch路径不带入新包。

`launch_context_rl_evaluation.py --package PACKAGE --manifest-sha256 SHA`默认只校验
并打印命令，不访问远程；显式`--launch`才取得原共享租约、检查7卡空闲及固定镜像，
上传包并复验hash后创建容器。只挂7卡；策略/source/evidence/config均只读，输出和
cache写本地/data。attempt在创建前保存，准确句柄在start前保存；已有attempt拒绝
重建。wait连接失败保持租约、600秒后等同一容器；不据此重启任务。
这些入口已通过CPU模拟测试，尚无正式RL包或真实远程启动，不把代码存在称运行验收。

容器布局约定 `ROOT/config.json` 和输出 `ROOT/evaluation/`；后者运行前不存在。
在既有热写存储规约和单卡隔离环境中运行：

```
python tools/jev/run_context_rl_selection.py --config ROOT/config.json --output ROOT/evaluation
```

依次调用既有 HF、vLLM 探针，审核真实源/导出张量、processor 和原生 adapter，
验证256条输入/概率精确一致后，调用既有完整9226条选择集评测。HF探针自己创建
`evaluation/run`；父入口不提前创建。任何子阶段失败写 exit_code=1，不运行后续阶段。
配置/存储预检在输出创建前失败时可能没有终态文件，容器实际退出状态仍须保留。

## 收集与独立审核

`tools/jev/collect_context_rl_selection.py --launch LAUNCH --output COLLECTION`
读取准确容器身份、固定镜像及真实终态；运行中或身份不符拒绝收集，不启动新训练。
LAUNCH 要有 `container_id`、`image`、`remote_root`、`config_sha256`。
使用只读 volumes-from 收集 `config.json` 和 `evaluation/` 全部已生成文件，
包括失败现场；没有完整终态/预测文件不阻止归档。逐文件 hash 与归档人口一致后
写旁路 `COLLECTION.transport.json`，该文件不能放进已认证 collection 内部。
传输失败保留目录/归档/准确句柄，不能通过重新启动评测来覆盖失败。

`tools/jev/audit_context_rl_selection.py --spec AUDIT_SPEC --output PROOF`
重新从正式训练 proof 重建计划，独立核对：

- 完整收集终态和启动配置 hash；所有收集文件必须有传输 hash。
- 真实 source/export 张量、processor、adapter，以及256条 HF/vLLM 输入和概率。
- 完整9226条来源/顺序、候选问题、音频 hash、输入收据、逐条文件与最终预测一致。
- 实际 selection runtime profile、推理策略身份、重新计算的指标。

AUDIT_SPEC 包含 `collection`、`transport`（JSON路径）、`expected_config_sha256`、
`step`，及与准备时相同的 `experiment/stage1_proof/rl_proof/stage1_root/rl_root/run_id/algorithm`。
输出保留 source_policy_id 与 inference_policy_id，不能因张量相同就改写预测ID。
审核不会重新生成完整人口的processor张量，报告明确保留这个限制。

## 已验证与剩余工作

固定训练镜像52ba94e1的CPU容器f87bcc176e14已exit0/nonOOM，101项测试全部通过，
耗时146.73秒。独立collector核对容器/image、未挂NPU设备、全部源码与runner hash、
真实终态及日志；证据`context-rl-evaluation-cpu-v1.json`。覆盖准备/打包/模拟启动/
收集/单点审核/曲线汇总，其中模型及SSH仍为测试替身，不是实际NPU评测验收。
新增prepare阶段拒绝旧诊断开关，避免改变导出契约/数值执行；现有Stage1评测模板
未启用这些开关。没有改变原冻结训练或已排队连续RL包。

合成测试覆盖完整9226条逐记录审核、实际指标复算、错点/错指标/错记录/运行中拒绝、
调度顺序、失败收尾及安全归档。模型前向在这些测试中使用替身，不构成NPU效果。
另以既有真实 Stage1 prefix step2 和 resident 导出权重调用新 authenticator，
正确权重通过、step1错误来源拒绝；证据 `context-rl-selection-export-check-v1.json`。
它只证明复用的真实权重核验路径，不是正式 RL 或完整 selection 结果。

新增`tools/jev/audit_context_rl_curve.py --spec CURVE_SPEC --output REPORT`：共同
Stage1/RL来源、experiment、run_id和algorithm只提供一次，逐点指定step、collection、
transport文件和expected_config_sha256。入口重新审核实际产物，不信任持久化passed标记。
支持从step0开始的连续评测前缀；不满11点不选模型。完整曲线同时输出推理策略与
源checkpoint身份，不改写预测ID，并保留指标退化。合成绑定测试不代表实际完整曲线。

剩余：连续验收后封存正式634步 runtime，使用真实完整训练产物生成评测发布包，
实际执行已准备的launcher，再collect→audit→curve。任何一步未完成，
不能将离线 CPU 通过或单点传输成功称为完整 RL 评测准入。

## 正式训练与两步验收入口的差异

当前排队的`train_context_online_candidate.py`显式拒绝正式派发，最多两更新；
每步保存完整FP32 master/Adam四分片、assembled execution、joint policy、adapter，
然后通过IPC激活下一行为策略。原排队版本中`checkpoint_interval`还没有控制这条路径。
不能只将工程标志切成false就称为正式634步版本。

连续与恢复验收通过后，版本化正式入口须将两种生命周期分开：每一步仍导出并
激活真实LoRA/head联合策略；优化器恢复checkpoint及评测保留节点按64…576、634保存。
不需要为每一步保留Adam分片；用于rollout的策略和adapter在服务确认切换后，才可
按明确所有权回收非评测节点。固定Stage1 reference不能回收或改成当前策略。
原两步candidate及排队pins不修改；这个变更影响恢复/生命周期，需新的独立验收，
不能套用已有两步证据声明已完成。模型前向、奖励、优势和数值配置保持固定。

## 间隔 checkpoint 候选的独立核验

`prepare_context_checkpoint_runtime.py`已生成独立有界候选，保留原两步守卫，
每步导出/激活策略，按间隔及段末保存恢复状态；不改变原排队训练器。
`audit_context_checkpoint_lifecycle.py`消费收集后的run目录及run相对路径SHA256映射：

```bash
python tools/jev/audit_context_checkpoint_lifecycle.py \
  --root COLLECTED/run --files RUN_FILES.json \
  --start-update 0 --stop-update 2 --checkpoint-interval 64
```

逐步核验策略manifest与资产、激活回执及四rank记录；恢复节点核验四分片、
execution及commit，非恢复节点禁止混入恢复文件。该工具只读、不删文件、
不启动训练，不替代模型张量/Adam内容/采样复算及准确终态审核。
24项CPU测试通过，测试设备和引擎使用替身，证据context-checkpoint-lifecycle-cpu-v1.json。
完整正式collector接入及NPU验收仍待原连续更新/恢复链完成后推进；不能据此发布Stage2。

`prepare_context_checkpoint_package.py`现已封存独立同容器两步候选包
`.local/decision-objective-research/context-checkpoint-shared-npu-v1-package`，
manifest SHA `87a141b776bbc8c32284cdc341165e3e5b5f44cba58db8e009146b4a72843c4b`。
沿用原起点、日程、种子、算法及runner，仅替换checkpoint候选trainer、间隔64和本包路径/pins。
期望发布策略1/2、只保存恢复checkpoint2。其step2应与原每步保存版本逐张量比较；
既有fresh-process verifier要求恢复回执，不能直接把两次从step0运行冒充恢复重放。
37项CPU测试已通过；此记录当时尚无候选远程容器或launcher/collector接入。

后续已接独立launcher/collector及保存频率对照工具，48相关CPU tests通过，
launcher仅做prepare-only、collector尚未真实终态运行。接续driver483225等待原恢复链，
并重新验证连续与新进程恢复后才取得共同设备租约。新链10文件pins已封存，勿改排队源码。
collector每步保留原采样/奖励/优势复算及真实策略/adapter验证，仅保存点验Adam，
另加生命周期核验；最终对比两次从step0执行的两步结果，不冒称新进程恢复。
证据context-checkpoint-queue-v1.json；尚无此候选NPU结果或正式Stage2准入。

## 正式训练收集后的审核候选

`python -m tools.jev.audit_context_rl_training --spec SPEC.json --output PROOF.json`
消费collection、transport、launch、experiment、schedule、initial_policy六项。
后三份JSON输入中的transport/launch/experiment在CLI按路径读取；schedule为固定日程目录，
initial_policy为已认证Stage1完整终点目录。transport必须先绑定准确终态容器及全部收集文件。
候选逐步审核完整634更新、两轮40513人口、每步采样/优势和策略、固定保存点优化器。
输出供完整RL评测计划使用；它不替代Stage1来源准入和正式runtime发布证据。
45项组成模块CPU测试及真实历史checkpoint helper验证通过，尚未完整634编排执行，
未接正式transport collector或自动派发，不把候选passed结构作为实际验收证据。

终态transport现有独立候选：

```bash
python -m tools.jev.collect_context_rl_training --launch LAUNCH.json --output COLLECTED
```

仅从准确exited容器只读收集，不挂NPU设备；成功/失败都归档，执行源码或启动绑定
漂移在归档后拒绝。输出COLLECTED.transport.json与COLLECTED.tar；transport成功不等于
训练成功，后续audit需检验正常终态、完整人口和实际张量。大文件/归档采用流式hash。
35项CPU tests过（模拟SSH/docker、真实reader脚本和tar），尚无该collector的远程执行。
来源身份/正式runtime封版仍是独立条件；不得直接将现有两步候选开关改成正式准入。
