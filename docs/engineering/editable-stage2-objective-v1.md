# 编辑一个Python文件调整Stage2奖励与优化函数

**后续主要编辑文件：`extensions/stage2_objective.py`。**
这不是固定奖励菜单。研究人员可以修改奖励公式、奖励组合、优势处理或最终可微分loss，
训练循环继续负责模型forward、反向传播、DP累积、optimizer、LoRA导出与恢复。

## 两个编辑入口

`compute_rewards(ctx)`：默认返回每条query的ranking/existence奖励映射，张量为
`[组件数, G]`。可直接改成连续奖励、分段奖励或按任务定义的新公式。返回值会交给
`compute_loss(ctx, rewards)`；若自行实现loss，两者之间也可以约定新的reward结构。

`compute_loss(ctx, rewards)`：返回`(标量loss, metrics字典)`。可以调用
`ctx.baseline_loss(rewards)`保留已有目标并叠加项，也可以重写整个优化函数。
默认实现保持既有排序/存在性GRPO及辅助loss/KL。修改示例：

```python
def compute_loss(ctx, rewards):
    loss, metrics = ctx.baseline_loss(rewards)
    # 示例：额外的概率监督项。不是推荐权重或正式实验方案。
    import torch
    labels = torch.tensor(
        [r['binding']['labels'] for r in ctx.records],
        device=ctx.logits.device, dtype=torch.float32,
    )
    per_query = (ctx.logits.float().sigmoid() - labels).square().mean(-1)
    extra = (per_query * ctx.real_requests.to(per_query.device)).sum()
    extra = extra * ctx.world_size / ctx.global_real_requests
    metrics['extra_brier'] = float(extra.detach())
    return loss + ctx.options.get('extra_brier_coefficient', 0.1) * extra, metrics
```

传入ctx还包含真实行为记录、身份、当前可微logits、固定Stage1 reference概率、
真实样本mask、global denominator、world size及任意JSON参数`options`。
梯度必须通过当前logits传递；不能把指标浮点数包装成假loss。padding必须零贡献，
按global real query分母归一化，不能在microbatch内重新求均值后改变全局权重。

若引入“完全正确/部分正确”的整组任务奖励，可以在本文件中实现joint-action loss。
不要将整组reward直接复制给各bit后声称它等价于joint GRPO。已有`context_task_reward`
只是可选计算工具，不是限制扩展能力的固定菜单。

## 实际训练接入

新版本训练包使用 `tools.reports.context_plugin_trainer`，生成自原cached-reference
trainer，仅替换actor loss加载入口，并将插件身份写入checkpoint contract。
actor使用局部函数作用域注入，不改原模块全局变量；原行为/输入/策略数值检查保持。
配置新增`objective_plugin`对象：

```json
{
  "path": "${STAGE2_OBJECTIVE_FILE}",
  "sha256": "${STAGE2_OBJECTIVE_SHA256}",
  "options": "${STAGE2_OBJECTIVE_OPTIONS}"
}
```

共享宏片段在`configs/portable/operations/stage2-objective-plugin.json`。路径和hash由
本地宏绑定；打包时复制这个文件到运行包并绑定对应容器路径，纳入完整payload pins。
每次编辑后发布新文件hash/配置；运行中的文件不热改。`options`允许新增参数，
无需训练器逐个认识新reward名称。文件加载一次后当前实验持续使用同一版本。

这是一份可信项目Python程序，不是沙箱：允许写算法代码，不限制在几种固定公式内。
框架验证API、文件内容hash、可微且有限的标量loss、标准metrics与行为身份。
即使完全重写loss且不调用`baseline_loss`，也执行原行为数值一致性、reference形状/
无梯度/有限内部概率以及arm、真实样本mask和DP分母检查，不能通过省略KL绕过输入准入。
自定义loss的数学正确性、缺标处理、归一化、奖励尺度和信用匹配仍需对应测试。
候选actor保留rl/auxiliary/kl日志，并将自定义顶层`int`/`float`指标以
`jev/objective/<名称>`传给原训练器的metrics通道；例如上面的`extra_brier`。
名称必须为非空字符串，数值必须有限；字符串、布尔值和嵌套对象不进入数值聚合。
这些数值沿用原训练器的microbatch/rank聚合语义，不自动转换成全数据集指标；
完整population评分仍由独立评测脚本生成。插件hash与options写入checkpoint身份。

## 当前验证和边界

### 两步验收的部署与收集入口

`tools.reports.context_objective_training_operations`使用宏模板
`configs/portable/operations/objective-training-operations.json`。`--action validate`
重建候选包、原Stage1/reference准入及固定镜像bootstrap证据，只生成部署命令；
`--action run`在明确的有限派发配置下执行共享四卡租约、一次性启动、准确容器wait、
collect，然后调用`audit_objective_training`独立审核，写入带退出码的终态。
此入口尚待完整实际准入与NPU运行验收，不应直接并入正式队列。

```bash
python -m tools.reports.context_objective_training_operations \
  --config configs/portable/operations/objective-training-operations.json \
  --local /path/to/local-macros.json --action validate
```

扩展源码整体目录只读挂载，引用的共享数据、Stage1/reference和旧前向源码继续只读；
容器实际挂载也必须通过原运行审核。结果审核模板为
`configs/portable/operations/objective-training-audit.json`，既核对原source/rollout/
truth/reference/optimizer，又核对插件checkpoint身份与完整实际reward/loss轨迹。
`objective_trace_audit`才是插件实际目标重放，原`updates[].branches`仍表示原始采样信用。
成功两步验收不开放正式训练，仍需相同插件身份的新进程恢复和效率验收。

### 同插件新进程恢复

恢复包入口`tools.reports.prepare_objective_recovery`读取已经完整独立审核的两步
`parent_audit`，共享模板`configs/portable/operations/objective-recovery-candidate.json`。
只复制第一步LoRA/head/optimizer与必要marker状态，reference仍固定Stage1终点，
推理服务从第一步behavior策略启动。插件API/hash/options写入恢复checkpoint contract；
迁移文件路径允许，改公式或options不允许作为同一轨迹恢复。

生命周期`tools.reports.context_objective_recovery_operations`复用两步部署宏模板，
将`training_package`绑定到恢复包规格，执行一个从step1到step2的更新。输出审核
`tools.reports.audit_objective_recovery`保留原恢复审核，并重放step2完整实际目标轨迹；
随后逐张卡比较连续与恢复训练的master、Adam、BF16导出、采样、reference/behavior及
实际reward/loss轨迹。必须是不同容器，不能拿同一运行目录充当新进程证明。
这些入口已完成宿主夹具验证，真实NPU恢复仍待两步前驱通过；尚未开放正式训练。

### 一次性验收串接

`tools.reports.run_objective_acceptance`提供配置化的prefix→独立审核→recovery封包→
新进程运行→独立比较入口，模板`configs/portable/operations/objective-acceptance-sequence.json`。
`--action detach`脱离交互会话运行，记录PID/start ticks、日志、配置、源码pins及终态；
任何阶段失败立即停止，不重试，不自动派发正式训练。`--action validate`仅校验配置。
这是有限验收入口，不是主评测队列替代品；只能在既有四卡owner释放租约后启动。
非阻塞租约失败会保留失败证据，不允许靠循环重试与既有队列抢卡。
本入口不修改当前冻结队列。`objective_acceptance_successor`可通过准确pidfd等待既有
控制器成功终止，再调用此序列；共享模板为
`configs/portable/operations/objective-acceptance-successor.json`。前驱身份与配置hash、
后继配置和源码pins在等待前后重验；前驱失败不派发、不重试、不唤起模型。
2026-11-18已按此方式登记v0.0.69后台后继，等待当前完整训练/评测/迁移流程，
没有新增NPU执行。两步/恢复通过后仍需尾批和效率准入。

### 实际奖励轨迹与固定脚本重放

候选训练配置设`objective_trace=true`，可在`engineering_prefix`不超过两次更新时记录
实际传给`compute_loss`的奖励。每个rank的microbatch轨迹先缓存在内存，反向传播后
每次更新一次性提交到`run/rank-N/objective-traces/objective-UUUU.json`。
只保存判决logits、行为记录、reference概率、mask/分母、奖励、loss及插件身份；
不保存音频特征或主模型权重。正式训练默认关闭，不允许用此开关打开全量轨迹。
启用轨迹时拒绝插件原位修改输入或reward，保证记录的公式输入等于实际消费的输入。

审核入口为`tools.reports.audit_objective_traces`，共享模板
`configs/portable/operations/objective-trace-audit.json`。本地宏绑定插件path/hash/options、
DP宽度、每卡batch、microbatch、loss绝对容差及逐update的groups/各rank轨迹路径与哈希。
`updates`条目格式：

```json
{
  "update": 0,
  "groups": {"path": "/local/collected/run/groups-0000.pt", "sha256": "<collected-sha256>"},
  "traces": [
    {"rank": 0, "path": "/local/collected/run/rank-0/objective-traces/objective-0000.json", "sha256": "<collected-sha256>"},
    {"rank": 1, "path": "/local/collected/run/rank-1/objective-traces/objective-0000.json", "sha256": "<collected-sha256>"}
  ]
}
```

```bash
python -m tools.reports.audit_objective_traces \
  --config configs/portable/operations/objective-trace-audit.json \
  --local /path/to/local-macros.json
```

先由原独立审核确认groups对应真实日程、可信TRAIN监督和行为/reference journals，再将
收集哈希绑定给此脚本；不得由轨迹自身推导期望人口。脚本拒绝缺少rank/microbatch、
重复/错序query、mask/分母/reference错配，重算实际自定义奖励及loss。奖励和metrics
要求精确一致；loss容差由验收配置明确固定，当前CPU测试为0，不自动放宽。
当前重放在CPU执行，设备相关或随机插件不能假定可精确重放，需独立准入证据。
重放成功只证明已记录公式输入/输出的一致性，不证明真实模型前向、完整梯度或optimizer
更新正确，不替代完整训推、恢复、运行包身份和端到端独立审核；CLI不会放行训练。

CPU测试覆盖默认loss及梯度与原实现完全相等、DP/microbatch/padding、仅编辑reward
或loss即可改变梯度、stale行为拒绝、文件hash变更、错误reward/断梯度拒绝，
以及真实生成的finite/formal训练源码和隔离actor挂接。
宿主loss测试使用独立CPU surrogate；另有`tests/test_objective_plugin_native.py`
在固定训练镜像中直接调用原生verl，覆盖默认loss/梯度、DP分母及修改reward/loss的梯度变化。
2026-11-18 04:31 UTC，v0.0.55固定镜像CPU容器退出0、nonOOM，独立收集确认
21项通过、0失败/错误/跳过（其中4项原生verl检查）。镜像身份为
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
证据：`/srv/multimodal-jev-xingchao/operations/context-objective-plugin-cpu-v0.0.55/report.json`。
可复用配置为`configs/portable/operations/context-objective-plugin-cpu-tests.json`，
由`tools.reports.launch_fixed_cpu_tests`和`tools.reports.collect_fixed_cpu_tests`驱动。
这是无NPU的固定镜像数值检查，DP测试是分片梯度等价性，不是分布式进程或NPU训推一致验收。
未切换正在运行/排队的冻结包，也没有派发新增训练。后续NPU有限验收通过后可按既有授权
直接发布接管。


2026-11-18接续更新：下一轮直接使用本可编辑目标接口，固定双信用公式作为默认实现。
原等待整轮固定版本完成的v0.0.69后继已退役，v0.0.68有限验收控制器已直接启动。
配置政策见`configs/experiments/stage2-editable-objective-default-v1.json`，
具体交接与四卡评测接续见[多卡评测](multidevice-evaluation-v1.md)。
默认公式兼容不等于任意新公式自动准入；实际新公式、options和恢复身份仍需冻结与验证。
