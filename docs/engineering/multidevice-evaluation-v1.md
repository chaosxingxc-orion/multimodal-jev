# 可配置多卡评测与下一轮奖励入口

## 任务与资源

2026-11-18：评测使用 **N 张卡、N 个独立 TP1 推理实例**，各实例加载同一冻结
base/LoRA/head/marker。四卡默认服务 batch32，`macro_queries = N × batch × waves`：
四卡一波128，两卡两波128，一卡四波128。设备、实例对应关系由配置派生，无训练进程。
HF256对齐检查先在配置列表第一张卡上运行；进程确认退出后才启动全部服务，不增加卡数。
服务内部仍保留已冻结的attention8/acoustic32分段执行；接收32条不等于内部一次无分段计算。
现有runtime仅允许服务batch32/48，封包时拒绝其他值。

原选择评测仅启动两个TP1服务；另两张卡保留为训练角色，HF结束后空闲。
传输池本来支持1–8实例，此次修改外层封包、设备可见性、客户端与独立审核，保留原模型源码。
四卡的引擎计算部分相对两卡有接近2倍的理想上限；启动、HF、CPU输入加载和审核仍占时间，
不能据卡数宣称端到端加速。CPU资源上限维持原容器24核/128GiB，四服务容量须实测。

## 输入、分发与结果

- 全量SLURP selection9226、native60，固定音频缓存与任务定义；不读标签作分片依据。
- 每个macro内部按真实有效文本/音频长度估计开销，整段rollout分配给累计负载较低的实例。
- 一个rollout只交给一个实例；全部结果齐备后按原输入索引还原，再消费下一macro。
- 尾批保留真实样本数。实例失败、缺失、重复、错序、错policy或输入hash不一致均不得发布成功macro。
- 全量汇合后统一调用原评分器，保持top1、set F1、NLL、Brier及分母；不平均分片F1。
- 每个实例的ready/terminal、设备、policy/runtime及实际journal都独立审核。
  HF256面板必须覆盖每个服务；任何服务没有对齐证据均拒绝验收。
- 保留macro完整性、实际容器readonly挂载、源码/数据/适配器身份与独立collection审核。

## 代码与配置入口

模型源码不修改。新增的外层组件：

| 文件 | 职责 |
| --- | --- |
| `tools/reports/context_multidevice_evaluation.py` | 拓扑、外层runner/client生成、只读部署与多实例macro审核 |
| `tools/reports/prepare_multidevice_selection.py` | 从可信原评测包生成新包，独立重建逐字审核 |
| `tools/reports/multidevice_selection_operations.py` | validate/run/audit，复用准确容器wait、collect和共享租约 |
| `tests/test_multidevice_selection.py` | 1/2/4实例、真实HF子进程退出、包重建、服务故障与journal检查 |

共享模板为 `configs/portable/operations/multidevice-selection-package.json` 和
`multidevice-selection-operations.json`。本地宏集中绑定：

```json
{
  "EVALUATION_DEVICES": [4, 5, 6, 7],
  "EVALUATION_SERVICE_BATCH_QUERIES": 32,
  "EVALUATION_WAVES_PER_MACRO": 1
}
```

还需绑定原包完整配置 `EVALUATION_PARENT_PACKAGE`、新本地包路径、远端路径和manifest hash。
`MULTIDEVICE_SELECTION_OPERATIONS`包含共享部署字段(host/lease/shared_root/driver_root)、
`selection_package`（解析后的准备配置）、`package`、`manifest_sha256`、`operations_root`、
唯一`tag`和`selection_dispatch_allowed`。个人地址只写`.local/config`，不用修改模板。

```bash
python -B -m tools.reports.prepare_multidevice_selection \
  --config configs/portable/operations/multidevice-selection-package.json \
  --local .local/config/<evaluation>.json --action prepare
# 将返回manifest写入新的审核/部署宏；不改运行中配置。
python -B -m tools.reports.prepare_multidevice_selection \
  --config configs/portable/operations/multidevice-selection-package.json \
  --local .local/config/<evaluation-audit>.json --action audit
python -B -m tools.reports.multidevice_selection_operations \
  --config configs/portable/operations/multidevice-selection-operations.json \
  --local .local/config/<evaluation-operations>.json --action validate
python -B -m tools.reports.multidevice_selection_operations \
  --config configs/portable/operations/multidevice-selection-operations.json \
  --local .local/config/<evaluation-operations>.json --action run
```

运行命令必须由持久化后台调度器启动，并遵守同一四卡租约。`run`包括collect和独立审核，
不要把单独exit0当作成功。旧训练/评测冻结包、pins、manifest均不原位修改。

当前接入的是SLURP selection；底层传输可复用到FSC/MINDS，但其任务适配器及发布审核
尚未接入此新入口，不能把本次测试称为目标集多卡验收。

## 对照与验收

先锁同一checkpoint、完整9226样本、候选顺序、阈值、缓存、镜像、模型源码和FP32读出。
两卡32×2波与四卡32×1波都使用macro128，避免把batch改变混入实例数效应。
比较每条logits/prob及判决、完整指标；记录每实例HF覆盖数。原9226结果可作逐条一致性基线，
历史耗时只作参考，正式效率结论需相同条件下的匹配运行。

分别报告：容器总耗时、模型启动/HF耗时、完整推理`result.seconds`与queries/s、
各实例实际rollout数量/时长和负载、CPU编码/IO时间、collect+独立审核时长、峰值内存。
增加服务仍不能消除每macro的CPU准备与barrier；若主要成本在CPU/IO，应据分项计时决定后续。
不在此轮更换模型、修改阈值或优化测试集得分。CPU替身通过与真实NPU速度结论分别记录。

## 下一轮Stage2改用可编辑目标

下一轮唯一奖励/loss入口为`extensions/stage2_objective.py`。
`compute_rewards`默认返回原ranking/existence奖励，`compute_loss`默认调用原双信用目标；
固定版本是该接口的默认配置，无需再完整运行独立固定目标分支。
实验政策见`configs/experiments/stage2-editable-objective-default-v1.json`。

2026-11-18现场交接：原controller2388164和仅等待原整轮结束的后继2433863已准确身份退出，
四卡空闲及原容器终态确认后才切换；未停止远端容器，旧结果保留。
交接证据：`/srv/multimodal-jev-xingchao/operations/context-objective-handoff-v0.0.71/terminal.json`。
首次v0.0.70把只读观测的attempt误认为派发attempt，安全退出并恢复旧controller；
v0.0.71限定实际作业层级后完成交接，失败证据保留。

已启动可编辑目标有限验收控制器，配置`.local/config/context-objective-acceptance-v0.0.68.json`，
入口`tools.reports.run_objective_acceptance`，PID2452638/start ticks356227246；这只记录启动事实，
当前live状态需现场核验。先两次更新，再新进程恢复；通过后仍需tail/效率验收与正式配置发布。
不能把此有限序列称为正式634步GRPO已启动，也不能沿用旧固定版本的optimizer轨迹冒充插件续训。
Stage1的11个点已逐点审核，完整curve汇总当时尚未完成，必须保留并完成原完整曲线发布。

## 本次验收与具体候选

- 固定镜像CPU验证 **41/41通过，0失败/错误/跳过**；容器退出0、nonOOM，已独立收集。
  报告：`/srv/multimodal-jev-xingchao/operations/context-multidevice-cpu-v0.0.74/report.json`。
  这是配置/生命周期/journal测试，模型概率与上游来源使用CPU夹具，不能声明NPU已验收。
- 调度与逐条结果对比另有14项宿主CPU测试通过。比较脚本
  `tools.reports.compare_multidevice_evaluation`读取两份独立审核配置，先核对完整logits/prob/
  输入/policy/指标，再计算推理吞吐与容器耗时；不会把缺样本当作提速。
- 真实Stage1终点318四卡包已生成：
  `/srv/multimodal-jev-xingchao/operations/context-multidevice-selection-v0.0.72-prepared`，
  manifest `bd741254000679922e77016de18266e9167b03ce20be8a161e32f06555ebb63e`。
  源码、适配器、输入和HF256面板沿用原包，实例为卡4/5/6/7各TP1，服务batch32，macro128。
- 独立重建与部署控制器已启动：PID2458284/start ticks356306918，配置
  `.local/config/context-multidevice-evaluation-v0.0.75.json`，绑定1514项源码pins。
  它先通过并行CPU引擎重建候选，完成部署校验，然后用pidfd等待可编辑目标验收成功，
  再取得同一四卡租约执行评测。前驱失败即停止，单次派发，无抢占、重试或定时模型分析。
  **目前没有四卡NPU评测结果或加速倍数结论**。
- 原Stage1完整曲线发布独立在CPU后台继续：PID2456036，配置
  `.local/config/context-stage1-complete-curve-v0.0.73.json`，不占用NPU，复用内容绑定的并行审核。

推荐启动入口现在可直接使用持久化序列（`package`可引用已生成的manifest，无需重复封包）：

```bash
python -B -m tools.reports.run_multidevice_evaluation \
  --config configs/portable/operations/multidevice-evaluation-sequence.json \
  --local .local/config/<evaluation-sequence>.json --action detach
```

`MULTIDEVICE_EVALUATION_SEQUENCE`集中绑定package、deployment、固定镜像CPU报告、pins、
CPU/IO workers、900秒监控间隔和可选准确前驱。运行记录在配置指定output及其`-driver`目录。
本次新增调度只负责有限验收之后的单点评测，不自动派发尚未准入的正式训练。

2026-11-18更新：v0.0.75评测后继绑定的旧Stage2前驱被明确替换，已安全退出，
当前不处于有效排队状态。Stage2启动优先；四卡评测需在训练资源释放后以新配置绑定
新的成功终态，再执行。保留原配置/失败证据，不能宣称四卡评测已完成。
