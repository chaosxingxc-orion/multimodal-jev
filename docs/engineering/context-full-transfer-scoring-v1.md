# 全量三角色评测的最终评分入口

## 2026-11-16：固定队列已接通，实际评分尚待

`run_context_full_transfer_successor.py` 已通过本地宏
`.local/config/full-transfer-successor-v1.json` 启动一次后台等待，portable模板为
`configs/portable/operations/context-full-transfer-successor.json`。实际驱动
PID1919061/start341852998/cmdhash
`69bb90a4958bcea5c840ed723f1114e724b5487e2265f16cc8159332d4fa5aa4`，准确身份已核验，
202个依赖hash一致；waiting.json绑定源曲线驱动1918252。勿重复派发。

源曲线成功后重新准入冻结的三角色；每数据集每个唯一policy仅推理一次，角色
别名复用同一结果。FSC、MINDS、SLURP TEST全部推理完成后才调用生产评分器，
目标分数不反馈到训练或选点。失败即保留现场停止，不自动重试。

相关26项测试通过，实际本地配置验证通过；模型/远端运行在测试中为替身。
当前队列等待源曲线，没有真实全量目标预测或新版GRPO迁移成绩。下文关于
正式源训练/评测尚待完成的限制仍适用。

`tools/reports/collect_context_full_transfer_scores.py` 将已完成的全量目标推理收集结果
转成原生任务指标及相对起点的差值。该入口不启动推理、不选择checkpoint、不修改阈值。
它继承已部署的同一套工程配置；机器路径只在本地宏文件中绑定。

## 比较对象和指标

每个数据集必须同时提供以下角色：

- `common_Stage1`：共同起点。
- `source_selected_new_GRPO`：仅由SLURP源域selection完整曲线选出。
- `fixed_endpoint_new_GRPO`：预先固定的634步终点。

源域选择结果可能是起点，也可能是终点。这时两个角色必须引用完全相同的已执行
operations/collection，复用预测，分别保留角色和零差值，不重复推理寻找更好结果。

FSC为完整3793条、31候选；MINDS为8168条、14候选，保留14地域变体/12语言指标；
SLURP clean TEST为12991条、60候选，单独标记为源域留出集，而非跨数据集迁移。
复用既有原生评分器，报告top1、固定0.5阈值的集合指标及NLL/Brier等概率指标。
当前标签监督的是原生互斥事件，不将结果改称开放语义存在性或槽位值生成成绩。

## 执行顺序与证据

1. 从原始发布的源域曲线重算每个请求角色的policy绑定。不同远程目录中的
   portable source JSON因路径不同而hash不同；比较原始源证据身份，不比较这些副本hash。
2. 对每个唯一policy重新调用`context_full_transfer_operations.audit_collected`：
   重审源训练与selection、实际权重、容器启动/终态/传输、全量预测及同策略HF面板。
3. 核对所有角色完整输入manifest、processor、运行profile和数据集本体一致。
   从已认证的预测文件读取同一份字节并核hash，避免审核后替换预测。
4. 全部角色通过后，加载原始数据适配器，将人口、请求、候选、processor、tensor receipts
   和rendered hash与实际portable推理输入逐项比较。原始适配器保留说话人/来源等
   原生评分所需信息；这些字段不补入模型输入。
5. 最后读取独立绑定的目标标签，计算三个角色及相对起点的差值。任一角色失败则
   不计算目标指标，不通过删角色、缩人口或重选policy完成报告。

只有第2步实际完整审核通过，最终报告才写`runtime_provenance_verified=true`。
测试夹具通过不能证明真实模型运行通过。该入口不依赖已有JSON里的“passed”字样
替代源、容器、传输、输出与HF的重新审核。

## 本地配置与调用

portable入口为 `configs/portable/operations/collect-context-full-transfer-scores.json`，
其唯一宏为 `FULL_TRANSFER_SCORING`，解析后的对象包含：

| 字段 | 内容 |
|---|---|
| `entries` | 上述三个角色的字典，每个含`policy_id`和完整`operations`对象 |
| `original_evaluation` | 对应原始输入适配器配置（dataset、inputs及必要source_release） |
| `scoring` | 既有原生评分契约；FSC为population/task绑定，MINDS和SLURP为contract绑定 |
| `output` | 新的本地输出目录，不覆盖历史产物 |

其中operations使用实际全量部署生成的对象，不可填入尚未执行的运行收据。
本地宏文件采用`jev-local-macros-v1`格式，不入库。

```bash
PYTHONPATH=src:tools/jev python -m tools.reports.collect_context_full_transfer_scores \
  --config configs/portable/operations/collect-context-full-transfer-scores.json \
  --local /path/to/local-macros.json
```

成功写入`config.json`与`scores.json`，保留源绑定、三角色完整运行审核、原生指标和差值。
输出目录已经存在时拒绝运行。当前真实正式源634步曲线及目标全量推理尚未完成，
因此尚无生产评分配置或真实新版GRPO迁移成绩，不能用单测的合成分数填报。

## 本轮验证记录

真实原始输入适配器和portable全人口输入已经实际逐项核对：FSC3793、MINDS8168、
SLURP12991全部通过，未打开目标gold、未运行NPU推理。证据为
`/srv/multimodal-jev-xingchao/operations/full-transfer-scoring-input-binding-v1/report.json`，
实际执行30333终态0。该检查复用生产的`match_original_inputs`，绑定原始配置hash及
portable manifest，不仅检查条数，也比较全部请求、tensor receipts和rendering。

首轮32项测试通过119.15秒。新增归档/可搬迁源身份/渲染拒绝测试后的组合首次
58通过、1失败：本机pytest临时目录位于tmpfs，原子归档函数按生产存储规约拒绝写入。
已在该单测明确替换挂载准入边界，生产local-storage检查保留；原子写入/重命名和
fsync仍真实执行。完整人口的分数在测试中由合成预测计算，源训练/容器/HF证明
采用测试替身，不作为真实模型成绩或真实运行认证。
上述唯一失败项修正后单独复测通过（97552终态0，1项/6.23秒）；其余58项在组合
运行35741中已通过，未修改它们覆盖的生产实现。所有本轮检查已终态。
