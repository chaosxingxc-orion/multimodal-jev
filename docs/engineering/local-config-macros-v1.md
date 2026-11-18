# 共享配置与本地宏

所有新增、修订的可执行配置必须把机器绑定集中到本地宏文件。共享配置保留任务语义、
算法参数和逻辑资源名；路径、主机别名、设备映射及本地文件身份通过`${NAME}`引用。
不在每个配置里重复写个人目录或服务器地址。模型/数据内容版本仍须校验，宏化不放宽
训推一致性、数据人口或冻结依赖的要求。

## 团队使用

本地绑定唯一入口默认为`.local/config/macros.json`，已由现有`.gitignore`排除。
团队模板为`configs/local-macros.example.json`，复制到本地后填写部署地址和产物身份。
可用`--local /path/to/macros.json`显式指定；其次使用`JEV_LOCAL_CONFIG`环境变量，最后
采用默认入口。环境变量只选择本地文件，不隐式覆盖宏值。凭据不要放入共享配置或模板，
沿用凭据管理渠道。

共享操作模板位于`configs/portable/operations/`。从仓库根目录运行：

```bash
PYTHONPATH=src:. python -m tools.configure \
  --config configs/portable/operations/committed-signal.json \
  --local .local/config/macros.json \
  --output .local/config/resolved/committed-signal-v1.json
```

`--output`写入本地解析配置和`.provenance.json`，记录模板、本地绑定及解析结果hash；
此步骤仅解析，不启动报告、模型或训练。之后将生成路径传给对应固定脚本的`--config`。
运行新任务时须为宏中的产物输出目录选择新的版本，不能覆盖已完成报告。
解析后的文件包含本地路径，仅用于本地执行和追溯，不作为团队共享模板。

## 解析协议

- `${DATASETS_ROOT}/slurp`：字符串拼接；宏可引用其他宏。
- `"${TRAIN_DEVICES}"`、`"${SIGNAL_STEP}"`：整个值为宏时保留列表、整数等JSON类型。
- 缺失宏、循环引用、非法名称、未解析表达式或将列表/数值嵌入字符串均报错。
- 不执行shell、Python表达式或命令替换，不自动展开`$HOME`或猜测默认机器地址。
- 宏仅替换JSON值，不替换对象键；普通无宏配置也可由同一解析器处理。
- 本地写入复用存储规约和持久提交；重复解析同一版本只校验，不覆盖；部分提交保留现场。

统一API为`multimodal_jev.local_config.load_config(template, local)`，返回解析后的配置
和不包含宏值的来源记录。`tools.configure`作为兼容现有JSON加载器的前置入口，避免
每个训练/推理/清理脚本独立实现替换逻辑。旧加载器不能直接读取宏模板，必须消费解析产物。

## 当前覆盖与冻结边界

已实现通用JSON解析器、CLI、本地绑定文件及团队模板，已为训练信号统计和概率错误
分层提供共享宏配置。10项测试通过；两份模板在当前本地绑定下与旧配置的解析值逐项相等，
实际本地写入及重复解析通过。字节hash不保证与旧配置相同，不借此覆写旧运行身份。

当前库存另有大量历史JSON及TSV/TXT配置；**本次没有宣称全部完成迁移**。已封存的
`configs/releases/`、运行/排队任务引用的配置与证明保持原样，作为历史证据而非新任务的
共享模板。逐个工具发布新版本时迁移到统一入口；不能直接全局替换路径破坏hash或来源证明。
当前通用层只支持JSON，TSV标签表和TXT资源列表暂不做字符串宏替换；若作为执行配置
使用，须用宏化JSON描述其资源位置，并继续校验原文件内容。

该机制解耦配置的机器绑定，不自动搬迁数据、改写历史归档的绝对来源路径或保证已有
分析器可在任意工作目录运行；现有分析入口仍要求从仓库根目录执行。

## 训练及推理服务模板

`configs/portable/training/context-grpo-actor-v1.json`与
`context-grpo-service-v1.json`已从实际arithmetic-v2运行包派生。模型、数据、设备、缓存、
IPC、策略契约、源码pins和发布证明统一通过本地宏绑定，保留global128、micro8、GRPO
及原数值协议。`SOURCE.json`记录原配置hash，可用固定工具复核：

```bash
PYTHONPATH=src:. python -m tools.verify_portable_training \
  --local .local/config/macros.json \
  --source .local/decision-objective-research/context-rl-grpo-arithmetic-v2-package
```

上述source为本机已有验收包；团队提供自己的同内容包位置。校验器只读，不派发。
实际两配置逐字段一致，唯一预期变化是actor的`training_dispatch_allowed=false`。
三项测试覆盖模板无本机路径、团队改绑不改变算法参数、原包或算法变动时拒绝等价声明。
本机解析文件位于`.local/config/resolved/context-grpo-{actor,service}-portable-v1.json`，
核验记录为`.local/config/training-portability-verification-v1.json`。

这两个模板尚未接替现有launcher的打包和准入流程。新主机的镜像、策略契约、pins、
资源挂载与发布证明必须由同一工程验证并重新发布，不能将旧证明的地址改掉就视作通过。
示例中的空契约/空pins是待填写占位值，解析成功不代表内容合法；不要直接开启派发。
当前本地宏的派发开关也为false，不影响原运行包中的既有开关。

集中绑定文件扩展后，其整体hash会改变；原解析产物继续保留历史来源记录。再次解析
使用新的输出版本，不覆写已发布配置，也不把宏更新传播进已运行任务。

## 执行配置复用的宏依赖范围

新`tools.configure`产物使用`jev-materialized-config-v2`来源记录：首次生成时保留整个
本地绑定文件hash用于追溯；再次解析以模板hash、解析后语义hash、实际用到的宏及其
递归依赖hash判断复用。只修改未引用的清理目录等宏，不再迫使训练配置重新发布。
修改被引用的模型路径、参数、宏别名表达式或输出文件仍会拒绝覆盖；旧来源记录保持原字节。

旧v1来源记录继续严格比较整个绑定文件hash，不静默升级。此变更仅适用于执行配置
生成器；日程、轨迹、报告流程的发布/源文件校验仍保留其既有整体绑定策略，不能据此
假定所有已完成workflow都可在宏文件编辑后复用。后续运行继续用冻结的宏快照。

28项配置/编排测试通过。真实actor模板在独立本地盘完成三项检查：首次生成、修改
无关CLEANUP_OUTPUT后already_resolved、修改ASR_MODEL后拒绝且配置/来源记录不变。
证据`/var/tmp/jev-config-reuse-kzj6dcm5/verification.json`；未修改默认本地宏或启动训练。

## Ascend运行环境计划

模板`configs/portable/training/ascend-runtime.json`集中引用TRAIN_HOST、TRAIN_IMAGE、
TRAIN_ROOT、TRAIN_MOUNT_SOURCE_CONTAINER、ASCEND_DRIVER_ROOT及现有数据/设备宏。
用`PYTHONPATH=src:. python -m tools.plan_runtime --config configs/portable/training/ascend-runtime.json --local …`
生成参数列表；工具不调用SSH/Docker、不创建容器，只提供待验收的部署计划。

镜像固定为实际NGA衍生镜像`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
这是本地Docker image ID，不是可直接拉取的registry digest。已记录栈为torch2.10.0+cpu、
torch-npu2.10.0.post7、transformers5.14.1、verl0.8.0.dev0、vLLM0.30.0+empty、
vLLM-Ascend0.30.0rc2，版本证据见[npu运行记录](npu-audio-runtime-v1-result.json)和
[镜像说明](runtime.md)。训练与推理服务共用这套镜像；不要用旧CPU smoke镜像替代。

保留24CPU/128GiB内存/8GiB共享内存、四NPU、本地缓存、只读共享资产及日志轮转。
10项计划检查通过，实际生成argv与原GRPO launch逐参数相等，证据
`/var/tmp/jev-runtime-plan-m9yiv8kc/verification.json`。这是对历史启动记录的本地比对，
没有新增目标机版本探测或跨机器训推验证。原冻结launcher尚未切换到此计划入口。

新部署需验证目标机image ID、已成功退出的挂载来源容器及其继承挂载、本地存储/容量、
资源租约和完整发布身份；这些检查不能由字符串路径或同版本号替代。现有源码和策略
绑定由新发布流程生成，禁止直接复用本机container ID。默认本地宏未改，本次验收使用
独立--local文件，团队示例中的挂载来源ID为占位值。
