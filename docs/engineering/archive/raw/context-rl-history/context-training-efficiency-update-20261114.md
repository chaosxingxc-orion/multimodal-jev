# 训练效率复查：热缓存实测与实验周转

## 既有日志补充：有效token与音频暴露统计

新增独立离线入口 `tools/jev/context_input_throughput.py`，不修改冻结训练代码或
排队collector。输入为已独立收集的目录和证据JSON，读取的config、四rank终态、
每步记录与batch metadata均对照collector的文件hash；统计完整配置范围，缺记录拒绝。
11项CPU测试覆盖零权重padding、并行rank耗时、后台输入重叠、不同rank阶段最大值、
错分母/分区/计时/保存标记及真实文件篡改拒绝。

复用原admitted prefix两步，256有效样本、726471有效输入tokens、741.810秒音频，
sum(max-rank step time)=44.567秒；得到5.744样本/秒、16300.559输入tokens/秒、
16.645音频秒/秒。两步均保存checkpoint，常规热身后窗口为null，不能称稳定吞吐。
统计为未padding输入暴露，不是实际执行FLOPs或生成token速度；不含初始化/评测/收集。
其他耗时逐rank用total−preparation−actor求出再取最大，不将不同rank阶段最大值相减，
也不把混合审核/同步/保存开销称纯I/O。后台worker耗时不叠加到step耗时。
报告：`context-input-throughput-diagnostic-v1.json`。

完整Stage1独立收集完成后可直接复用：

```bash
python tools/jev/context_input_throughput.py \
  --collection .local/decision-objective-research/context-admitted-full-npu-v1-collected \
  --proof docs/engineering/context-admitted-full-npu-v1.json \
  --output .local/decision-objective-research/context-full-input-throughput-v1.json
```

当前尚无全量收集结果，未执行上述全量命令。第11步起区分保存步与常规步，
至少32常规步才提供first32/last32窗口；不将短前缀外推为稳态性能。
原collector自身的全量时长统计保留，此工具补足输入token及音频暴露分母。

本报告补充同日 `context-training-efficiency-review-20261114.md`，不替换历史计时。
用户主动询问触发只读现场检查；175 远程时间 2026-11-14 13:02:48 UTC。

## 本次实测

正式初始化的全日程前两步容器 `1bec47078c3a` 已 exit 0、非 OOM。
独立 collector 已核验完整人口日程、四 rank checkpoint、两份联合策略与
128 个 BF16 执行张量及原始 marker。证据为
`context-admitted-prefix-npu-v1.json`。这是两步执行与保存验收，不是全量训练或收益结论。

配置：Qwen3-ASR-0.6B、BF16、LoRA q/v rank4 alpha8、共享 Transformer 概率头、
DP4/FSDP、global128、每卡 micro8、累积4；全学习人口40513、两轮634步。
本次复用特征缓存，尚未使用完整本地音频镜像，命中前仍读取并校验 Ceph 音频。

| 分项 | 第一步 | 第二步 |
|---|---:|---:|
| 输入后台服务 | 3.464 s | 4.421 s |
| 前台等待输入 | 3.465 s | 0.000066 s |
| 输入准备（四 rank 最大） | 4.040 s | 0.479 s |
| actor 更新（四 rank 最大） | 19.809 s | 15.531 s |
| 完整更新（四 rank 最大） | 26.207 s | 18.360 s |
| 输入缓存命中 | 128/128 | 128/128 |

后台服务与计算重叠，不应再次加到完整更新时间。第二步约6.97训练槽位/秒，
actor约占84.6%；每rank计算的 total-preparation-actor 最大约2.36秒，
包括审核、组装、联合策略保存及同步，未进一步分项，不能全称磁盘写入时间。
本次每步保存，完整运行配置为每64步及终点保存。
容器墙钟94.30秒，两步完整时间约44.57秒；其余约49.73秒包含启动、初始化及收尾等，
没有细分计时。反复短跑会放大这些固定成本，不能据此外推常驻训练吞吐。

旧 Final60 CE 全量实测普通步均值24.734秒，其中准备9.048秒、actor15.620秒；
末64步完整38.985秒。新旧输入长度、头和目标不同，禁止用两者计算可归因加速比。

## 当前周转阻塞

现场4–7卡无运行进程；0–3仍有既有Omni服务。此刻空闲不代表整个等待期间一直空闲。
旧训练终态审核 `98629d12` 自10:20:35运行至本次检查约162分钟，仍非终态；
旧全量selection尚未产生，不把审核时间算作actor训练成本。
后续读取原镜像任务进度为35840/40513（88.5%）、记录耗时2297.07秒；
全量本地音频镜像和特征缓存尚未一起验收，原driver继续，未重启任务。

## 瓶颈判断与执行顺序

1. 已确认输入路径影响旧全量稳定性；本次热缓存第二步实现计算与输入重叠。
   完成已有本地镜像/特征缓存流程后，在同一常驻训练中取热身后至少32步窗口，
   报告p50/p95、有效样本和token吞吐、前台等待、峰值显存、保存成本。
2. 源码确认固定投影将Linear拆成128行、8块的多次bmm，micro8、长度约3k时
   每次投影约24次bmm；head逐请求执行，验证包含设备到主机同步，loss也逐样本构建。
   Backbone按Linear细分FSDP。它们都是profiling重点，尚无各项耗时占比。
   固定小块和逐请求头服务于已验收的数值一致性，替换需要配对输出/梯度/恢复校验。
3. 同global128比较micro8×4和micro16×2，先覆盖长序列并测显存。
   micro32×1为后续候选。旧实现曾micro32 OOM，不能据0.6B参数量推断当前一定可用；
   LoRA减少训练参数和优化器状态，不消除长序列反向激活。
   当前日程和入口固定micro8，更改必须版本化，不能原位改正在验收的配置。
4. 依据profiler优先替换最大成本项：合批头/损失和合并校验，较粗粒度FSDP或
   冻结骨干复制加可训练参数同步，投影kernel优化。保留已验收基线，不同时修改所有项。
5. 实验周转也要优化：复用不可变输入与审核证据，后续CPU审核可与NPU评测并行，
   联合验收后发布结果；保留旧任务，避免慢审核导致重复重跑。两步正确性验收完成后
   转入常驻全量运行，性能测量并入该运行，不继续无限追加两步探针。

RL目前仅有实际音频单更新验收，尚无连续在线rollout→更新吞吐。
G16应复用每条音频一次前向所得概率采样，不能当成16次音频前向；
正式RL另计rollout、actor、策略切换、参考策略和评测成本。
本轮未修改已冻结训练源码、训练配置或重启作业；只完成现有终态收集与分析。
