# 知识卡实验的配置入口

从台账选择的提交运行根目录 `run_experiment.py`。机器路径、镜像与设备绑定仍由本地宏提供。修改下一组实验时，新建或修订它自己的操作配置；不要编辑已经派发的配置或共享基线。

| 要调整的内容 | 配置位置 | 由谁读取 |
| --- | --- | --- |
| Stage1 的 epoch、学习率日程、LoRA、每卡批量、正负例数和 EMA | Stage1 操作配置的 `design`、`lora`、`window`，以及已绑定的父训练日程 | 封包器派生每轮步数，知识卡为每个 epoch 分配输入，训练器消费冻结日程 |
| Stage1 种子、线程、集合通信超时、梯度裁剪、AdamW | [公共执行配置](../configs/experiments/card-execution-v1.json) 的 `stage1`；在新操作配置顶层用同名对象局部覆盖 | 封包时写入 `training.json.stage1_runtime`；有限验收、正式训练、优化器和断点复用同一份值 |
| 基座或微批对照 | 操作配置的 `devices`、`widths`、`spans`、`max_time_ratio`、`logit_atol`，以及公共配置的 `tower_comparison` 或 `batch_comparison` | 对照双方使用同一训练参数、数据窗口和审核口径 |
| 全 TRAIN 打分、端点评测 | 打分操作的 `seed`、`torch_threads`、`widths`、`replicas`、`scoring_ssh`；联合评测操作的 `reference_inference`、`sample_blocks` 和 `historical_comparison.experiment` | 推理进程设置种子与线程；历史对比结果字段随参考实验生成，台账检查引用相同字段 |
| Stage2 视图和训练 | 目标构建操作的 `schedule`、`reward`、`loss`、`selection`，父训练配置的 `view_execution`，封包操作的 `card_input.inference_devices` | 视图按实际轮数生成并统计全部目标；推理卡必须独占且有足够的验证样本 |

例如，在**新** Stage1 操作配置的顶层添加以下片段，仅覆盖本次实验需要的字段。默认使用操作的 `seed` 同时控制采样与训练；显式指定 `actor_seed` 可将两者分开。

```json
{"stage1": {"actor_threads": 4, "grad_clip": 0.5, "optimizer": {"betas": [0.85, 0.99]}}}
```

精确续训不能改变已绑定的训练参数。老的 JEV-210 恢复游标没有显式执行参数，只有[已登记的旧配置摘要](../configs/experiments/card-legacy-stage1-runtime-v1.json)可补全并验证；无法证明等价时拒绝续训。历史封包按原提交复现。新包调整日程、卡数或评测参考时，还要同步注册期望收据并重新做有限验收。工程规约要求的有限验收（两次训练更新、四次推理）继续执行。

新台账可以通过根目录的 `run_experiment.py` 选择 [device-pool-v1 操作模板](../configs/portable/operations/card-stage1-device-pool-v1.json)。把[设备宏示例](../configs/experiments/card-device-bindings.example.json)的字段合入本地绑定，在 `CARD_DEPLOYMENT.devices` 填 `${CARD_STAGE1_DEVICES}`，在 `CARD_SCORE_ENDPOINT_DEPLOYMENT.devices` 填 `${CARD_SCORING_DEVICES}`。已有部署对象中的主机、镜像、资源和挂载继续按本机配置。示例为同一主机各阶段的独立分配。

| 阶段 | 操作模板（位于 `configs/portable/operations/`） | 改卡后的行为 |
| --- | --- | --- |
| Stage1 | `card-stage1-device-pool-v1.json`、`card-stage1-endpoint-device-pool-v1.json` | 每卡 16 个锚点，4 卡全局 64，8 卡全局 128。封包重算轮长和末点，有限验收按原压力样本定位所在批。末点审核读取封包的设计摘要。 |
| TRAIN 打分 | `card-scoring-device-pool-v1.json` | 每张卡启动一个打分实例，同机可另选物理卡。按打分卡数分片，来源按原训练收据和容器挂载校验。 |
| Stage2 | `card-tpo-targets-device-pool-v1.json`、`card-tpo-parent-device-pool-v1.json`、`card-tpo-package-device-pool-v1.json` | 构建视图和绑定训练器都读取 `CARD_STAGE2_LAYOUT`。每卡有效批量 64 时，4 卡为 256，6 卡为 384。推理核验读取独立的 `CARD_STAGE2_INFERENCE_DEVICES`。 |
| 联合评测 | `card-paired-device-pool-v1.json` | `CARD_EVAL_DEVICES` 决定校准分片和评测 worker 数。样本块仍按数据集原人口收集。 |
| 新旧基座对照 | `card-training-tower-comparison-v1.json` | `TOWER_DEVICES` 的两组物理卡须等宽且互斥。两侧使用同一批序和相同样本数，各窗口须能整除全局批量。 |

例如将打分从 8 卡改为 `[1,3,5,7]`，只改本地的 `CARD_SCORING_DEVICES` 即可。新台账应把卡相关的期望收据（rank 数、步数和逐卡推理核验）按所选拓扑登记。冻结台账按原提交复现，精确续训沿用原卡组和分片。

CPU 自检：在有 PyTorch 和 pytest 的项目环境运行
`PYTHONPATH=src:. python -m pytest -q tests/test_card_device_config.py tests/test_card_experiment_config.py tests/test_card_contrast.py tests/test_class_logit_ema.py tests/test_210_window_recovery.py`。覆盖非连续设备封包、来源核验、实际 AdamW 更新、梯度裁剪、跨轮训练输入、全轮目标分母及历史恢复校验。设备上的数值和性能另以 NPU 验收为准。
