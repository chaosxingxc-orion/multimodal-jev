# 配置

共享配置使用JSON及`${NAME}`宏；机器路径、主机和设备映射集中在本地绑定中。
先复制[团队示例](local-macros.example.json)到`.local/config/macros.json`，填写本机地址及
所需产物身份，再从仓库根目录运行固定工具。`.local/`不入库；可用`--local`指定另一份
本地绑定。示例中的hash和空契约是占位值，必须用已验证资产填写，解析成功不代表准入成功。

| 工作 | 共享配置 | 入口 |
| --- | --- | --- |
| 生成actor/service执行配置 | [prepare-training-configs.json](portable/operations/prepare-training-configs.json) | `python -m tools.run_operations --config …` |
| 生成Ascend部署计划（不启动） | [ascend-runtime.json](portable/training/ascend-runtime.json) | `python -m tools.plan_runtime --config …` |
| 完整TRAIN日程 | [training-schedule.json](portable/operations/training-schedule.json) | `python -m tools.prepare_training_schedule --config …` |
| 归档统计→概率分层 | [report-pipeline.json](portable/operations/report-pipeline.json) | `python -m tools.run_operations --config …` |
| 终态归档轨迹导出 | [trajectory-pipeline.json](portable/operations/trajectory-pipeline.json) | `python -m tools.run_operations --config …` |
| 本地可再生成缓存清单 | [cache-cleanup.json](portable/operations/cache-cleanup.json) | `python -m tools.cache_cleanup --config …` |

常用调用：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/prepare-training-configs.json \
  --local .local/config/macros.json
```

报告动作重放还需本机已有的torch/numpy环境；本机核验环境可增加
`PYTHONPATH=src:.local/verification-python:tools/jev:.`。这不是可分发的依赖安装方式，
合作团队应使用已锁定镜像/依赖环境。配置生成工具`tools.configure`也可单独将任意共享
JSON模板解析成现有工具可读取的普通JSON，不执行训练。

每次任务使用新的输出版本；运行期间冻结本地宏副本。只改未引用宏时，新版执行配置
生成器允许复用；已发布workflow仍执行其原有来源校验。失败保留现场，不自动覆盖或重试。

`portable/`是新增团队入口；`releases/`及被历史实验绑定的其他配置是不可变证据。
不要批量替换历史路径、修改hash或把旧训练许可复制到新环境。完整配置库存及远程
训练队列尚未迁移，现有任务继续消费原包。

当前本地入口及模板的版本清单见[local-operations-v1](releases/local-operations-v1/README.md)。
从仓库根目录运行`sha256sum -c configs/releases/local-operations-v1/SHA256SUMS`核对登记文件。
这份清单不包含个人宏文件，不替代训练发布包或目标机器的运行验收。

详见[本地宏协议](../docs/engineering/local-config-macros-v1.md)及
[日常操作能力与验收范围](../docs/engineering/routine-operations-v1.md)。
