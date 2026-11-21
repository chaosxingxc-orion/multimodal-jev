# 最小实验结果保留规约

2026-11-19 用户要求：已归档的过程内容也要删除，只需锁定一套工程基座并能够从
保存内容重现结果。该规约替代“所有过程信息先完整归档后才可删除”的长期做法。

## 长期保留与临时保留

| 内容 | 保留方式 |
| --- | --- |
| 工程、镜像、共享模型/数据/特征/reference | 按内容身份共用一份；实验只记版本、路径宏和hash |
| 任务、划分、随机种子、阈值、reward/loss和启动配置 | 每次实验保留小型配置与代码版本引用 |
| LoRA、Transformer head、必要marker | 最终模型 + 源域选定模型，身份相同仅留一份 |
| loss/reward/时延/显存曲线 | 精简step/rank指标；保留微批指标向量，避免丢失聚合语义 |
| 完整评测曲线 | 保存各预登记点的概率预测、样本/候选ID、gold版本和评分配置，可重新评分 |
| optimizer | 运行中仅最新恢复点；封闭实验默认删除 |
| 逐步execution、adapter和重复processor | 消费者结束后删除；不转存大归档 |
| 原始rollout、group张量、重复rank轨迹、传输tar | 当前旧审核消费完成即删除；新基座采用在线校验和紧凑收据 |
| 中间评测checkpoint | 评测与选模期间暂留；完成后只留最终与选定参数 |

复现指标、重新加载模型推理与逐位重放整个训练是不同承诺。删除轨迹后，不再声称
能够重新执行旧版逐步全轨迹审核；保留当时审核结论、原始hash清单和精简指标。
不通过保留多份大tar来维持该能力。未完成的审核与评测不得伪造已完成状态。

## 当前 v0.0.93 接入

固定脚本 `tools/reports/minimal_training_result.py`，共享入口
`configs/portable/operations/minimal-training-result.json`，本地 `${MINIMAL_TRAINING_RESULT}` 宏。
执行顺序：生成最小包并校验 → 等待既有审核终态且其准确PID/start_ticks退出 →
复制审核收据 → 写显式删除清单 → 校验文件身份 → 删除过程文件与传输tar。
脚本拒绝symlink、保护路径、超预算、原始文件变更和异常终态，不自动重试部分删除。
普通Python后台进程每900秒检查，完全不依赖模型审核。

实际已生成：
`/srv/multimodal-jev-xingchao/operations/context-stage2-minimal-result-v0.0.1`，
14个结果文件，33,989,338 bytes，约32.4 MiB（不含随后产生的清理清单和审核收据）。
`training-curve.json` 3,004,474 bytes，保留634步×2rank指标。
该训练包不包含后续selection预测，评测产物仍独立保存。

后台记录：
`/srv/multimodal-jev-xingchao/operations/context-stage2-minimal-retention-v0.0.1-driver/started.json`。
本地宏 `stage2-minimal-result-v0.0.1-macros.json` 位于同一operations根目录。
`CAPSULE.json` 记录构建时审核pending；完成后的 `AUDIT-RECEIPTS.json` 记录独立审核已消费，
`COMPLETE.json` 才表示实际删除完成，并含删除字节数。没有COMPLETE不得声称已释放容量。

当前脚本只清理控制机上本轮已收集的过程数据与tar；远端运行目录、历史Ceph归档与
将来训练生产端的少写/滚动删除还需分别接入同一规约，不能将该局部接入称为全部上线。
v94运行评测资产及运行中的冻结包保持原身份；不为压缩空间破坏评测依赖。

验证：10项CPU测试覆盖原始文件删除后曲线重建、模型保留、幂等、审核未就绪、审核失败、
活跃审核进程、内容变化、保护路径、删除预算及路径白名单。
