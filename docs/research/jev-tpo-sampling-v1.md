# Jev TPO 的训练集采样与预登记

本轮从 0.6B Stage1 的第 160 步出发。采样单位是 clean TRAIN 中的录音，训练动作空间是同一条音频的 60 个意图。每个动作的效用可离线枚举。对正确项，效用取视图人口除以类别数和该类录音数的乘积。其余项效用为零。这个权重对应当前视图均匀抽样下的宏召回。用 Stage1 缓存的 FP32 sigmoid 概率反算 log odds，加上按温度缩放的效用，再用 softmax 得到固定 TPO 排序目标。harness 继续读取独立 sigmoid。

## 一个训练批如何构造

沿用 Stage1 冻结难例加简单回放视图，6,248 个唯一录音。每轮无放回，每条音频一次前向。最多 25% 录音优先组织为混淆对。若 A 的真意图为 a，而得分最高的错误候选 b 与 a 打平或超过 a，从真意图为 b 的不同录音中找 B。优先选择也混淆为 a 的 B。对 a 要求 p(a|A) 至少比 p(a|B) 高 0.1，对 b 的要求反向。每条录音只承担自己真意图那一侧的 hinge。同一错分标签没有身为“最小语义对”的含义。

剩余录音中，同一官方话语、同一可信意图、不同音频内容的两条可配为不变性对。用在线模型在两个输入上的 60 个 Bernoulli 分布计算平均 JS。其他录音保留逐条目标。每个槽位同时承担全候选 TPO 投影、独立概率 NLL、真项与最强负项的概率监督和间隔。配对不改变逐条目标的权重。两个方向都以全局真实录音数为分母，padding 为零。两轮固定种子重排伙伴，所有成对关系留在同一 microbatch。

| 实际 TRAIN 构造 | 第一轮 | 第二轮 |
|---|---:|---:|
| 混淆对数 | 781 | 781 |
| Stage1 跨录音间隔不足的方向数 / 总方向数 | 1,189 / 1,562 | 1,153 / 1,562 |
| 不变性对数 | 1,269 | 1,242 |
| 仅逐条目标的录音数 | 2,148 | 2,202 |

采样结果和数值按冻结配置用 build_tpo_view 与 audit_tpo_sampling 脚本重算。25% 和跨录音项系数 0.3 是首轮实验设定。TRAIN 上的间隔违反率只回答新 loss 是否有优化空间。

## 比较与判读

两组从同一 Stage1 出发，使用相同的 34 次更新、相同录音、同一配对日程与优化器。投影组使用完整目标，对照组只令投影权重为零。这样可估计增加 TPO 投影的增量。混淆采样本身的独立增益需另有采样开关对照，本轮不归因。固定报告 Stage1、两个 Stage2 末点的 clean SLURP 及 FSC native31、MINDS native14 的宏 F1 和准确率、独立概率 NLL 与 Brier。用同一完整人口和候选定义。源任务宏 F1 增加且跨域宏 F1 均不下降，再报告达到联合目标。阈值只用源域 TRAIN，均不得用测试或迁移数据选设置。阶段性损失只作训练诊断。

本次冻结评测清单含 SLURP 官方清洗测试集 12,991 条、FSC 官方测试集 3,793 条、MINDS 冻结迁移集 8,168 条。MINDS 取自上游 train split，本项目不用其录音、标签或统计量训练或调参，因此它是项目的留出迁移集，不称为 MINDS 官方测试集。三个端点逐条使用相同输入和候选顺序。比较脚本分别报告 TPO 减 Stage1 与 TPO 减 direct 的四项原始指标差值，NLL、Brier 越低越好；宏 F1、准确率越高越好。前者可回答整套 Stage2 是否改善，后者可回答在相同训练预算和采样下投影项是否额外改善。两种差值均为固定种子点估计。

收集命令（本地宏绑定本轮两个运行目录、评测清单、Stage1 身份与输出位置）：

合作团队复现时，从已审核的 Stage1 策略、同策略的完整 TRAIN reference 和已冻结难例视图出发，在本地宏绑定这些上游身份及新的输出目录。以下命令从仓库根目录执行；`VIEW_LOCAL`、`TRAIN_LOCAL`、`EVAL_LOCAL` 是三份本地宏文件，`TRAIN_PACKAGE`、`EVAL_PACKAGE` 是对应宏声明的包目录。请为新实验使用新目录；已运行的包不可原位重建。只想复查现有采样时，直接运行第二条 Python 命令。

```bash
export PYTHONPATH=src:.:tools/jev
cfg=configs/portable/operations
python3 -m tools.reports.build_tpo_view "$cfg/tpo-stage2-contrast-v1.json" "$VIEW_LOCAL"
python3 -m tools.reports.audit_tpo_sampling "$cfg/tpo-stage2-contrast-v1.json" "$VIEW_LOCAL" --target-effect
python3 -m tools.reports.prepare_tpo_windows "$cfg/tpo-window-package-v1.json" "$TRAIN_LOCAL"
python3 -m tools.reports.native_seal_operations --config "$cfg/native-seal-operations.json" --local "$TRAIN_PACKAGE/operations-local.json" --detach
```

训练控制器依次执行有限验收、TPO、direct。独立收集到有限验收收据后即可封装和登记评测：评测控制器等待训练控制器成功退出，再取得设备租约并执行三个端点，不需要人工盯守。最终计分仍需全部评测分片成功。

```bash
python3 -m tools.reports.prepare_tpo_evaluation "$cfg/tpo-evaluation-v1.json" "$EVAL_LOCAL"
python3 -m tools.reports.native_seal_operations --config "$cfg/native-seal-operations.json" --local "$EVAL_PACKAGE/operations-local.json" --detach
```

本轮的结果收集命令如下。复现时将 `--local` 改为绑定新运行的比较宏文件。

```bash
PYTHONPATH=src:.:tools/jev python3 -m tools.reports.evaluate_tpo_endpoints \
  --config configs/portable/operations/tpo-compare-v1.json \
  --local .local/config/tpo-compare-194.json --after-evaluation
```

该命令等待已登记的评测进程终止事件。只有训练和评测均成功、各端点身份匹配且覆盖清单全部录音，才会生成比较报告。不会因为测试集分数触发重训。

在第一步更新前，以冻结的 Stage1 参考和发布的投影目标逐条核算。在 3,903 条正确项低于最强错误项的训练录音里，1,163 条（29.8%）的投影目标已把正确项升至严格第一，余下 2,740 条的目标仍未把正确项升至严格第一。这组错误录音平均有 11.52% 的排序概率质量重新分配，按两个分布的总变差距离计算。15 条并列样本的目标均把正确项升到严格第一。该测量说明了直接排序及间隔损失的作用，终点改善程度由独立评测给出。可在 `tools/reports/audit_tpo_sampling.py` 的原采样审计命令上加 `--target-effect` 复算。

2026-11-20，181 上 v0.0.194 的正式 TPO 组已完成 34 次更新。训练收据 `ops-194/tpo-collected.json` 验证通过，端点为 `34992e12d6d8b53a696fdb042dbc86a11be19b735ace6ba8718daf2203ee77fd`。这是训练完成证据；留出集收益以三端点评测的独立收集结果为准。

## 独立留出评测结果（v0.0.194）

TPO 和关闭投影的 direct 对照均从同一 Stage1 开始，各完成 34 次更新。八个推理分片、独立 CPU 评分和端点比较全部通过。每个端点覆盖同样的 24,952 条录音。两个预登记比较均达到源域 macro-F1 上升且两个迁移集均不下降的标准。

| 数据集 | 录音 | Stage1 macro-F1 | direct macro-F1 | TPO macro-F1 | TPO − Stage1（百分点） | TPO − direct（百分点） | Stage1 → TPO 准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLURP clean TEST60 | 12,991 | 80.97% | 79.14% | 81.66% | +0.69 | +2.51 | 84.54% → 85.44% |
| FSC native31 | 3,793 | 4.91% | 3.08% | 5.89% | +0.99 | +2.82 | 10.28% → 11.94% |
| MINDS native14 | 8,168 | 71.64% | 68.14% | 71.82% | +0.17 | +3.68 | 72.00% → 72.23% |

独立候选概率的质量与排序分开判断。SLURP 的 NLL 从 0.01785 升至 0.02033，Brier 从 0.00452 升至 0.00479；FSC 分别从 0.14928 降至 0.14337、从 0.03238 降至 0.03193；MINDS 分别从 0.17146 降至 0.13929、从 0.04294 降至 0.03686。两项均为越低越好。相对 direct，TPO 在三套数据上两项都更好。

在这一固定种子与训练预算下，增加投影提高了三套数据的排序结果；混淆采样本身的独立贡献没有被隔离。FSC 的绝对分数仍低，MINDS 的排序提升很小；两者是跨数据集意图判决，不能外推到新任务或断言多种子下显著。完整数字取自 `eval-194-ops/score-collected.json`（SHA256：`03f3b20bb03a92ec3988afdee1ef71f39ecce1337a901285d5f82ad88ee5eafe`），差值见比较宏输出 `comparison-194.json`。
