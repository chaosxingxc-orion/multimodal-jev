# FSC学术参考与配置迁移的同人口比较

> 当前接续更正（2026-11-15 16:41 UTC）：原 arithmetic-v2 FSC/TEST 后继已因 GRPO
> selection 审核失败停止，driver599824/600506 及下文早期等待描述均属历史。
> 新统一后继 PID1330344 已实际启动并等待准确恢复 PID1328893；仅在完整 GRPO
> 曲线完成后接原 RLOO、比较、FSC 与 TEST/MINDS。原 spec 和旧失败记录保留，
> 权威接续状态见[恢复记录](recovered-research-successors-v1.md)，尚未执行 FSC 迁移。
> 本次重新认证当前spec绑定的官方模型、3793音频、预测与归档终态全部通过，
> 见[fsc-arithmetic-reference-revalidated-v1.json](fsc-arithmetic-reference-revalidated-v1.json)。
> 不重新评分或启动推理，尚无Jev迁移成绩；FSC官方参考仍是已训练模型，不能称零样本。


> 2026-11-15当前接续：新增独立只读入口
> `tools/reports/compare_fsc_context_reference.py`，仅比较当前context初始/SLURP选中
> 两策略的native top1，不恢复下文已取消的14臂矩阵。driver589596等待现有
> FSC队列582739，1102项身份封存；不会派发新推理或增加SLURP发布前置项。
> 当前尚无context FSC实际预测或迁移比较成绩。

现有SpeechBrain容器已在历史清理中删除，重新docker inspect明确返回no such
object；原始终态、独立对齐证明、模型、预测仍在。新入口明确采用归档来源，
校验此前完整对齐文件hash、终态/运行身份、所有模型文件、3793条音频字节及
原预测hash。实际26项归档文件/模型身份和3793音频复验通过，无gold重评分、
无新推理；不把归档核验声称为当前容器检查。

报告仅在现有冻结迁移成功后执行；先重新调用现有实际runtime/输入/策略审核与
评分入口，再核对外部参考对应同人口、同定义，计算两策略与参考的native top1
配对差值、共同正确/错误及各自独对数量。使用10说话人聚类、seed941/2000次
bootstrap；不提供外部概率指标，不选择目标域checkpoint，不作同预算算法排名。
SpeechBrain是FSC任务监督模型，Jev是SLURP训练后的配置迁移，训练条件不同。

64项host CPU测试通过/2.48秒，覆盖新两策略比较、归档拒绝、前序失败不评分及
既有迁移评分接口；集成中的模型/runtime为替身。实际资产核验另有独立证据。
见[fsc-context-reference-host-v1.json](fsc-context-reference-host-v1.json)。新工具位于
`tools/reports`，不进入当前按`tools/jev`封存的推理包，不改1071项原队列pins。

> 2026-11-13状态更正：本页控制器启动/等待描述为历史记录。已沿v1→v7核对两条替代链，
> v7在转向全量SLURP两阶段训练时被显式停止，原PID均不存在；旧矩阵根目录当前不存在，
> 没有完整迁移比较结果。不得重启取消队列或把旧progress当作存活证据。
> 见[状态核对收据](fsc-history-reconciliation-v1.json)。SpeechBrain原3793预测hash仍一致，
> 已有3774/3793分数保留；此次未重新评分或执行推理。


2026-11-12。已有SpeechBrain官方权重3774/3793原生联合命令结果与本轮FSC
迁移3793录音逐条音频对齐。新增报告在整个七策略矩阵验收后复用已有预测，
不派发新推理、不使用目标错误选择checkpoint或模板。

报告覆盖step0及D/Q/P各step6、step12的direct/structured原生模板，共14组。
每组报告Jev和SpeechBrain绝对准确率、Jev减参考的差值、两者都对/都错及各自
独自正确数量；使用相同录音配对，按10位说话人聚类bootstrap，seed941、2000次。
所有比较共享speaker抽样，保持原录音权重；这不覆盖训练种子波动、跨说话人
同命令相关性或多重比较。额外报告外部参考macro-F1、逐类/说话人及组件准确率。

SpeechBrain未解析或本体外输出留在完整分母计错，不删除、不修复。
外部模型只有原生语义输出，不复制其成绩到Jev的重排/重绑定视图；概率NLL
也不跨不同读出接口比较。该模型用FSC任务监督训练，本轮Jev为SLURP训练后
冻结权重的配置迁移；同音频仅提供能力定位，不构成同训练预算算法优劣证据。

执行入口先调用完整矩阵评分器，认证实际容器退出、协议前瞻绑定、所有后端
输入/概率验收，再重新认证外部参考容器、模型文件、音频与预测哈希。
复算外部joint正确数须与已有独立复核一致。完整矩阵未完成时不得评分。

29项测试覆盖完整人口、语义/音频错配、非法输出、缺失/重复预测、错误汇总、
未完成矩阵以及前序失败不得评分。9/1不等大小speaker的合成样本中参考准确率
0.9，speaker区间[0,1]，避免将10条录音当独立样本导致过窄区间。
见[合成测试记录](fsc-external-report-fixture-v1.json)，不是实际FSC新分数。

只读控制器PID3650919已确认实际argv存活，等待原FSC矩阵控制器PID3565864，
轮询600秒。原队列和协议未修改；报告在独立进程生成，不阻塞后续训练使用设备。
[启动记录](fsc-external-report-controller-v1-launch.json)保存冻结源码和配置身份。
实际比较尚未产生，迁移正收益、负迁移或零收益均须完整披露。
