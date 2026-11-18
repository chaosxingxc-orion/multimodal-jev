# 简单 Stage2 启动、审核与恢复

2026-11-13。入口已实现、本地模拟主机验证通过，真实NPU生命周期尚未验收。
`configs/experiments/slurp-simple-sampled-rl-v1.json` 的正式派发开关仍为false。
本页记录执行方式，不构成放行证明。现有Stage1不改快照、不抢占。

## 当前候选 rc2（2026-11-13 19:29 UTC）

恢复第三步checkpoint后继续第四步的路径已接入同一加载器和连续训练循环。
正式准入增加独立`resume_update`证据，要求同一连续来源、精确master/Adam恢复、
128条恢复概率及一批真实更新/导出/概率审核。本地相关回归141项通过，非NPU验收。
1360份源码/配置封存为`.local/simple-final-v1/stage2-release-candidate-rc2-source.tar.gz`，
SHA256 `a630ddaa1af2257eb86936720a7b6992042d1869dc88112aa6e39451330c9b86`。
固定镜像CPU容器`1680ac442df4f4065ae1cf7f9e8f0d3935b966b59bf1fd9f312d01debe7aa9a1`于19:39 UTC按间隔观察exit0，304项通过、无跳过。
真实连续NPU与恢复再更新仍缺，formal false。rc1归档保留，工作树不再等同rc1。
本地归档、远端1360文件与runner已逐hash一致，两份配置与rc1相同。
精确17文件测试清单304项已全部执行；
`collect_stage2_release_candidate_rc2.py`已独立核验全部304项、
零跳过、只读源码挂载及完整远端身份。执行收据为
`slurp-simple-rc2-cpu-acceptance.json`，来源收据为
`slurp-simple-rc2-source-transfer-check.json`。本轮只归档CPU结果，未派发NPU任务。

## 历史候选 rc1（2026-11-13 18:35 UTC）

当时源码及两份权威实验配置封存为一个候选，不再以不同时间的零散测试相加
代替同版本验收。1355份Python源码和2份配置的逐文件hash见本地
`.local/simple-final-v1/stage2-release-candidate-rc1-manifest.json`。
归档为`.local/simple-final-v1/stage2-release-candidate-rc1-source.tar.gz`，SHA256：
`6194e16d3dce93c3aed982bd486ed9472adff7e924e218d323786fdbd1fabe5f`。

固定镜像CPU容器
`53ac0a9e083b6b7f53311affa8089fe59ef1caf1af8f509278f0985822855c44`
运行16个测试文件，覆盖主机入口、阶段初始化/恢复、共用循环、全量selection、
末批/终态审核与RPC。源码只读，运行写本地/data，无NPU设备；18:44 UTC到期确认exit0，275项全部通过、无跳过；远端1357份源码/配置、归档和runner均独立核对一致。收据为
`.local/simple-final-v1/stage2-release-candidate-rc1-cpu-verified.json`。来源替身和合成概率的测试不构成真实音频NPU验收。

候选中的正式派发开关仍false。不能直接把候选称为已发布工程；后续实际NPU
运行须记录所用快照，并与候选逐文件对齐。有必要修复时建立新候选，不原位改此归档。
最终Stage1及其完整评测、真实连续NPU/恢复和正式放行证据仍必须满足下列条件。

## 固定前置条件

- 最终Stage1累计634更新、四卡checkpoint独立审核及完整9226 selection验收。
- 同算法常驻DP4连续更新、独立审核、新进程恢复及恢复后再更新的独立审核。
- 镜像、模型、processor、BF16 LoRA、数据release、日程与reward配置发布。
- 175物理0–3卡作为actor、4卡作为TP1服务，启动服务前五卡空闲。
- 运行数据写入本机`/data/xingchao/multimodal-jev/simple-final-v1`；Ceph只读。

## Stage1收尾与连续验收交接

每条命令的后继都必须等待前一具体句柄真实成功退出，按至少600秒观察；下面不是
可直接连续执行的shell流水线。未完成634步时不启动终态审核。

1. `.local/simple-final-v1/launch_terminal_checkpoint_audit.py --mode stage1`
2. 审核成功后：`.local/simple-final-v1/collect_stage1_acceptance.py --mode checkpoint`
3. 收集成功且4卡空闲后：`.local/simple-final-v1/launch_stage1_selection.py`
4. 完整HF/vLLM评测成功后：`.local/simple-final-v1/launch_stage1_selection_finalize.py`
5. 独立finalizer成功后：`.local/simple-final-v1/collect_stage1_acceptance.py --mode selection`

这些Python入口均以`python3`调用。新收集器一次SSH逐文件重新hash，核完整终态、
原配置hash、证据图冲突与原报告字节；selection还绑定两后端评分配置、原预测hash
和单独指标文件。不会把worker exit0单独当验收，已存在不同报告拒绝覆盖。
17项本地合成检查通过；175已闭合两步前缀的1298文件图与原报告字节实际核验通过，
并确认该前缀不能通过634终态要求。真实Stage1终态及selection收集仍未执行。

0–4卡全部空闲后，可执行默认工程fixture：

```bash
python3 tools/jev/launch_simple_resident_train.py --phase service
# 到期核验实际ready后，才启动trainer。
python3 tools/jev/launch_simple_resident_train.py --phase trainer
```

trainer退出后独立审核使用`launch_terminal_checkpoint_audit.py --mode sampled-resident
--algorithm grpo_v1`；审核成功后用`collect_sampled_audit.py --algorithm grpo_v1 --resident-train`。
上述两入口位于`.local/simple-final-v1/`。随后使用
`tools/jev/launch_simple_continuous_restore.py`，审核模式`continuous-restore`及收集参数
`--continuous-restore`完成新进程状态/概率恢复证据。

**恢复范围必须分开报告：**当前`probe_simple_continuous_restore.py`恢复第三步checkpoint
并检查128条概率，不执行optimizer更新。连续fixture本身从已更新策略及Adam恢复后
进行两次更新，但这不等同“第三步连续checkpoint新进程恢复后再更新”。后者尚无真实
NPU证据；正式恢复入口已有实现/CPU覆盖，不能将概率恢复收据扩大成该能力验收。
服务在训练连接正常关闭后退出；实际释放设备仍需核查，不因trainer退出就假定五卡空闲。

连续来源独立审核成功、旧服务结束且五卡空闲后，补恢复再更新验收：

```bash
python3 tools/jev/launch_simple_resident_train.py --fixture-resume --phase service
# 到期确认新服务ready，再执行下一条。
python3 tools/jev/launch_simple_resident_train.py --fixture-resume --phase trainer
# trainer实际成功退出后派发独立审核；审核实际成功退出后收集。
python3 .local/simple-final-v1/launch_terminal_checkpoint_audit.py --mode sampled-resume-update --algorithm grpo_v1
python3 .local/simple-final-v1/collect_sampled_audit.py --algorithm grpo_v1 --resume-update
```

以上是待执行交接命令，不是已通过结果；不能将其连成无等待流水线。

## 正式启动

正式放行后，首次运行两阶段命令：

```bash
python3 tools/jev/launch_simple_resident_train.py --formal --phase service
python3 tools/jev/launch_simple_resident_train.py --formal --phase trainer
```

trainer必须等待服务真实ready并通过容器/设备/挂载认证。不能紧接服务启动就假定ready。
启动器不会停止旧任务、覆盖原目录或重复派发已有句柄；若观察超时，保留同一container ID。
`.local/simple-final-v1/observe_due.py`自动发现`stage2-*-launch.json`，每个句柄至少600秒观察一次。
它只观察，不自动重启、评估放行或继续派发。

## 完整终态审核

确认训练容器已退出且exit0后，使用唯一审核tag：

```bash
python3 tools/jev/launch_simple_stage2_audit.py \
  --training-tag stage2-grpo_v1-resident-train \
  --audit-tag stage2-grpo_v1-terminal-audit
```

该入口重新检查源容器，创建固定镜像CPU审核快照；不挂载NPU设备。
完整审核要求634步、全策略序列、四rank状态和11个完整selection点，不接受短跑替代。
审核进程到期观察并成功退出后：

```bash
python3 tools/jev/collect_simple_stage2_audit.py --audit-tag stage2-grpo_v1-terminal-audit
```

收集器复查远端所有证据hash，保留原始报告字节；不将worker exit0单独作为验收。

## 中断后的显式恢复

先确认源容器确已退出。未运行完、退出非0或缺少terminal.json，可以审核一个完整前缀；
仍在运行、状态未知或仅观察超时，均不能作为恢复来源。选择的恢复点必须有全部四卡提交、
此前每次更新证据及所有到期selection。下面的64仅是命令示例，不代表已有此checkpoint。

```bash
python3 tools/jev/launch_simple_stage2_audit.py \
  --training-tag stage2-grpo_v1-resident-train \
  --audit-tag stage2-grpo_v1-prefix64-audit --completed-updates 64
```

若第64步评测未完成，不把它跳过；应明确选择一个更早的完整恢复点，保留失败后缀。
审核成功并收集后：

```bash
python3 tools/jev/collect_simple_stage2_audit.py --audit-tag stage2-grpo_v1-prefix64-audit
python3 tools/jev/launch_simple_resident_train.py --formal --phase service \
  --resume-audit-tag stage2-grpo_v1-prefix64-audit --run-id resume1
python3 tools/jev/launch_simple_resident_train.py --formal --phase trainer \
  --resume-audit-tag stage2-grpo_v1-prefix64-audit --run-id resume1
```

两次派发必须使用同一run-id及审核tag；服务ready后才能启动trainer。旧服务若占卡，
启动器会拒绝；需要核实其关联训练确已停止后再单独收尾，不自动杀进程。
新训练tag是`stage2-grpo_v1-resume1-resident-train`，后续审核指向该tag。
恢复保留原实验配置路径、最终Stage1参考、Adam状态与累计游标，不清零优化器或另开634步。
新段不复制历史评测；独立审核沿原始文件复算历史，检查恢复初态并审查新增更新。

## 验证范围

主机入口的SSH/Docker及外部运行证据在单测中使用替身，覆盖实际命令分支、只读挂载、
设备选择、完整预算与拒绝路径。CPU真实张量恢复/下一次更新、合成9226评测另有证据。
这些均不能替代真实NPU恢复或证明任务指标提升。正式效果仍看预登记完整selection曲线，
TEST不用于恢复点、配置、奖励或模型选择。
