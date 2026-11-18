# Context probability RL：正式训练发布与收集

此页是现有工具的接入顺序。固定任务及算法仍见`context-prob-v1-baseline.md`。
代码已准备不代表真实准入完成；完整Stage1已收集，完整selection及连续/恢复/保存频率NPU验收仍待。
不得修改原`configs/releases/context-prob-v1`或排队包以跳过这些步骤。

当前执行入口更新（2026-11-15）：采用`launch_context_formal_training_v2`、
`launch_context_rl_evaluation_v2`、`run_context_formal_experiment_v2`及
`run_context_baseline_sequence_v2`。恢复镜像Ascend入口、精确只读发布依赖；
连续候选另固定短IPC。损失、数据和训练参数不变。新版编排45项CPU测试通过，
真实CPU acl导入及完整9226条加载通过；尚非连续NPU或正式RL完成证据。
当前sequence driver550847，spec为
`.local/decision-objective-research/context-baseline-sequence-v2-spec.json`，
状态为`.local/decision-objective-research/context-baseline-sequence-v2/status.json`，
498pins；comparison driver550848，状态为
`.local/decision-objective-research/context-baseline-comparison-v3-status.json`，500pins。
本次上述pins及原364冻结身份均通过。旧518938/533061因前序真实失败退出。
当前selection548477、continuous549655、replay549657、checkpoint549658均本地存活；
禁止重复派发。最近selection远程确认running/nonOOM；普通观察下限1794684560.6309571。
证据`context-runtime-launch-repairs-v1.json`。下方旧入口与驱动记录保留作历史，
实际执行以本段v2入口和当前队列为准。

## 输入与发布顺序

1. 原Stage1完整634步和对应9226 selection独立收集后，构造来源spec，包含
   `experiment`、`training_root`、`training_proof`、`selection_root`、`selection_proof`。
   三个JSON字段填本地文件路径，两个root填已收集目录。
2. `python -m tools.jev.prepare_context_stage2_source --spec SOURCE_SPEC --output SOURCE_PACKAGE`
   重新执行来源绑定，封存原Stage1 policy及其processor，并针对原policy ID派生adapter。
   输出PACKAGE.json的SHA256作为共同起点身份；两个RL算法共用同一包。
3. 原连续、独立进程恢复和interval64队列成功收集后，构造acceptance spec：
   `continuous`、`fresh_replay`、`checkpoint_frequency`各含`collection`、`report`、
   `launch`本地路径；另给`candidate_package`和`candidate_manifest_sha256`。
   该spec是输入路径，不是人为填写的passed凭证。
4. `python -m tools.jev.prepare_context_formal_package`接受
   `--source-package`、`--source-manifest-sha256`、`--source-specification`、
   `--acceptance-specification`、`--algorithm grpo_v1`或`rloo_v1`、`--output`和
   `--remote-root /data/xingchao/multimodal-jev/simple-final-v1/context-rl-grpo-RUN`。
   RUN使用小写字母/数字/连字符，RLOO路径用context-rl-rloo-RUN。
   入口重验冻结配置、完整来源及三次实际执行，重新比较恢复和checkpoint频率，
   然后生成独立RELEASE.json与PREPARED.json。不会自动启动。
5. `python -m tools.jev.launch_context_formal_training --package PACKAGE --manifest-sha256 SHA`
   默认仅校验并打印命令；加`--launch`才进入原175设备租约，使用固定镜像、4–7卡、
   DP4/global128/micro8、全量缓存。继承已验收容器的只读依赖挂载；新包和Stage1起点
   只读，训练/cache写本地盘，Ceph只读。

## 运行及收尾

`python -m tools.jev.run_context_formal_experiment --spec SPEC --output NEW_DIRECTORY`
默认只构造计划、不调用远程；加`--launch`会串联正式训练、终态收集、独立训练审核、
11个完整选择点评测及最终曲线重审。SPEC含`package`、`manifest_sha256`、
`initial_package`、`experiment`、`stage1_root`、`stage1_proof`、`template`，均使用
本地真实已收集/发布路径；template来自已验证的Stage1选择集配置。
NEW_DIRECTORY必须全新且不位于源包/收集目录中。每个命令、spec和状态都有独立记录。
不支持自动重跑部分失败链；保留原目录/launch/attempt后针对原句柄处理。
单算法完整曲线不代表整个研究目标完成，另一算法、TEST及跨数据集迁移另行执行。

启动器在远程创建前记录attempt，在start前记录准确container ID。任何不确定的
SSH start/wait结果均保留原句柄和设备锁，600秒后重试同一wait，不自动重建。
训练共享同一常驻service/actor拓扑；每步刷新策略，64、128…576、634保存恢复状态。
没有额外的中途评测进程与actor争卡，完整训练收集后按预定11点执行选择集评测。

终态后执行`python -m tools.jev.collect_context_rl_training --launch LAUNCH --output COLLECTION`，
独立归档失败与成功产物。transport完成后再通过
`python -m tools.jev.audit_context_rl_training --spec AUDIT_SPEC --output AUDIT_REPORT`核验。
AUDIT_SPEC包含collection、transport、launch、experiment、schedule、initial_policy；
transport/launch/experiment指向JSON文件，initial_policy指本地SOURCE_PACKAGE/policy。
通过后接`context-rl-selection-runbook-v1.md`的完整11点评测链；单点分数不替代完整曲线。

两算法分别新建运行目录、固定同Stage1 reference和共同输入日程。Stage1 Adam不跨阶段
继承。正式版当前只接受从task step0运行到634，未发布完整长任务中断后的恢复入口；
已有保存状态及恢复验收不能冒称该调度入口已经实现。

## 已有验证的范围

本地打包、启动器和运行绑定48项相关CPU测试通过；测试中完整Stage1/NPU准入使用
替身，SSH使用替身，真实文件复制/hash/归档与flock执行。固定训练镜像CPU回归另行
记录：固定镜像7d60454aef01独立收集79项CPU测试通过/35.89秒，exit0/nonOOM，
证据`context-formal-publication-cpu-v1.json`。尚无真实完整起点包、正式RELEASE或正式
GRPO/RLOO运行，不得把测试发布物当实验。

## 条件接续入口（2026-11-15）

`tools/jev/run_context_baseline_sequence.py`串联上述现有入口，首先等待原Stage1选择链
和checkpoint验收链的本地锁，逐一要求成功终态。随后重新绑定共同Stage1起点，
先发布并完成GRPO训练及11点曲线，再发布并完成同起点RLOO曲线。
任何前序失败、源码漂移、发布或执行失败均停止并保留原句柄；不自动重跑。

当前driver为518938，spec为
`.local/decision-objective-research/context-baseline-sequence-v1-spec.json`，
状态为`.local/decision-objective-research/context-baseline-sequence-v1/status.json`。
467文件身份已固定，现有364冻结锁和369/4/5/10队列pins保持；禁止重复启动。
本次启动的是等待接续流程，尚未实际发布正式RL包或分配新NPU。
21项本地测试通过（7队列与14实验编排），执行及模型用替身；真实本地锁/漂移
检查已测试。此前完整实验编排固定镜像CPU v2另有118项通过/36.59秒，证据
`context-formal-experiment-cpu-v2.json`；v1缺测试夹具失败记录保留。

## 两算法完整结果比较

`python -m tools.jev.compare_context_rl_curves --grpo GRPO_EXPERIMENT --rloo RLOO_EXPERIMENT --output REPORT`
对两份完整实验目录重跑训练及曲线审核，检查共同初始化与reference、全人口日程、
源码/初始资产、对应step0逐条概率及完整11点。正式配置仅允许算法和隔离运行路径不同。
报告所有匹配点、各自相对step0的变化、最终/选中点增益、退化、加权奖励与零信号比例；
实际容器wall与累计最慢rank步时分别展示。四物理卡资源小时不重复计算共卡service，
不包含评测成本，不声称单次实验统计显著或selected点可替代TEST。

条件接续driver533061（v2）仅等待原双算法序列的owner.lock及成功状态，不轮询NPU或派发
模型；成功后输出`docs/engineering/context-rl-matched-comparison-v1.json`。此文件当前
尚不存在，不能将测试合成数据当作研究结果。状态路径为
`.local/decision-objective-research/context-baseline-comparison-v2-status.json`，
源码身份469项已锁；不要重复启动或修改其依赖。26新测试+12已有曲线测试通过，
证据`context-rl-curve-comparison-cpu-v2.json`；集成中的实际模型/审计调用使用替身。

比较器v2已按正式collector目录读取`evaluation/selection/predictions.json`，
对应fixture已校正，38项CPU测试通过。旧等待驱动528969退出并归档，
原518938训练/评测队列保持不变；本次复核469项pins通过。
