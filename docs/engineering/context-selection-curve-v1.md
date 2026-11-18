# 固定选择集的完整RL曲线汇总

入口`tools/jev/summarize_context_selection_curve.py`根据冻结experiment及预测文件
逐点重新计算完整9226条selection指标，再汇总task-stage曲线。训练和运行证据仍由
独立collector认证；这个工具不会把一个自报run_id当作已经核验的checkpoint来源。
无实际完整RL预测输入，本轮仅CPU测试与CLI编译通过。

输入manifest示意（路径和digest必须替换为实际不可变产物）：

```json
{
  "run_id": "actual-released-task-run",
  "stage": "task",
  "algorithm": "grpo_v1",
  "experiment": "configs/releases/context-prob-v1/experiment.json",
  "experiment_sha256": "actual-config-sha256",
  "initial_policy_id": "completed-stage1-policy-id",
  "checkpoints": [
    {
      "step": 0,
      "policy_id": "completed-stage1-policy-id",
      "predictions": "actual-step0-predictions.json",
      "predictions_sha256": "actual-predictions-sha256"
    }
  ]
}
```

运行：`PYTHONPATH=src python tools/jev/summarize_context_selection_curve.py
--manifest <manifest.json> --output <new-report.json>`。

检查点必须为冻结的0、64…576、634的连续前缀；缺中间点、重复或乱序均拒绝。
每点固定selection身份、9226人口、60候选顺序、阈值0.5；step0绑定本Stage1终点。
完整曲线按set-F1最高→binary NLL最低→最早步选择；不完整前缀的selected_checkpoint
明确为null。保留全部点、相对step0变化、相邻下降区间、无穷NLL、未定义precision。
不把缺失NLL当作0，也不根据中途曲线重选阈值或改奖励。

14项相关CPU测试覆盖缺点/乱序/人口变化、NLL tie-break、退化和无穷值；
证据`context-selection-curve-cpu-v1.json`。CLI将从预测文件重新评分，但真实曲线
尚未运行；报告明确runtime_provenance_verified=false，需与独立运行证据联合使用。
未新增显著性阈值或成功判据；曲线变化不自动证明RL项的独立因果贡献或FSC迁移收益。

## 收集后的训练策略与评测策略绑定

现有Stage1评测会从训练snapshot重导出推理snapshot，collector分别保留
source_policy_id与inference_policy_id。二者可能相同；不能因为更新编号相同就
认为权重一致，也不能把预测内的policy_id直接改写成训练端ID。
新增离线`audit_context_evaluation_policy.py`，复用现有bundle验证和逐张量精确比较，
输入manifest的source/evaluation各绑定collection目录、proof路径及SHA、
policy_relative_path、policy_id。两端成功终态和每个policy文件必须由collector绑定，
然后校验协议、processor文件hash及完整129张量（含冻结marker）。
不加载语音骨干、不派发NPU，不修改冻结评测或排队代码。

真实旧admitted Stage1 step2→resident-switch b/policy的检查已通过，两个ID均为
54711002...、129张量精确；故意替换为step1的a/policy后因LoRA张量不等被拒绝。
证据`docs/engineering/context-evaluation-policy-binding-v1.json`。
这不是完整selection或Stage2曲线，也没有发现现有真实样本的ID不一致。

正式RL评测仍待连续验收后接入：原Stage1 launcher按Stage1终态来源选权重，
不能直接把其step64等解释为RL step64。须从对应RL run的commit绑定策略，
保留原始推理ID及训练来源，完成0/64…576/634全人口评测，再进入曲线汇总；
如重导出产生不同资产ID，需显式消费精确等价证据，不能放宽曲线起点检查。
本次只补权重绑定检查，没有宣称完整RL评测调度已实现或发布。
