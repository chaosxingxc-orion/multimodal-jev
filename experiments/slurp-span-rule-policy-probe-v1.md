# 三种不可变策略的规则迁移开发探针

目标是确定现有 ASR0.6B Span 基座和已训练 temporal adapter 是否能消费显式
规则，而非用重新制作的数据默认宣称决策 CoT 有效。

固定 balanced-rule-controls-v1 seed17 的首个 batch/基础规则，共九组，
三种 arm × 两种极性 × 四种事实组合 = 216 个请求、27 条 learning 音频。
选择发生在模型推理之前，无测试内容、无权重更新、无自动选优。

比较策略：

- 原始 Qwen3-ASR-0.6B，无 adapter；
- `slurp-native-spans-bf16-v1/run/adapter`：先前三角色 SFT，当前 RL 的共同起点；
- `slurp-span-verl-vllm-resume-v1/run/policy-1/adapter`：上述起点加两次 RL 更新。

这不是等预算继续 SFT 对照，不能把 SFT→RL 的差值全归因为 RL 算法。
这些策略此前已经接触过 learning 来源，不能称作未见音频泛化；新规则问题
只用于开发诊断，也不是论文基线复现。并列角色上的两次 RL 更新很少，阴性
结果不证明所有规模、数据量或后续算法都无效。

固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
物理 NPU5，BF16 基座及未合并 adapter，HF eager、完整音频+问题模板，
一次前向多位置 typed 读出，无文本自回归或转写→文本模型。
本实验尚未验证这些新输入的 vLLM 一致性，不准入 RL。

按规则报告 Final accuracy、balanced accuracy、完整 Span accuracy，另报告
structured/reordered 相对 Final-only 的配对 wins/losses，R/NOT R 的预测翻转及
两者同时正确比例。“模型事实预测 + 确定性执行器”与 Final 输出分列，
一致率不是准确率；gold 事实只用于评分，绝不代入可部署执行器。

原始预测、输入/波形/特征身份和概率逐条保留，独立 verifier 重算 argmax、
概率归一化、数据覆盖、模型/输入身份及全部指标。更广跨任务泛化需独立来源
和任务划分，正式 RL 改进需等预算 SFT 及多 seed 对照。

## 完成结果

三项均正常结束并独立复核。三问 Final accuracy：base 50.00%、SFT 48.61%、
RL2 50.00%，没有稳定规则迁移收益；SFT 预测事实加规则执行器为 83.33%，
提示直接 Final 组合仍是明显缺口。完整数值及限制见
`docs/engineering/span-rule-policy-comparison-v1.md`。
