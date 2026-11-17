# 2026-11-10 开发集基线运行记录（归档总结）

本文件总结原始运行记录，原始 md 已移入 `archive/raw/2026-11-10-development-baselines.md`。
记录中的“running/等待”只代表 2026-11-10 当时，当前状态必须现场核验。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| FSC 数据 | 30,043 个 WAV；官方 train/valid/test = 23,132/3,118/3,793 | 数据人口；不是模型成绩 |
| FSC 音频 SFT | 47 次更新完成，合并重载误差通过预设阈值 | 工程验收；不是泛化 |
| 原始音频策略 | FSC 186、SLURP 168 个开发决策；输出格式全部无效，严格正确 0 | 早期接口失败 |
| SLURP SFT 开发评测 | 格式 168/168；非空目标正确 0/81；全 null 控制 87/168 | 只有格式学习，没有语义迁移 |
| vLLM 波形路径 | 最大 log-prob 误差 0.16854，超过 0.02 门槛 | 预处理差一帧；不是训练收益 |
| vLLM 特征路径 v4 | 单条误差 0.0000802；完整 actor 批次 0.0000566，12/12 通过 | 固定录音/身份；工程一致性 |
| SFT 起点 RL | 真实资产预检通过；尚未执行 NPU engine rollout/完整更新 | 当时状态，不是 RL 结果 |
| 文本对照 | 四组启动并产出预测，未完成汇总 | 历史诊断；文本串联不再派发 |

**结论：这次记录确认了数据人口、SFT 格式学习和 vLLM 预处理修复，但没有证明语义迁移、
RL 收益或正式跨任务泛化。**

## 失败/过期路径

- 把格式合法当作语义正确；SLURP 非空目标仍为 0/81。
- 把 waveform/feature 预处理修复当作训练加速或模型提升。
- 用 2026-11-10 的 running 状态判断当前进程存活。
- 把文本转写对照当作端到端音频主任务结果。

## 保留证据

- 原始记录：`archive/raw/2026-11-10-development-baselines.md`。
- 相关收据仍在 `docs/engineering/`：`fsc-sft-init-v1-receipt.json`、
  `slurp-sft-development-v1-result.json`、`vllm-waveform-hf-audit-v1-result.json`、
  `vllm-feature-hf-parity-v4-result.json`、`rl-sft-preflight-v2-receipt.json`。
