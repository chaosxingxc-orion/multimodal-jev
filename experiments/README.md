# 实验协议与入口

每个实验使用独立目录保存协议、配置引用和运行入口；运行前固定数据划分、指标、阈值、
随机种子、对照与失败判据。协议变更记录修订原因，不用结果反向改写预注册条件。

大规模数据、模型及完整输出保存在 Git 之外。结果摘要登记在 [wiki](../wiki/README.md)，
并保留清单、版本、证据路径以及适用的校验值。

当前协议入口：

- [FSC 冻结声学基线](fsc-frozen-audio-full.md)
- [音频 SFT 初始化](fsc-audio-sft-init.md)与 [FSC 开发评测](fsc-schema-evaluation.md)
- [SLURP 迁移评测](slurp-transfer-evaluation.md)
- [ASR／参考转写文本对照](text-evidence-baselines.md)
- [AudioJev 官方 MMAR 复现](audiojev-mmar.md)
- [音频 RL 迭代](audio-rl-iteration.md)
- [冻结参考策略 KL 数值验证](reference-kl.md)

协议存在不代表实验完成；实际结果和限制见 wiki 中带日期的记录。
