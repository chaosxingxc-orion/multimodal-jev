# D/Q/P完整学习曲线的收益归因

2026-11-12 UTC。新增`compare_intent_matched_pilots.py`读取全部三臂实际训练及
独立验收结果，逐项复核真实容器退出、固定镜像、学习来源日程、连续谱系和共同起点。
训练arm来自训练config，不根据推理模板推断；D权重的Q模板诊断不会冒充Q训练。
缺少任一步或遇到验收失败则拒绝生成完整比较，不挑选有利的部分检查点。

D主指标读取D模板；Q/P读取相同Q模板。每步同时保存Final accuracy、observed
macro-F1、正确概率均值与NLL的绝对值、相对各臂自身起点的变化。输出Q−D、P−Q、
P−D三个对比及起点校正后的变化差异；逐来源计算仅左臂正确/仅右臂正确的数量。
NLL明确标注越低越好。Q/P同输入同初始化的step0结果必须一致。

这只是学习人口的配对描述，不产生置信区间或显著性声明，不代表独立评测或多跳推理。
相同来源使用和更新数不等于相同算力。未把过程reward或兼容率的上涨代替Final收益。
数学汇总与实际入口、已有曲线/逐步比较共29项测试通过；覆盖起点差异、错误训练arm、
非匹配优化器、来源日程变化、曲线不完整以及容器/验收失败。入口测试运行时用替身，
不能作为真实三臂已完成的证据。

完整待评分配置见共享目录`intent-dqp-matched-effects-prepared-v1/config.json`及同名
仓库准备收据。尚未执行三臂评分，Q/P结果仍未就绪。工具可在全部12步验收后运行：

```bash
PYTHONPATH=src:tools/jev .venv/bin/python tools/jev/compare_intent_matched_pilots.py \
  --config /mnt/cephfs/xingchao/experiments/multimodal-jev/intent-dqp-matched-effects-prepared-v1/config.json \
  --output /mnt/cephfs/xingchao/experiments/multimodal-jev/intent-dqp-matched-effects-prepared-v1/result.json
```

后续自动收尾已接入（2026-11-12 UTC）：intent-dqp-matched-report-controller-v1，
PID3691383，等待当前Q/P控制器v3真实P12完成后复核全部36个训练/suite对并
生成13个曲线点及Q−D、P−Q、P−D差异。固定原评分协议和来源日程，未增加
训练或NPU推理。已核对全部36个现有/冻结训练配置的非arm因素一致。
21项本地入口/数学/队列测试通过，覆盖前驱失败禁止评分和不完整曲线禁止发布；
未运行新固定镜像测试。启动后精确argv确认存活；[队列证据](../../../intent-dqp-matched-report-controller-v1-evidence.json)。
当前未生成完整比较，不将已排队视为过程奖励收益。
