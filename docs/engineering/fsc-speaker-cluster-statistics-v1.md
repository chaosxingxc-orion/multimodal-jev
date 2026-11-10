# FSC 六视图的说话人聚类统计

`fsc_cluster_comparison.compare` 保留严格六视图评分，在同一录音上配对 parent 与
候选，再以说话人为重采样单位，连同其全部录音一起抽样。统计量为录音加权的
candidate−parent 正确率差，不是说话人宏平均，也不是把六视图合成独立样本。
固定 10,000 次重采样、seed17、百分位 95% 区间；各视图使用相同说话人抽样序列。

分别报告六视图的正确率、格式/schema 有效率，以及四项配对全对、六视图全对，
共 23 个相关结果。保留修复/退化数，不输出基于独立录音假设的显著性结论。
区间是边际描述，未做多重比较校正，也不反映教师候选选择或训练随机性的方差。
只有一个说话人时不生成区间；经验差值完全相同时的零宽区间也不是总体确定性。

`compare_fsc_test_clusters.py` 要求队列成功结束，重新检查每个确切 Docker 的状态，
调用现有完整 TEST loader 验证原始 3,793×6 输出，并固定 protocol/target 哈希。
全部条件满足后才生成单独附加报告，不修改运行中快照、官方分数、提示或选优规则。

```bash
PYTHONPATH=src .venv/bin/python -B tools/jev/compare_fsc_test_clusters.py \
  --root <completed-official-test-root> --protocol-hash <registered-sha256> \
  --output <new-statistical-report.json>
```

36 项统计/六视图/原有聚类测试通过，覆盖同说话人多录音、组内相关、缺失和
冲突的说话人、部分输出、多个 replicate、未完成队列和运行中容器拒绝。

## 已完成 TRAIN selection 的兼容性检查

对 `frozen-qwen3-omni-selection256-v1` 两个成功退出的实际容器重新核验，
匹配模型/源码/解码身份后运行统计。256 条录音、58 位说话人，结果文件：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-selection-speaker-clusters-v1.json`，
SHA256 `c0f18f6f909372f505d2bf6fb6e9fb7cedeccff73c9ebfd2a3eb82ce82c74ab0`。

| 指标 | parent | p1 | 差值（百分点） | 描述性 95% 区间（百分点） |
| --- | --- | --- | --- | --- |
| native 正确 | 101/256 | 224/256 | +48.05 | [+41.73, +54.66] |
| 缺候选正确 | 1/256 | 1/256 | 0 | [−1.16, +1.11] |
| 六视图全对 | 0/256 | 1/256 | +0.39 | [0, +1.24] |

这是已参与候选选优的 TRAIN selection 数据，存在选择偏差，不能将区间称为
独立确认提示收益的证据。本次没有据此更改候选、优化提示或读取未完成 TEST 输出。
缺候选原始和 p1 的唯一正确样本并非同一条：修复 1、退化 1；不要把总数相同
误解为逐条行为相同。原生收益不能替代 schema 拒识或完整 Jev 能力结论。
