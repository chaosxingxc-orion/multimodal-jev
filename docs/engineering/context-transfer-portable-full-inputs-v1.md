# 全量无标签迁移输入与宏批评测

`context_transfer_portable_inputs.py`将已认证的原生目标输入转成可搬迁的完整输入包：
FSC3793×31、MINDS8168×14、SLURP clean TEST12991×60。与已有有限parity包区分，
本包完整覆盖已锁定人口；它不证明模型表现、实际缓存读取或全人口后端精度一致。

准备阶段重新执行原数据集adapter，逐条比对独立population、candidate IDs及input
receipts。输入包只保存请求、渲染hash、缓存receipt和独立全人口证据，不复制音频、
模型或gold。原audio路径保留作逻辑cache key；运行机器的cache_root由本地宏配置，
无需访问开发目录下的原审核文件。SLURP另绑定源clean发布身份。

读取器验证独立population manifest hash、完整文件集合、每条请求/渲染/候选/位置/
输入tensor身份，拒绝缺样本、额外文件、gold字段、软链接、改变发布版本；不能只
通过重新生成外层manifest替换独立证据。实际cache bytes仍由推理时只读encoder验证。

`evaluate_context_portable_transfer.py`复用锁定源码的原全量macro evaluator与输出
审核器，仅替换输入加载器。仍先核验完整SLURP selection曲线，固定共同Stage1、
source-selected GRPO、634终点三角色，再打开目标输入；不从目标分数选策略。
宏推理、完整日志、结果重建保持同一实现，额外绑定实际记录的portable manifest。
源权重/曲线实际审核、目标宽度后端parity及部署/评分仍由外层生产流程负责。

`audit_context_transfer_input_coverage.py`独立证明有限parity面板恰好来自该完整
人口：核对全人口hash、全input receipt hash、候选/processor及每个面板索引和请求。
该报告只有输入覆盖意义，不将有限面板结论扩大成全人口精度保证。

固定入口：

```bash
python -m tools.reports.context_transfer_portable_inputs \
  --config configs/portable/operations/prepare-transfer-portable-full-inputs.json --local LOCAL_MACROS
python -m tools.reports.audit_context_transfer_input_coverage \
  --config configs/portable/operations/audit-transfer-input-coverage.json --local LOCAL_MACROS
python -m tools.reports.evaluate_context_portable_transfer \
  --config configs/portable/evaluation/context-portable-full-transfer.json --local LOCAL_MACROS
```

三个模板的宏分别为TRANSFER_PORTABLE_FULL_INPUTS、TRANSFER_INPUT_COVERAGE、
TRANSFER_PORTABLE_FULL_EVALUATION。客户端配置包含portable_full_inputs及独立源曲线、
策略、macro_client身份；不接受旧inputs替换portable身份。所有机器路径集中于本地宏。

真实输入准备证据（不是NPU推理）：

| 数据 | operations目录 | manifest SHA256 |
|---|---|---|
| FSC3793 | fsc-portable-full-inputs-v1 | f633f781b89bf39f2a6f395e8f98c3af788095e523048f0679836edbbb37f9e6 |
| MINDS8168 | minds-portable-full-inputs-v1 | eb59dadd1ba349a2153892175fe5e5daba55c842412c5a1bb16db6aaac50ac56 |
| SLURP clean TEST12991 | slurp-test-portable-full-inputs-v1 | 62fc3422f7eb9ec3bcb523054e9fbfeec0a0e9b688780f19db80c17b5a45d470 |

两者实际prepare/load均成功（7552/89467终态0），独立full/parity输入覆盖报告
`fsc-full-parity-input-coverage-v1.json`与`minds-full-parity-input-coverage-v1.json`
已完成（49742/66704终态0），分别64及259条面板全部一致。

SLURP首次准备81588终态失败：旧macro-inputs-v1 manifest缺source_release，现
evaluation绑定clean-v2，校验在写输出之前拒绝。旧证据保留，新macro-inputs-v2
使用已有adapter重新认证并补发布字段，不原位修改旧包或放宽发布一致性。
实际v2重建84349已终态0，manifest为
`8e79ba336729b63d8044ea353d8b78a5449fdadee53fbd9f9a763246916a4e6d`，
population/candidate/input-receipts三文件hash与v1逐项相同；源clean-v2 hash为
`2418e2a974b529fe95a4fd41c0c99673e10dc0b698e91e1136eef127a7dd72f2`。
失败收据`slurp-test-portable-full-inputs-first-attempt-v1.json`已归档，未生成旧失败输出。
随后使用本地宏slurp-test-portable-full-inputs-v2.json，真实12991全量包准备及load
重新核验73295终态0，使用上表目标目录；没有复制模型、音频或读取TEST gold。
SLURP full/parity输入覆盖91352终态0，269条面板全部与12991完整人口对应；报告
`slurp-test-full-parity-input-coverage-v1.json`。至此三份全量输入与原面板均已实际
重新加载并核验，不代表实际cache bytes或后端推理已经执行。

验证：portable输入及原macro客户端组合30项通过2.84秒；增加receipt字段拒绝后
14项通过2.43秒；输入覆盖6项通过2.17秒。真实宏池/独立journal重建执行，模型
和原数据adapter在单测中使用替身；上表实际数据另经原adapter和独立文件核验。
完整生产部署/三角色评分仍待实际源曲线与parity验收，无新迁移成绩。
补充source_release缺失拒绝和成功绑定回归后，portable/coverage/client合计22项
通过2.83秒，包含真实宏池/日志、完整源角色校验和独立输入绑定。
