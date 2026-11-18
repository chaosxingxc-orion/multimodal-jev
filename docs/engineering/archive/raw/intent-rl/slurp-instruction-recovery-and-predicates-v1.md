# 全量指令恢复与可迁移Q工程进展

2026-11-12。工程验收与任务效果分开登记。

## 已核验

四卡训练前缀实际退出0，执行冻结全量日程的前两批/256录音。
独立检查发现导出名`.default.`替换为空字符串丢失分隔点；原始运行快照和检查点未改。
`span_adapter_export`修复后续导出；CPU工具从四rank的112个FP32 master重建BF16权重，
与原错误命名文件逐张量精确一致，Adam状态/游标/256条监督元数据通过核验。
PEFT实际重载修复文件，所有112个LoRA参数精确一致。

修复路径：`experiments/multimodal-jev/slurp-instruction-checkpoint-repair-v1/run/adapter-step0002`。
新adapter身份`85120a606c313c142e11c7444a1242cb526898bf9fa82251423b891771e7b1ac`。
该变化是命名修复，不是重新训练。
[状态证据](../../../slurp-instruction-checkpoint-repair-v1-verified.json)。

`slurp-instruction-prefix-parity-v3`实际退出0；对全部256条预先固定输入重新计算HF/vLLM
全部候选与联合概率边界，最大logprob误差9.5367431640625e-7，联合比率
[0.9999990463261383,1.000001192093606]，argmax变化0。
[概率证据](../../../slurp-instruction-prefix-parity-v3-verified.json)。
v1缺少验证入口，在推理前exit2；v2完成HF但vLLM因NPU初始化后fork退出1；v3明确spawn。
各失败证据保留，数值阈值未放宽。前缀没有保存训练驻留FSDP的输出，不能将本验收
称为驻留输出重载逐请求一致。

## 已实现，尚未完成实际全量训练

`slurp_instruction_resume`锁定前驱config/launch、实际终态、四rank master/Adam、修复
适配器与独立概率验收；全日程检查要求实际完成的processor证据，不能用live进度替代。
`train_slurp_instruction_npu`新增精确恢复、跨两epoch游标和可配置检查点间隔，继续使用
原生verl actor更新。入口通过语法检查，恢复/日程/适配器及Q相关测试84项通过；
新增实际NPU恢复及完整634更新仍待执行，不能只据单元测试宣称完成。
全量processor最近实际运行观察为40,897/81,026，已进入epoch1；保持600秒普通检查间隔。

## 与最新任务方向对齐

主题问句包已撤回，新的[任务契约](../../../../research/22-transferable-decision-questions.md)固定
判决算子、配置概念定义。`semantic_spec`将规约编译进现有Span requirements；
`decision_questions`支持意图成员/语义意图/槽位存在/绑定，固定菜单与监督映射分离。
监督要求完整显式positive/negative/unavailable划分；缺标不转false或unknown。
全量现有指令日程只作为基座工程与指令能力起点，不冒称已训练这些新Q。

`slurp-predicate-menu-cost-v1`已实际退出0并独立核验CPU processor结果：固定16条学习录音，
同样的Final和规约下比较0/2/8/60个前置类别命题，定义菜单不依赖音频标签。
0/2/8个Q的最大长度为2,189/2,479/3,369；60个Q最大11,138，16条全部超出4096。
必须限制一次菜单而保持完整Final；菜单和轮换规则不得按gold选择。8Q只证明本批可容纳，
不是全人口长度准入。这是输入成本审计，不是小样本主训练。现有scenario/action描述的
语义定义质量仍须改进；长度检查通过也不代表新Q监督获准。
[长度证据](../../../slurp-predicate-menu-cost-v1-verified.json)。
