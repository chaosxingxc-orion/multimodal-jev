# 原生类别、语义谓词与Span位置分别绑定

旧readout-v1仍以I_class引用原生候选，不能直接用于65个独立I_present。
本轮发布[readout设计v2](../../configs/tasks/slurp/evaluation-readout-layout-candidate-v2.json)，
仅作任务设计，不改生产encoder或训练器。

- C00–C59是原生60类描述菜单的别名，显式绑定Final输出符号。
- I00–I64/S00–S54引用独立语义卡，不借用C别名，不表示原生成员关系。
- Q00–Q07是一次调用的局部问题，显式绑定谓词别名与Span位置。
- 每个过程Span使用A=false/B=true；Final的A..Z/a..z/0..7属于另一个读出位置。
  字符相同不代表任务类别相同。单token属性仍须实际tokenizer验证。

全部17面板分别登记Direct/Q-input/Q-credit。Direct保留同面板知识只省略过程问句和
输出；Q-input/Q-credit的动作位置及成员完全相同。缺标只影响监督，不删除动作。
尾部1题面板的Final在位置1，不按固定8题构造虚拟动作。

三个布局分开控制：canonical保持原绑定；candidate_order只改原生候选显示顺序；
neutral_symbol_rebinding保持原生候选显示位置，重绑其别名/输出符号。两种干预均
保持语义Q及其定义引用不变。原生类别重排不应改写语义谓词；每个输出有明确反向映射。
任务ID及绑定表的offline字段不自动写进actor输入，不允许gold/转写/审核理由泄漏。

[编译工具](../../tools/jev/build_semantic_readout_design.py)复用已有类别布局函数；
8项相关测试通过。另从native-menu及全部semantic panels独立验证180个原生绑定、
120个谓词与17面板的全部位置，见[核验](semantic-readout-layout-check-v2.json)。
配置SHA256 `058a5ef560ac2659b41ec8d07f887bd93862f8e09c0afdcfd2402de271338d57`。
评测v2已接此设计，旧readout-v1保留历史。

尚需定稿FSC同I_present迁移契约、全量监督与actor语义字段实际渲染/上下文预算。
本轮不证明模型消费了定义，也不证明跨布局输出一致或训练收益；这些留给统一工程
验收及冻结实验。任务与监督未准入，不派发训练。
