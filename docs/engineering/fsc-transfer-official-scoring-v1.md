# FSC完整官方名单与正式评分入口

2026-11-12 UTC。已将本地官方FSC TEST的3793条录音、10个说话人固定为`/mnt/cephfs/xingchao/datasets/jev/fsc-span-transfer-test-population-v1/population.json`。全部官方CSV哈希与已发布FSC receipt一致；逐条核验测试音频文件哈希、规范路径、说话人路径对应、无重复音频，跨划分文件路径及说话人无重叠。未在此工具中重新认证跨划分波形去重。无类别平衡或按难度删样本，名单按官方顺序发布，只导出音频身份及说话人信息，不导出转写或gold。

新增`score_fsc_span_transfer.py`作为正式评分入口。要求一个事先固定的协议，记录task、population、固定源域checkpoint列表及选择回执、镜像、运行profile、绑定seed317和说话人bootstrap seed941/2000次。每个gate启动时的config必须已经包含协议/task/population哈希；看过该次目标域结果后补写协议不能通过。起点及至少两个后续检查点全部完成后才会提取目标标签并评分，不输出部分矩阵供中途挑checkpoint。多个检查点允许拥有相同adapter内容，以保留更新无效的结果。

每个gate独立核对实际Docker退出0、镜像、config/source、所有输入pins、HF/vLLM阶段终态及READY，随后用`fsc_backend_records`重算全候选概率一致性。目标标签从已核验的官方CSV离线提取，同时逐条重验名单、音频哈希、说话人和TRAIN定义的31合法命令。Final不由组件修复。输出六视图的各checkpoint分数、相对起点的配对变化、模板收益与RL交互、共享说话人抽样的95%区间。

50项相关测试通过，包括名单/音频/说话人变更拒绝、跨划分泄漏拒绝、协议迟注册拒绝、矩阵未完成时不得读取目标标签、后端内容校验与配对统计。尚未针对真实全量模型输出执行该入口；当前没有源域checkpoint最终选择，也没有正式目标域推理。12条TRAIN工程后端包仍单独准备，不可能冒充此完整TEST名单。

协议中的源域选择回执与历史目标域接触情况需要如实登记。程序验证固定文件与执行身份，不能从布尔字段证明研究人员从未接触目标域。项目已有历史FSC实验，不能宣称从未见过FSC。此次名单/评分建设未使用目标域分数调整训练、模板或选择checkpoint。
