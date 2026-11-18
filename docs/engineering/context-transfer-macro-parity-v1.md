# 简单任务迁移的实际候选宽度一致性检查

## 2026-11-16：顺序驱动已排队，尚未执行NPU面板

固定入口`tools/reports/run_context_transfer_parity_sequence.py`已经启动，PID1900510，
start_ticks341333918，cmdline SHA256
`1e910d72ef0400fde208b62c81d0568d09fa268d03028f25f02e213f17a09662`。
本地宏`.local/config/transfer-parity-sequence-v1.json`绑定95个文件；准确/proc身份和
全部pins本地复核通过。`operations/transfer-parity-sequence-v1/waiting.json`确认等待
tail-successor PID1895486准确进程及成功终态。前置链仍为100步服务batch对照→尾批验收。

尾批成功后，驱动再次独立审核尾批，依次执行FSC64、MINDS259、SLURP TEST269的
HF-first部署/收集/精度审核。每个数据集一次，失败即停止，无自动重试；不产生任务评分，
不启动正式训练。原封包dispatch=false保留，独立本地dispatch宏启用这三次有限诊断。
相关sequence/operations/tail回归27项通过（2.29秒，session45553终态0）；测试不代表
实际NPU成功。截至训练13:44:51UTC双rank22/100，三数据的NPU面板仍未执行。

## 2026-11-16 13:41 UTC：HF先行的实际部署包已准备

`prepare_context_transfer_parity.py`复用当前service32父包，仅新增7个评测模块，
全部1098个原Python文件逐字节未变；policy/adapter仍是共同Stage1，基座模型和
音频cache只读共享。三个包均独立进程成功导入客户端、portable输入和服务runtime，
证据包旁`source-validation.json`。这是当前本地Python导入检查，非固定镜像/NPU验证。

`context_transfer_parity_operations.py`复用同一四角色设备lease/存储/准确句柄生命周期：
preflight先在隔离6号卡执行HF，成功退出后才创建service0/1（6/7卡），客户端无NPU。
cache只读挂载到HF/client，服务只读模型和策略；整个过程不训练、不读取目标标签。
客户端按HF成功终态绑定reference文件hash，运行真实macro并独立重建日志；收集器
再核验四容器镜像/设备/命令/终态、传输全部文件、HF与两service运行模式、相同面板、
logits/prob/阈值/top1以及保存的报告。输出只证明有限面板，不声称全人口任务收益。

实际封包：

| 数据 | 本地operations子目录 | manifest |
|---|---|---|
| FSC64 | fsc-macro-parity-v1-prepared | 729d54f31d5ca0ab40c763a09d81306c71b4c0d5daa0513b26a94c20fe5d845d |
| MINDS259 | minds-macro-parity-v1-prepared | b251209b38eff8b2666aed815d1adca8909d839cee16274659b96e3467ca569a |
| SLURP TEST269 | slurp-test-macro-parity-v1-prepared | 5a31cf03e74575eaa9996a41a56a9037fe524943eebb4995443ec77142070df9 |

portable模板`prepare-transfer-macro-parity.json`/`transfer-macro-parity-operations.json`，
本地宏`.local/config/{fsc,minds,slurp-test}-macro-parity-{prepare,operations}-v1.json`。
所有包dispatch=false，未调用远程或抢占当前100步训练。40tests2.77s通过：真实封包/
hash、原包字节保护、候选宽度/设备/policy/pins错配拒绝、只读挂载和独立报告重算；
模型tensor/设备观测为替身。三个实际包完整validate通过，HF/client命令编译成功。

下一步在原效率与尾批工作释放lease后执行固定镜像HF→双TP1→独立collect；失败
保留现场，不自动重试、降低精度标准或修改任务定义。当前尚无新NPU对齐证据。

## 2026-11-16 13:32 UTC 修订：保留既有256条验收范围

旧MINDS与SLURP TEST runtime已有固定256条length/hash面板。最初v1的35/46条
长度极端面板不足以替代它，已撤回其替代资格；旧文件保留为准备过程记录。
当前v2取原256条集合与音频长度极端的并集，实际MINDS259、SLURP TEST269条；
FSC仍64条/10说话人。不得将旧v1用于降低现有验收范围。面板选择仍与目标标签、
模型输出和效果无关；新测试直接调用旧panel_indices验证完整包含关系。

v2本地宏为`.local/config/{fsc,minds,slurp-test}-macro-parity-panel-v2.json`，产物
`operations/{fsc,minds,slurp-test}-macro-parity-panel-v2`。内容digest分别为
`44a343fbb8e2a90fbe6e2cdee1790240414e0643cf4f8317d24038457f5e5a2e`、
`a22815dbaf7563a50ac7cddf365714bb520f4b9e0e723e68dbc1950cb74edcdd`、
`3542158ee356d8033b826ba65427d1a771320b4a3c1fe408b1074825aa6e3790`。

新增`context_transfer_parity_inputs.py`和portable
`prepare-transfer-parity-inputs.json`：完整原准入器复核后，只封存已选无标签请求、
原始独立cache证据、面板和文件hash。消费时同时绑定独立panel文件hash与bundle
manifest，重建请求/渲染hash、candidate顺序、marker与tensor关系；运行时无需
访问开发审核目录，cache root可本地重绑定，原audio路径仅作为既有请求/cache key
的一部分保留，不读取原始音频。明确full_evaluation_allowed=false；该包只供
一致性检查，不能取代全目标人口准入。

FSC/MINDS真实包已完成并即时load回验：operations/{fsc,minds}-portable-parity-inputs-v1，
manifest分别`421f9f6aa29991cd21cef466ca52c848855dfd111ed1f742e4b7bd176ec035be`、
`c49e7de6fd2cc2704b68a8af0e49c1f1229520a5804790a47daf7e14c4b57883`。
25tests2.24s通过：包含搬迁后删除源文件仍读取、缓存根改配，以及重新封hash的
请求/位置/音频错配、标签注入、符号链接与独立panel变更拒绝。实际cache字节与
NPU仍未执行；后文v1面板数字仅为历史记录。

后续SLURP269条便携包也已生成并load回验成功，manifest
`1cf02a02e86a98c40fc4aa544d77e65a3442108e0803c6e473716c91f82647bd`，
路径`operations/slurp-test-portable-parity-inputs-v1`。完整相关回归59tests3.10s通过；
FSC/MINDS各5个输入文件，约1.10MB/2.49MB，无模型/音频复制。独立HF与macro
配置模板`configs/portable/evaluation/context-portable-parity-{hf,macro}.json`均以
本地宏提供路径、策略、缓存、设备和连接；它们不是已经验收的NPU部署包。

`tools/reports/context_transfer_macro_parity.py`为FSC31、MINDS14与SLURP clean TEST60
提供同一个无标签面板、HF参考和实际双服务macro比较入口。仅验证工程一致性，
不产生任务分数，不选择权重、阈值或提示词；完整目标推理和跨数据集收益仍须
另行验收。SLURP TEST是源域独立测试，不能计为跨数据集迁移。

配置模板`configs/portable/operations/context-transfer-macro-parity.json`从本地宏
`TRANSFER_PARITY`读取完整配置。`--phase prepare`复用现有全人口输入准入器，
只读取已认证的请求与processor收据，不读远程cache、不执行模型。选取文本长度
和音频特征长度各自最短/最长16条的并集；FSC额外取每位说话人ID最小的一条，
最终按原人口顺序排列。样本id、音频hash、请求hash、全部输入张量身份、marker
位置、完整候选顺序和processor都被绑定。

MINDS现有无标签source_rows没有语言字段，不能据此声称全语言覆盖。本入口
不为取语言信息打开带gold的inventory，报告明确language_coverage_verified=false。
如果后续要验证语言分层，需要另行提供已认证的无标签语言映射；不按预测错误或
目标分数挑选面板。

`--phase hf`须由独立设备生命周期派发，在一个隔离NPU上加载实际BF16策略，
核对共享基座、LoRA/head和已注册token，使用原actor runtime与只读缓存编码器。
一次前向取得实际logits，在设备上计算FP32 sigmoid，保留两者及输入身份。
HF和macro必须在分离的进程中执行，不能同时把两个模型加载到同一设备。

`--phase macro`连接已启动的双TP1实例，用统一macro评测入口提交完整面板，
其内部rollout仍可切分；完成后独立重建journal和实际输入收据，再和指定hash
的HF参考逐项比较。精确门槛同时要求logits和prob相等，并报告最大误差、0.5阈值
翻转和top1变化。仅概率一致不足以证明logits相等，FP32 sigmoid舍入可掩盖差异。
任何差异保留comparison和失败终态，不自动放宽阈值或重试。

实际本地准备已完成：

| 数据 | 全人口 | 面板 | 已覆盖元数据 |
|---|---:|---:|---|
| FSC native31 | 3793 | 64 | 10名测试说话人及两类长度极端 |
| MINDS native14 | 8168 | 35 | 两类长度极端；语言覆盖未知 |
| SLURP clean TEST60 | 12991 | 46 | 两类长度极端；源域独立测试 |

产物为`operations/{fsc,minds}-macro-parity-panel-v1`；本地配置
`.local/config/{fsc,minds}-macro-parity-panel-v1.json`。面板内容digest分别
`ec60be373c206c95eca1cf35d3e2e9d4802e7d1d6ca66f3ec5697071ea1858a2`、
`bd8c379971d218041b66ce5463b02888a5e9b3a123875f5ddbb8097518ca75bb`。
SLURP产物为`operations/slurp-test-macro-parity-panel-v1`，对应同名本地宏；digest
`3e22c744a041c98c949590878417b25b4372233318dd3073418307685793cbd9`，
完整源TRAIN/selection任务定义与TEST无交集核验沿用已有准入器，实际准备终态0。

50项测试2.84秒通过：覆盖14/31/60宽度、标签变化不影响选样、长度边界、错误身份/
候选/顺序/非有限数拒绝、概率舍入掩盖logit差异，以及真实MacroRolloutPool调度、
文件落盘和独立journal审核。模型与HF输出为替身，不能作为实际NPU一致性证据。

当前只完成面板准备和客户端实现。专用封包、HF与双TP1隔离启动、精确容器收集
以及实际NPU比较仍待；报告固定runtime_provenance_verified=false和
full_population_parity=false，不以面板一致性代替完整人口正确性或效果收益。
