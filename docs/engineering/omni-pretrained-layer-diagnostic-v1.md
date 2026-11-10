# 预训练 Omni 全六条逐层诊断入口

`diagnose_omni_pretrained_layers.py` 将已验证的只读 `PaddingTrace` 接到真实
Thinker text model，保留原 TRAIN6 全部六条首决策及固定 false+EOS。
每条只比较原五条件中的 audio_padding 与 text_padding，保持音频特征公共宽度、
FP32输入特征、CPU BF16 autocast、原始波形边界及模型参数不变。
有效token与位置逐位一致，只有mask排除的左padding不同，不读目标值。

## 运行约束

完整执行须绑定原精确诊断容器
`0de5e886048040b9fa5489586f4a7fcddea88085701bd1818a8a6a94946e36db` 的成功Docker终态，
并核验应用终态、intent/source快照、identity哈希、全部30条记录与已保存logits。
inspection必须由外部协调者实际inspect捕获并固定哈希；文件自身不是独立信任根。
缺少完整依赖时拒绝加载权重。入口不自动调度、不重试，也不修改现有运行。

固定镜像、模型资产与实际HF实现哈希必须与原诊断一致。12次冻结前向保存逐层
有效token张量与比较结果，并核对响应logits是否精确复现旧条件；若不复现，保留
差异，不将其自动归因于padding。每行参考tensor payload上限8 GiB，完整6行上限
48 GiB，另有序列化开销；没有优化器、梯度、生成或任务收益声明。
正式派发前还需登记CPU/内存/时间预算及精确容器，不因工具存在就自动执行。

## 预检查

`--preflight-only` 可在原诊断运行期间执行，仅验证真实processor、完整输入及两条件
的对齐，核验模型文件但不加载权重。它不要求原诊断已经终止，不代表完整执行可派发。
输入检查测试和既有trace异常清理测试共7项通过；固定镜像实际preflight另存结果。

实现源码 `8e7eaca` 的首次预检查容器
`08e3b135494578d6a8f90d6b583aa9e7b8e913fdbf409de4c375aeb7b3bf1593`，
2 CPU/16 GiB/1800秒上限，无网络/设备，模型与数据只读。
源树为 `omni-layers-preflight-v1/source`，完整模型诊断尚未派发。

## 预检查验收与依赖分支测试

上述精确容器现已退出0；独立核对完整源码、模型清单、原六条身份、response、
固定容器资源及两条件结果。六条左padding计数为 `[154,252,0,251,519,402]`，
与已验收原始前向的prompt长度逐条匹配。第三条是无需新增左padding的天然对照。
见[预检查验收](omni-layers-preflight-v1-verified.json)。模型权重未加载。

随后仅增强完整执行的 `complete_dependency` 分支：固定原 intent SHA，
核对实际容器命令、CPU/内存/无网络无设备、身份和source commit，避免重新哈希
改动的intent即可绕过原登记。预检查未经过该完整执行分支；新的分支不冒充已执行
预训练诊断。输入准备和前向代码未改变，既有快照保持原样。

包含成功完整信封、运行中/失败/OOM、命令/资源/identity/source篡改、结果缺行、
权重更新和重新哈希intent的反例；这些依赖测试明确mock logits审计，实际logits
审计另有独立测试。与trace清理、logits审计一共24项测试通过。当前原30条件诊断
尚未完成，完整模型逐层执行仍未派发，不以输入preflight替代该依赖验收。

## 原精度诊断的第三条天然对照

原精确容器仍运行中，保存logits审计已覆盖15/30条件，见
`omni-precision-train6-v1-logit-audit-partial-08.json`。第三条（row2）在既有输入
预检查中的左padding数为0；其reference→autocast→FP32 features→audio padding→
text padding四个相邻比较，完整BF16 logits逐元素相同，FP32响应logp最大差均为0。
这一无新增左padding对照与前两条text padding条件出现差异的观察一致，但尚不能
区分具体SDPA/MoE/算子形状数值机制，也不改变原0.1概率一致性门槛。
本次仅审计已保存tensor，没有重跑模型；全六条诊断和后续逐层定位仍未完成。

随后保存tensor审计覆盖21/30条件（partial-09）。第四条row3的text_padding条件
响应logp相邻最大差为0.9594621658325195，完整logits最大差1.21875；该行前面
autocast、FP32 features和audio_padding相邻条件完整logits相同。原始诊断仍在
运行，没有重放或放宽0.1门槛；这增加了padding相关差异证据，尚不证明具体算子根因。

保存tensor审计随后覆盖24/30（partial-10）和26/30（partial-11）条件。
第五条row4的autocast、FP32 features与audio_padding三个相邻条件完整logits相同；
text_padding（左padding519）响应logp相邻最大差为0.12417721748352051，完整logits
最大差为1.4375，仍超过原0.1响应概率门槛。第六条reference已保存，余下条件仍
运行中。只复算原保存logits，未重跑模型、放宽阈值或启动逐层模型诊断。

## 完整逐层执行预算

在原精确30条件容器成功退出且外部协调者核验全部保存logits后，下一诊断使用
固定镜像、16 CPU/160 GiB、无网络/设备、14400秒加30秒终止宽限。保持全部6条、
每条audio_padding/text_padding两次冻结前向；参考tensor payload上限48 GiB，
源数据/模型与原实验只读。2026-11-11派发准备时宿主可用内存约1834 GiB，共享盘
可用约1.6 TiB，预算可容纳该诊断；这是准备时观测，不保证后续资源不变化。
协调者先捕获原容器实际成功终态，再校验image/Cmd/环境/挂载/CPU/内存、源码与
Git快照、identity及全部30份logits；运行入口再次执行既有complete_dependency。
create成功后持久记录精确ID再start，失败/歧义不重试。后续实际身份单独记录，
本预算登记不表示已经派发或已经找到数值差异根因。

原精确容器实际inspect仍为running时，派发协调者已拒绝创建后续实验目录/容器，
保持依赖顺序。最新保存tensor审计partial-12完成29/30条件：第六条autocast、
FP32 features、audio_padding相邻完整logits均相同，text_padding仍在计算。
全30条件成功终态及后续逐层派发仍待完成，不用局部tensor审计代替前驱终态。

## 完整精度诊断已验收，逐层执行已派发

原容器现已在2026-11-11 02:58:29北京时间退出0。外部协调者重新inspect并核验
命令/镜像/挂载/环境/资源、完整源码与Git对应版本、全部30个保存logits及应用
结果，见[完整精度诊断验收](omni-precision-train6-v1-verified.json)。
六条样本reference→autocast→FP32 features→audio_padding的完整logits均一致。
audio_padding→text_padding的响应logp最大差依次为
`[0.14458656,0.38801003,0,0.95946217,0.12417722,0.67042732]`。
5条新增左padding样本全部超过0.1门槛；无新增padding的row2对照逐元素相同。
这是完整诊断的成功执行，不是数值一致性通过，更不是verl训练验收。

完成上述依赖验收后，逐层容器
`ad90d67b4f273cf171cf6b7fb0455a11f4367734b3a797f4fb0430935b916d28`
已按登记预算启动，实际inspect为running，当前加载Thinker。源码固定为
`03dca51ab0e784be47f4f9343e39e06dfe23c474`；容器intent SHA为
`c41837a7fa477879b69e0c29aa8d891cccde2ef60ff73a3195eaf5f40c206d56`。
见[逐层派发身份](omni-layers-train6-v1-launched.json)。它将完整覆盖原6条×2条件并
保留与旧logits的复现偏差，不改变模型参数，不把已知响应差异自动归因于具体算子。

## 首条逐层证据与 attention 分解预算

首条 row0 的两条件完整 logits 均逐元素复现原精度诊断。协调者已核验336份参考
张量的SHA、shape、dtype和有限性，并独立复算最终logits，见
[row0审计](omni-layers-train6-v1-row00-audit.json)。layer0输入相同，首个差异为
`layer-00.attention`：188166个元素不同，最大绝对差0.00390625；随后该层router
有65个token的有序top-k变化、26个token的专家集合变化。这只定位到attention
区间，尚不能归因于某个具体算子。原trace没有保存padding条件的逐层张量，
所以中间差异统计目前依赖固定源码的运行报告，不能由审计者逐项重算。

`diagnose_omni_first_attention.py`使用原row0已保存的实际输入，只加载原始第一层
norm/attention权重和pad embedding切片，按相同BF16/autocast/SDPA重建1076有效
token与154左padding两条件。固定原identity、row报告、HF实现、模型清单和协调者
实际捕获的原容器inspect；无完整模型生成、音频重处理、标签、梯度或参数更新。
保存两条件有效位置的RoPE cos/sin、input norm、Q/K/V投影、Q/K norm、RoPE后Q/K、
attention core及o-proj张量，使独立审计者可重新计算全部中间差异。

派发预算固定为同一镜像、16 CPU/16 GiB、1200秒加10秒终止宽限，无网络、无设备，
模型及原实验只读。源代码先提交，再固定快照、create保存精确容器ID后start；不
自动重试。必须先检查无padding输出与原trace逐元素一致，并检查两条件差异统计
是否复现原报告，才解释内部差异；不满足时保留失败证据，不宣称复现机制。原始
全6条逐层诊断继续执行；本项仅为row0定位，不替代全6条或NPU/verl概率一致性验收。

## 首层 attention 分解结果

固定源码`48d5504`的容器`a90aa28a7add50f685fa89471553643bcee788c6f55f98adf705afd79d30a934`
已退出0，诊断计时16.51秒。独立审计重新加载并核验24份两条件张量，确认RoPE
cos/sin、input norm、Q/K/V投影、Q/K norm、RoPE后Q/K共10个阶段逐元素相同。
第一个不同阶段为`attention_core`，161709个元素不同，最大绝对差0.000244140625；
o-proj后188166个元素不同，最大差0.00390625。无padding端点精确复现完整trace，
两条件端点差异统计复现原报告。见[独立验收](omni-first-attention-row0-v1-verified.json)。

实际mask路径不同：无padding时mask为None，padding时为`[1,1,1230,1230]`布尔mask。
从该精确容器读取的Transformers `integrations/sdpa_attention.py`表明：
`use_gqa_in_sdpa`在此版本主要根据torch版本、mask是否为None及head维度判定。
因而无mask路径传入`enable_gqa=True`，有mask路径先`repeat_kv`显式展开K/V；
这同时改变了SDPA调用形式，不能仅以“padding引起舍入”解释。仍需同形状、相同
有效Q/K/V的对照和实际operator观测，才能区分mask、GQA及kernel的贡献。
目前没有NPU概率一致性、完整六条机制修复或训练收益结论。

下一步`diagnose_omni_sdpa_dispatch.py`只读取已验收的有效Q/K/V及attention输出，
固定1076长度，比较隐式/显式因果mask、GQA/显式repeat、默认/math后端共8种组合。
每种调用用CPU profiler记录实际SDPA operator，保存输出供独立重算。先要求原
无padding端点精确复现，再解释变化；不重建padding、不加载模型、不重放任务。
预算同一固定镜像16 CPU/8 GiB、300秒加10秒、无网络设备；只读原实验与源码，
实际容器ID在start前落盘，失败不重试。本算子诊断不能直接证明完整模型修复。

## 同形状 SDPA 对照：mask/GQA 分支不足以解释差异

容器`3bd94c1051be75f8edb8dea8289470653960c6ec4c5b39fa60d1d6eea401338f`已退出0，
固定源码`b85f2bf`，8次算子对照耗时5.42秒。独立审计重新加载8份输出并复算全部
对照，见[SDPA验收](omni-sdpa-dispatch-row0-v1-verified.json)。默认后端四种组合
均调用`aten::_scaled_dot_product_flash_attention_for_cpu`，输出逐元素相同且都
精确复现原无padding端点。math四种组合也彼此相同，但与两种原始端点均不同。

所以在该row、相同有效长度和Q/K/V下，mask表达和GQA/repeat选择本身不能复现
padding差异，也不能将其归因于默认路径从flash退回math。尚需检验加入被mask
排除的位置后，序列形状、存储布局及计算分块对舍入的影响；当前未修改模型默认
attention实现，也不把math替换宣称为修复。完整六条trace仍运行，row1运行报告
同样首先在layer0 attention出现差异，两条件最终logits精确复现原诊断；该行完整
中间张量尚待独立审计。

## Padding 形状对照预算

`diagnose_omni_sdpa_padding.py`继续只消费已验收row0有效Q/K/V，保持有效因果图不变，
在默认/math后端分别执行7种布局：无padding、154左padding零值/有限随机值、154
右padding零值/有限随机值、只给K/V加左padding、只给Q加左padding。共14个算子
调用；所有被排除位置为合成值，不能冒充原完整模型的padding激活。实际mask只
允许有效query访问原始因果keys，保存每种有效输出及profiler算子身份。先核对
无padding端点与左padding端点是否逐元素复现已验收attention输出，再解释数值机制。
随机填充值与零填充值对照检验被排除值的影响，Q-only/KV-only对照分离两轴形状。
预算固定同一镜像16 CPU/8 GiB、300秒加10秒，无网络设备、无模型权重、无训练；
源代码提交后固定快照，create记录容器ID再start，不自动重试。即使局部对照通过，
仍需全模型和NPU/verl实测，不能直接宣称优化器或训练已准备好。

14条件实际容器`ae89d64b85635e1671bc1398a3df7d0e3b4ad95500d1ad840f19d42901440bba`
已退出0，主体10.39秒。独立审计重新读取全部14个输出、复算16个成对比较，见
[padding形状验收](omni-sdpa-padding-row0-v1-verified.json)。默认CPU flash后端：

- 只给K/V前面加154个排除位置，就精确复现原padding端点；Q无需增加位置。
- Q-only左padding，以及Q/K/V右padding，都精确复现原无padding端点。
- 左/右padding各自的零值与有限随机值输出均精确相同。
- math后端仍有非零差异：左padding189个、右padding167个元素不同；没有完全
  消除数值不变性问题，且其输出与原始默认后端不同。

该局部控制支持当前差异与有效K/V前存在被排除位置有关；有效Q/K/V与因果图相同，
被排除的有限填充值不是本组条件中的差异原因。低层kernel分块/归约机制尚未直接
证明，结论不能扩展到其他样本、NPU或整个MoE网络。右padding在首层算子上相同
也不保证后续MoE与完整响应概率相同；下一完整候选需覆盖原六条、保持0.1门槛。

row1也已完成独立参考张量审计：336份、784903680字节，978有效token，两条件
最终logits均逐元素复现原精度诊断，见[row1审计](omni-layers-train6-v1-row01-audit.json)。
该行首个差异仍是layer0 attention；逐层padding张量未保存的原限制仍然存在。
