# MINDS 原生标签描述迁移协议与完整音频转换

2026-11-15：原生标签描述诊断协议已冻结，全部8168条输入编译和8k→16k转换完成，
独立复读/重算全部通过。固定镜像CPU processor检查已exit0并完成全部8168条，
独立缓存/长度审核也已exit0，全部缓存张量审核及三个长度边界输入重新编码通过。
本次没有模型前向、目标集调参或迁移成绩；不改变SLURP GRPO→RLOO→FSC→TEST队列。

## 任务协议

入口为 `configs/tasks/minds14/native-descriptor-transfer-v1.json`，SHA256
`505cad6e299ffcbdfb39d751ed448c7409ae4bc762ccbf1c411336b9be08bd9f`。
使用上游全部8168条TRAIN作为项目冻结迁移人口，不称官方TEST、论文划分复现或AudioJev子集。
所有录音使用同一14候选菜单与现有English系统前缀；候选描述仅将原生标签下划线替换为空格。
不使用项目临时语义卡、不按样本gold挑题、不向actor提供语言标签、转写或原始含类别路径。

初始策略固定为已完成SLURP Stage1终点，后续策略仅由完整SLURP选择集选出GRPO checkpoint，
与FSC/SLURP TEST复用同一来源pair；禁止目标阈值、提示词及checkpoint调优。
阈值0.5，top1并列取固定菜单首项。报告完整人口的集合F1及配对差异、top1、macro-F1、
精确集合、NLL/Brier、空选和多选率；分别报告14语言/方言、14变体等权和12语言等权结果。
12语言汇总中三个English变体按录音合并。14候选的概率分数不与60/31候选分数混合求平均。

原始采集语义边界尚不完整，原生标签描述诊断不能替代语义规约或复杂Q的泛化研究。
没有说话人划分证明、预训练无重叠证明或总体显著性结论；详见doc65和冻结协议的limitations。

## 实际全量处理

原始8k音频保留于 `.local/decision-objective-research/minds14-audio-v1`。
统一请求包为 `minds14-context-inputs-v1`，8168条均使用同一渲染文本；
派生16k包为 `.local/decision-objective-research/minds14-context-inputs-16k-v1`。

转换使用scipy 1.13.1的resample_poly，up=2/down=1、Kaiser beta=5、constant零边界，
输出float32 FLOAT WAV。每条样本数严格翻倍、时长不变；不裁切、归一化、音量增强或裁幅。
最大幅度为1.4390668869，FLOAT编码保留该值，不以整数WAV静默裁幅。
总时长24.5972872222小时。原始WAV和冻结生产加载器均保持不变，本地提交保留fsync。

独立工具 `tools/reports/verify_minds_resampling.py`重新读取原始和派生的全部文件，核对
顺序、编码hash、波形hash、时长及请求文本，并用规定滤波参数逐条重算，数组精确相等。
这是独立于写入器的完整复算，仍使用同一个scipy实现，不声称独立滤波器实现验证。
音频转换及输入编译相关12 tests通过（0.79秒），全量复算另有真实数据证据：
[独立核验](minds-resampling-independent-v1.json)。

- 16k MANIFEST：`07586924872acba72ce0b58c650c6c6e609ab1d98c67eaceb231c5b3081881e6`
- 16k inputs：`973c4ed321bb79546037b6437ce6c0fca062a9d267b0e9febfafdc6a6c81cf1e`
- 独立核验：`b02c51874c1643a90cda616ca8adb585db644fdda6fe76fa1307e5a23fc1b0e9`

## 固定镜像CPU检查

175本地/data空闲约3.4TB、32.65%，固定52ba94e1镜像存在。CPU包及音频共8542文件
传输后逐个hash核对通过。容器 `9e3ffe5ebc6f40b48248eb1d7c591ca87cb8dd80bf3e84c8f8818acecd38d3c9`
现已exit0/非OOM；2 CPU、8GB、network=none、无NPU设备、无模型权重，输入只读，热写/data。
入口 `tools/reports/preflight_minds_context_inputs.py` 复用冻结CachedContextPreparer与processor，
检查每条14个决策位置、BF16 features、4096上下文上限及完整波形预处理收据；不截断超长请求。
同时保存实际processor/feature extractor源码和配置，后续核对长音频的完整特征覆盖。

启动收据：`.local/decision-objective-research/minds-processor-cpu-v1-{launch,start}.json`。
准确docker wait会话28910已返回0。终态及完整输入收据收集到
`.local/decision-objective-research/minds-processor-cpu-v1-collected/`，传输逐文件hash一致；
实际config/runner与封存包一致。完整编码用时239.60秒（不含启动/认证），最大1536 tokens，
最长1086400个16k样本，即67.9秒；357条超过30秒，最长有效特征6790帧。
本地复算全部收据的有效帧均等于原始样本数整除160；该收据分析尚不替代缓存张量复读。

实际processor源码默认truncation=False，n_samples=480000不是强制30秒截断；
feature extractor按hop_length=160产生帧、n_window=50补齐，并以有效帧计算音频token数。
实际源码/config已收集于 `minds-processor-source-v1`。完整独立审核还检查掩码连续、
补齐帧不计音频证据、14个决策位置、音频token数量、BF16、每条原始音频及缓存hash，
并对最长/最短/非整帧输入重新执行processor。边界核验4 tests通过（1.84秒）。

接续驱动631275已完成，准确审核容器
`8c910c811c8e27a6867550c292e667dce88167a630f10d97067ef8dd278a1823`，
源码为 `tools/reports/verify_minds_processor_cache.py`。同镜像2CPU/8GB无NPU，
原输入/缓存只读；已exit0/非OOM并自动收集审核及实际执行源码，hash逐项匹配。
状态入口 `.local/decision-objective-research/minds-processor-audit-cpu-v1-status.json`。
完整独立证据为 [minds-processor-independent-v1.json](minds-processor-independent-v1.json)：
全部8168条输入/缓存hash、张量、14个决策位置、连续有效帧、音频token数量通过，
最长67.9秒、最短1.365375秒及非整帧9.045375秒的实际processor重新编码与缓存精确相同。
这是全量缓存审核加三个样本的独立processor重算，不是全量processor双次重算或模型成绩。

## 迁移推理入口准备

新增 `tools/reports/minds_context_runtime.py` 及 `run_minds_context_pair.py`，
复用原BF16 actor/head与FixedContextRollout；原SLURP runtime文件和运行包未修改。
入口验证完整协议hash、全量输入、两个不同CPU终态、实际审核收据与公开摘要一致性，
拒绝改动阈值协议、失败终态、缺失波形覆盖及替换审核结果。
未来只读取已有缓存，cache miss直接失败，不在模型推理时重做预处理。

已对真实全部8168条输入执行本地证据绑定校验，见
[输入准入v2](minds-context-runtime-input-admission-v2.json)。
256条HF/vLLM面板由最长/最短各16条与固定ID hash顺序补足，独立于目标标签。
双策略均需先完成各自面板精确一致性，再开始任一策略的全量推理；来源策略仅接受
SLURP完整选择曲线封存的initial/selected pair。

28项CPU检查通过（31.96秒）：真实本地输入证据的篡改拒绝、概率/输入身份、面板完整性、
持久化记录和双策略失败停止。模型和子进程执行用替身，不能当作NPU路径验收。
尚缺来源pair实际绑定、部署包/独立模型输出收集、真实14候选NPU对齐和推理、目标评分，
没有MINDS NPU任务派发或迁移成绩。原SLURP基线与队列不增加前置条件。

## 部署、独立收集与评分实现

后续新增 `prepare_minds_context_package.py`、`launch_minds_context_pair.py`、
`collect_minds_context_pair.py`、`score_minds_context_pair.py` 和 `run_minds_context_experiment.py`。
这些工具在tools/reports中，原SLURP工具/运行包保持不变。部署复用175已审核16k音频和缓存，
按原请求路径只读挂载；来源pair与FSC/clean TEST一致。沿用同一设备锁、固定镜像、
单卡7、BF16 runtime；不在当前训练运行中抢占设备。

独立模型结果审核检查实际镜像、设备、挂载、环境、完整输出与逐条持久化预测。
评分器仅在该审核成功后解析目标inventory标签，按转换ledger核对原始/派生音频身份，
使用ContextEventMetrics固定14候选/.5阈值；三个English变体先按录音合并，再算12语言等权。
同时报告14变体等权、录音加权、各变体配对差异及修复/退化数量；无限NLL不当作0平均。

新增验证：部署/收集18tests通过（2.33s），评分9tests通过（1.89s），host编排10tests
通过（1.97s）。测试覆盖篡改拒绝、完整性、语言加权、评分顺序和失败不重试；涉及模型、
来源包与SSH的夹具为替身，不是固定镜像实际NPU验收或目标成绩。
原目标标签本轮没有逐条读取；评分契约在推理前封存于
`.local/decision-objective-research/minds-context-scoring-contract-v1.json`。

真实输入host计划校验通过：
[minds-context-deployment-plan-v1.json](minds-context-deployment-plan-v1.json)，
1091依赖绑定，spec为 `.local/decision-objective-research/minds-context-arithmetic-v1-spec.json`。
完整SLURP来源pair当前尚不存在，未构造占位或替代checkpoint，没有封存虚假的实际来源运行包。
尚待固定镜像检查新增工具、来源pair完成后实际打包/部署、真实14候选对齐及推理收集评分。
没有新增MINDS NPU派发或自动接续队列，原GRPO→RLOO→FSC→TEST链不变。

## 固定镜像检查与条件接续（后续状态）

固定52ba94e1镜像CPU容器 `3ed02eadfc34f3bbc368f4190b16ac0131ae1ac40296a2529addfc76c09b13d5`
已exit0/非OOM，76tests全部通过、无跳过，pytest用时29.19秒。传输和实际执行1121源码/
测试文件hash一致，日志、JUnit、终态、runner及manifest已独立收集；
证据 [minds-tools-fixed-image-cpu-v1.json](minds-tools-fixed-image-cpu-v1.json)。
该检查无NPU/模型权重；包括部署、收集、评分、host/pair流程、概率校验及音频边界，
来源/模型/SSH相关夹具仍为替身。此前真实本地输入绑定的5项检查不包含在该76项中，
真实全量processor与缓存审核另有已经完成的固定镜像证据。

条件接续driver644727已启动并由ps确认存活，状态 `waiting_exact_clean_TEST_owner`。
启动前核实原TEST driver600506存活且实际持有前序锁；新队列绑定1097依赖，
等待原 `slurp-context-test-arithmetic-v2.lock`，仅前序完整成功后才运行MINDS封存实验。
没有修改原四个驱动或给原基线增加前置条件；只是把MINDS接到clean TEST之后。
不读取FSC/TEST分数来选择模型，来源pair仍只由SLURP selection选择；失败保留句柄，不重试。
脚本 `.local/decision-objective-research/continue_minds_context_v1.py`，收据
`minds-context-arithmetic-v1-driver.json`，状态 `minds-context-arithmetic-v1-queue-status.json`。
当前只是等待队列，不是实际MINDS NPU已派发，更不是迁移分数。
