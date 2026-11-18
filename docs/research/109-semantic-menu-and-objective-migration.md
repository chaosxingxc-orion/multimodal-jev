# 语义问句菜单与损失分母迁移

本轮完成任务设计层的菜单、Stage1目标、上下文供给和评测人口算术迁移，未实现actor
renderer或派发训练。旧I_class/15面板配置及其测量证据保留为历史，不原地改成新任务。

## 17面板候选

[菜单v3](../../configs/tasks/slurp/menu-design-candidate-v3.json)分开原生Final60与
I_present65/S_present55。每族按固定seed317的ID hash排序，每面板最多4I+4S，
形成17面板；每个谓词完整一轮恰好出现一次。尾部不复制问题填满，问句按I/S交错。
这是继承的候选宽度，不是上下文验收结论。教师16面板按每族8问组织，与actor日程不同。

每个音频走相同完整面板，不查询gold、参考文本或预测主题。Direct/Q-input/Q-credit
在同一面板获得相同的原生60类描述菜单、该面板语义定义及必需政策/相邻边界；前者
仅Final输出，后两者具有完全相同Q和动作成员。原生类描述不是语义谓词充分必要定义。
不再给actor供应旧原生类语义草稿，也不默认一次输入所有65个语义谓词。

学习40513录音×17=688721请求/完整一轮；选择9226×17=156842请求/臂，
三臂规范布局470526请求/检查点。完整日程没有被128组质量审核子集替代。
实际音频/策略前向、G样本复用和评测缓存均须工程验收，不能直接把请求数当设备时长。

产物：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/semantic-actor-menu-design-v1/`。
report SHA256 `9d3c7e072b702dcea8c4968c80abc85c5a92db5a2dd2f09a64862922f7eb2d01`。
[构建工具](../../tools/jev/build_semantic_actor_menu.py)的3项测试覆盖顺序不变、65/55无重复
覆盖、ragged面板目标等价和非法支持拒绝。另从原始目录独立重建全部ID/hash排序/面板、
原生类别映射与人口算术，见[核验](semantic-actor-menu-check-v1.json)。

## Stage1与Stage2区别

[Stage1 v2](../../configs/experiments/stage1-objective-candidate-v2.json)保持原族系数，
每音频目标为mean17 Final CE +0.5/65×I的加权有效CE之和 +0.5/55×S之和。
流式请求内过程项乘17，再按真实请求求平均；空族和缺标不转移预算。
新分母不能由旧1/60正例比例或native slot缺失推断；实际极性权重仍等监督准入后计算。
不同请求间Adam更新不等于冻结参数下的全局目标求和，测试没有作这种等价声明。

Stage2保留现有每面板I/S各0.1 cap及配置问题数分母，**未静默修改奖励**。
这在17面板尾部产生不同逐谓词信用：I最后单问系数是通常四问中单题的4倍，
S末尾三问系数是四问单题的4/3倍；S只在14/17面板出现，因此完整一轮等面板平均的
族预算为0.08235294，I为0.1（全标签已准入且正确的算术假设）。
逐定义系数见[奖励暴露审计](semantic-menu-reward-exposure-v1.json)。
这不是模型效果；仅固定总cap不能称逐类型均衡。若改用全本体均衡预算，应单独登记
reward修订并对齐CE/KL/比较臂，不能边训练边改。当前Q-input/Q-credit固定同日程，
该设计仍待与全量监督一起冻结。

## 尚未完成的迁移

[评测v2](../../configs/tasks/slurp/evaluation-protocol-candidate-v2.json)更新语义算子、
17面板和成本；[上下文v2](../../configs/tasks/slurp/context-requirement-candidate-v2.json)
更新原生菜单与语义政策供给，selected_context_limit仍null。
主task、算法及reward设计入口已指向新候选；旧15面板readout布局不能被新输入继承。
还需独立native60符号绑定与语义Q引用的renderer/readout设计，以及FSC同I_present
迁移契约。FSC现有v2仍只是I_class成员迁移轨道，不能改名当作语义迁移完成。
类别均衡配置中的旧1/60结论只适用于成员控制，当前语义极性统计尚无准入数据。

本轮没有完成所有七份依赖配置的最终冻结，也没有通过processor/NPU/标准算法验收。
槽位校准普通检查仍遵守上次1794525361.849376之后至少600秒的约定，本轮未提前查询。

## 当前规范评测名单已物化（2026-11-13）

历史`build_slurp_evaluation_design.py`仍服务I_class/15面板旧协议，不修改其历史
身份。新`build_semantic_selection_design.py`绑定menu-v3/readout-v2、完整selection
和预物化协议快照，生成9,226录音、2,146来源组、17面板、三臂的470,526个唯一
请求，每臂156,842。产物为
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/semantic-selection-evaluation-design-v1`。

请求仅含录音/面板/实验臂/规范布局身份，名单另存音频路径、hash与来源组；不携带
native标签、语义真值或参考转写。每条音频固定51个请求，不按gold或预测改变日程；
面板分别打分再等权平均，不投票、不选最佳、不将17次请求宣称为一次完整集合输出。
候选顺序/符号重绑诊断未纳入本次规范名单，需另行冻结预算和上下文验收。

10项菜单、读出与名单测试通过；另不调用构建器逐项重建全部470,526坐标/请求ID、
9,226条白名单字段及学习/选择的录音/来源/hash隔离，核验23个依赖pin。见
[完整复核](semantic-selection-evaluation-design-check-v1.json)。report SHA256为
`890fa62f8e7e30fb5513b8e9f1a3e6b50d68b93d641e4f4b20beecc0e806cf93`。
评测v2已指向当前名单；原协议快照随产物保存，避免配置自引用hash。

本节仅完成评测身份设计，不是模型调用、语义标签或scorer验收。上文奖励分母与迁移
缺口为历史记录：当前reward-v3固定每题0.025；readout-v2与FSC-v3已有候选，分别
见doc110/111/114。新语义准入仍0，训练与评测派发保持关闭。

## 计分分母收口（2026-11-13）

当前evaluation-v2新增明确算术，FSC-v3引用相同语义计分边界；它们仍是候选协议，
不是已发布可执行实验。原selection产物中的protocol快照、report及470,526请求文件
均保持原hash；新算术不改变问题日程、输入或人口，未来执行还须绑定最终协议hash。

原生Final准确率在每个面板分别计算，再等面板平均。macro F1使用固定60类（FSC31类），
单类F1零分母取0；macro recall仅平均有gold支持的类，未支持类recall为null并报告类数。
source-equal accuracy先在每个来源组内算录音准确率，再等组平均；不替代主录音分母。
无效Final始终算错，并计入gold类FN。

语义问题只对独立准入positive/negative计分，unavailable/conflict/unassessed分别计数、
mask；bool以外的值不自动转换。无有效标签时F1为null，区别于有负标签但无TP/FP/FN时
采用zero-division0。非法正例记FN，非法负例既不是TN也不是FP。因此FPR=FP/全部负例
必须同时报告负例失败率=(FP+非法负例)/全部负例，以及输出有效率，不能用FPR0宣称可靠。
balanced accuracy=0.5×(TP/全部正例+TN/全部负例)，缺少任一极性则null。

I/S族分别按具有准入正例支持的定义等权macro；定义在17面板中各出现一次，先汇总同定义
的录音再平均定义，不等权平均尾部尺寸不同的面板macro。槽位集合EM/micro/macro只在
该面板所有槽位都有明确标签的固定录音上计算，并分别报告面板覆盖；部分标签仅计分
判决坐标。非空schema的全负样本允许正确空集EM1，缺失输出EM0；没有槽位问题的面板
是N/A，不能获得空集满分。不同面板不得拼成一次55槽位完整输出。

旧`slurp_jev_benchmark.py`报告supported-class native macro F1，类型micro F1使用
部分监督坐标；`score_slurp_jev_full.py`则为历史生成式bundle/官方SLU重构入口。
二者均不能直接当作上述新协议scorer，不原位改写历史分数。

`audit_semantic_metric_contract.py`只作为算术参考，7项测试含225种两槽位状态/输出
组合；[v2审计](semantic-metric-contract-audit-v2.json)保留边界样例，v1原样保留。
[独立算术复核](semantic-metric-contract-check-v2.json)不调用被测函数，用分数重算
原生小例及无效/缺标边界，并核验selection各文件hash未变。尚未验证实际预测解析、
完整人口计分、cluster bootstrap或模型端到端输出，independent scorer gate仍未通过。

## 请求终态与读出计分边界

evaluation-v2的prediction_ledger_contract补充计分前置条件：必须覆盖冻结名单中的
每个request_id及录音/面板/实验臂/transform。重复、额外、身份不符或仍未完成的记录
拒绝整体评分；不能把“没收齐”自动补成失败。明确的请求失败保留分母，所有节点无效，
不消费失败响应中的部分答案。成功完成但答案非法仍保留分母，不等同模型成功。

先用冻结的原始响应解析器拒绝重复JSON键及歧义外层，再按每个Span的符号表映射。
native Final的A/B与语义布尔的A/B是独立空间；不猜标签名、不自动转换整数或字符串
布尔值。节点缺失或符号非法只使该节点无效；出现未注册位置或非映射外层则整个输出
无效。Direct只具有Final，不能从Final反推过程答案。生产评分器还须验证原始证据、
模型/输入/音频/任务/协议身份，参考工具不代替这些校验。

`audit_semantic_prediction_ledger.py`以真实470,526请求名单上的合成失败记录检查完整
分母，拒绝缺失/重复/未知/错面板/未完成夹具；没有生成任何实际模型预测。16项针对
终态与符号映射的测试通过，连同算术测试共23项。另按当前17面板×3臂×3布局×60个
Final符号×2种布尔值核对18,360例映射，见
[名单审计](semantic-prediction-ledger-reference-audit-v1.json)与
[映射检查](semantic-prediction-ledger-check-v1.json)。这些用例未覆盖全部混合布尔向量，
也不是tokenizer或后端验收。原名单与输入不变，语义准入仍0，工程scorer尚未发布。

本次新增终态契约后协议hash发生变化；之前semantic-metric-contract-check-v2绑定的是
上一时点算术候选，不回写历史凭证。后续正式执行需锁定包含两部分的最终协议身份。
