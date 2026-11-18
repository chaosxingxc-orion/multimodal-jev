# 输入规则共享与评测变量分离

2026-11-13。延续任务定义→工程冻结→训练，本轮仅任务表示设计、CPU processor及映射核验。
没有模型forward、NPU训练或模型下载；历史运行快照未改。

## 规则共享的实际收益与限制

当前完整槽位规约重复供应不少逐字相同的范围、重叠和角色规则。设计probe新增显式
`shared_rules_v1`选项：重复的完整字符串（至少64字符）只存入规则库一次，原字段或
列表位置放置`{"$rule":"Rk"}`引用。引用说明要求在每个原位置、原类型/政策/边界
上下文应用完整原文。没有词内替换、规则改写、字段删除、音频/答案选规则。

审计展开可以逐字段恢复原规约，类型作用域、列表顺序与完整措辞均保留。这只是输入
字段保真，不能证明模型能理解引用或与重复原文具有相同效果。展开函数只用于审计，
不构成额外模型调用、生成解码或在线判决循环。

独立目录`panel-context-probe-v5-shared-rules`在固定CPU镜像执行与v4相同的90请求：

| 辅助Q预算 | 请求数 | 规则共享token范围 | 超8192 |
|---|---:|---:|---:|
| 0 | 30 | 5329–11082 | 27 |
| 2 | 30 | 5408–11242 | 28 |
| 8 | 30 | 5645–11721 | 28 |

最大长度14710→11721，下降20.32%；平均每请求减少1837 tokens，总token
1029664→864334。84请求变短，但无槽位的6请求增加86 tokens，因为仍带引用格式说明。
83/90仍超8192，不能把去重作为通过原上下文限制的解决方案。

全部90旧literal请求hash复现；另一份独立展开逻辑复算90输入相等；新三臂知识hash
相同，候选与问句未变；源文件hash、请求唯一性、Span定位和长度统计另行核对。
容器exit0、无NPU设备、未OOM；这里只测processor，不是模型上下文或资源验收。

候选入口：`configs/tasks/slurp/shared-rules-representation-candidate-v1.json`；
证据：`docs/research/panel-context-probe-v5-verified.json`。共享目录：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/panel-context-probe-v5-shared-rules`。

决定：`literal_v1`仍是probe默认与当前任务设计参照，引用式表示不选为主线。
不为满足8192而改变任务，selected_context_limit继续null；后续任务输入定稿后
再统一选择/验收上下文。16384仍只是可评估选项，不能从两条音频CPU长度推断NPU可用。

## 候选顺序与符号重绑定不能混为一个变量

既有评测设计包含canonical、candidate_order、neutral_symbol_rebinding三种布局。
然而`span_contract.span_labels`按候选列表位置赋予A..z0..9输出符号，直接重排候选
会同时改变展示顺序与语义到输出符号的绑定。现有`span_candidate_intervention`也已
明确标注该混合性质，不能冒充新协议要求的纯顺序对照。

新增任务级布局声明分别固定三项：

1. `display_position`：完整定义卡/候选条目的展示位置。
2. `candidate_alias`：C00..C59，定义和问句引用的中性别名；并不声称是单token。
3. `output_symbol`：A..Z、a..z、0..7，对应未来typed head读取；实际tokenizer仍待验收。

纯顺序对照只改展示位置，保持语义→别名→输出符号映射。纯重绑定保持语义展示
位置，同时一致更新定义key、所有I_class引用、候选别名、符号映射及评分逆映射。
辅助布尔Span仍使用A=false/B=true。语义ID仅离线关联与评分，不因映射审计进入actor。

对固定selection设计的60×3布局逐一核对，全部180可能类别输出均可双射还原语义。
当前按位置分配的encoder与声明的输出映射不一致数：canonical=0、order=60、
rebinding=58。这是接口不满足声明的证据，不是模型预测错误率。

同时发现旧抽象评测`q_full`将所有I放在所有S前，当前probe则交错I/S。
新任务候选明确交错I/S后接Final，45个面板/实验臂问句序列已与真实probe一致。
旧人口/请求坐标数不变，但旧分组顺序不再作为可执行当前输入。

入口：`configs/tasks/slurp/evaluation-readout-layout-candidate-v1.json`；
审计：`docs/research/evaluation-readout-axis-audit-v1.json`。
此处只补充声明及设计工具，不修改生产encoder、head或训练入口。任务冻结后必须实现
独立绑定接口，并验收所有变换的文本、完整规则、Span位置、token/head ID、上下文和
HF/FSDP/vLLM一致性，才可执行该评测。历史生成物和历史混合变换诊断保留原身份。

主目标仍未完成：新增语义监督、完整任务定义、标准RL生命周期、学术基线匹配和
持续RL提升/FSC迁移实测均待完成。本轮未缩减这些目标或解除训练派发限制。
